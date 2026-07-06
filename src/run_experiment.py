#!/usr/bin/env python3
"""Run the LLM-assisted multi-person identification pilot experiment.

The experiment uses local AMI/Simsamu audio and speaker annotations. ASR text and
LLM relabeling are real API calls, cached under results/cache/ for reproducible
reruns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_from_disk
from openai import OpenAI
from scipy import stats
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
MODEL_OUTPUT_DIR = RESULTS_DIR / "model_outputs"
FIGURES_DIR = ROOT / "figures"


ASR_FALLBACK_MODELS = [
    os.getenv("ASR_MODEL", "gpt-4o-mini-transcribe"),
    "gpt-4o-transcribe",
    "whisper-1",
]
LLM_FALLBACK_MODELS = [
    os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    "gpt-4o-mini",
    "gpt-4.1",
]


@dataclass
class RawSegment:
    dataset: str
    recording_id: str
    row_index: int
    segment_index: int
    speaker_original: str
    start: float
    end: float
    duration: float
    overlap: bool
    boundary_change: bool

    @property
    def key(self) -> str:
        return (
            f"{self.dataset}:{self.recording_id}:"
            f"{self.segment_index}:{self.start:.3f}:{self.end:.3f}"
        )


@dataclass
class TextSegment:
    dataset: str
    recording_id: str
    row_index: int
    segment_index: int
    segment_id: str
    speaker: str
    speaker_original: str
    start: float
    end: float
    duration: float
    overlap: bool
    boundary_change: bool
    text: str
    words: list[str]
    split: str


@dataclass
class ExperimentCase:
    case_id: str
    window_id: str
    dataset: str
    recording_id: str
    num_speakers: int
    noise_level: str
    base_noise: float
    seed: int
    eval_segment_ids: list[str]
    true_speakers: list[str]
    noisy_speakers: list[str]
    smoothed_speakers: list[str]


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, CACHE_DIR, MODEL_OUTPUT_DIR, FIGURES_DIR, ROOT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_jsonl_cache(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record[key_field]] = record
    return cache


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def compute_overlap_flags(starts: list[float], ends: list[float]) -> list[bool]:
    flags = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        prev_overlap = i > 0 and start < ends[i - 1]
        next_overlap = i + 1 < len(starts) and starts[i + 1] < end
        flags.append(bool(prev_overlap or next_overlap))
    return flags


def build_raw_segments(
    dataset_name: str,
    recording_id: str,
    row_index: int,
    starts: list[float],
    ends: list[float],
    speakers: list[str],
) -> list[RawSegment]:
    overlaps = compute_overlap_flags(starts, ends)
    segments: list[RawSegment] = []
    for i, (start, end, speaker) in enumerate(zip(starts, ends, speakers)):
        duration = float(end) - float(start)
        prev_change = i > 0 and speakers[i - 1] != speaker
        next_change = i + 1 < len(speakers) and speakers[i + 1] != speaker
        segments.append(
            RawSegment(
                dataset=dataset_name,
                recording_id=recording_id,
                row_index=row_index,
                segment_index=i,
                speaker_original=str(speaker),
                start=float(start),
                end=float(end),
                duration=duration,
                overlap=overlaps[i],
                boundary_change=bool(prev_change or next_change),
            )
        )
    return segments


def load_ami_rows() -> list[dict[str, Any]]:
    table = pq.read_table(ROOT / "datasets" / "ami_sdm_sample" / "sdm" / "test-00000-of-00003.parquet")
    return table.to_pylist()


def load_dataset_rows(max_ami: int, max_simsamu: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(load_ami_rows()[:max_ami]):
        audio = row["audio"]
        rec_id = Path(audio["path"]).stem
        raw_segments = build_raw_segments(
            "ami_sdm",
            rec_id,
            idx,
            row["timestamps_start"],
            row["timestamps_end"],
            row["speakers"],
        )
        rows.append({"dataset": "ami_sdm", "row_index": idx, "recording_id": rec_id, "audio": audio, "segments": raw_segments})

    simsamu = load_from_disk(str(ROOT / "datasets" / "simsamu"))["train"]
    for idx in range(min(max_simsamu, len(simsamu))):
        row = simsamu[idx]
        audio = row["audio"]
        rec_id = Path(audio["path"]).stem
        raw_segments = build_raw_segments(
            "simsamu",
            rec_id,
            idx,
            row["timestamps_start"],
            row["timestamps_end"],
            row["speakers"],
        )
        rows.append({"dataset": "simsamu", "row_index": idx, "recording_id": rec_id, "audio": audio, "segments": raw_segments})

    return rows


def select_candidate_segments(
    segments: list[RawSegment],
    min_duration: float,
    max_duration: float,
    pool_size: int,
) -> list[RawSegment]:
    candidates = [
        s
        for s in segments
        if min_duration <= s.duration <= max_duration and math.isfinite(s.duration)
    ]
    if not candidates:
        return []

    min_speakers = min(4, len(set(s.speaker_original for s in candidates)))
    best: list[RawSegment] = []
    best_score = -1.0
    for start in range(0, max(1, len(candidates) - pool_size + 1)):
        span = candidates[start : start + pool_size]
        if len(span) < min(8, pool_size):
            continue
        speaker_count = len(set(s.speaker_original for s in span))
        if speaker_count < min(2, min_speakers):
            continue
        duration_score = sum(min(s.duration, 8.0) for s in span)
        overlap_score = sum(1 for s in span if s.overlap)
        score = speaker_count * 1000.0 + duration_score + overlap_score * 2.0
        if score > best_score:
            best = span
            best_score = score
        if speaker_count >= min_speakers:
            break
    return best


def extract_wav_chunk(audio_bytes: bytes, start: float, end: float, pad: float = 0.12) -> bytes:
    """Extract a WAV chunk from embedded WAV bytes using only stdlib wave."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        start_frame = max(0, int((start - pad) * frame_rate))
        end_frame = min(total_frames, int((end + pad) * frame_rate))
        reader.setpos(start_frame)
        frames = reader.readframes(max(0, end_frame - start_frame))

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.writeframes(frames)
    return output.getvalue()


def model_dump_safe(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def transcribe_segment(
    client: OpenAI,
    audio_bytes: bytes,
    segment: RawSegment,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    sleep_s: float,
) -> dict[str, Any]:
    if segment.key in cache:
        return cache[segment.key]

    chunk_bytes = extract_wav_chunk(audio_bytes, segment.start, segment.end)
    errors: list[str] = []
    models = unique_keep_order([m for m in ASR_FALLBACK_MODELS if m])
    for model in models:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(chunk_bytes)
                tmp.flush()
                with open(tmp.name, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                    )
            text = getattr(response, "text", None)
            if text is None and isinstance(response, dict):
                text = response.get("text")
            record = {
                "segment_key": segment.key,
                "created_at": now_iso(),
                "dataset": segment.dataset,
                "recording_id": segment.recording_id,
                "segment_index": segment.segment_index,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "speaker_original": segment.speaker_original,
                "model": model,
                "text": (text or "").strip(),
                "raw_response": model_dump_safe(response),
                "errors": errors,
            }
            cache[segment.key] = record
            append_jsonl(cache_path, record)
            if sleep_s:
                time.sleep(sleep_s)
            return record
        except Exception as exc:  # API/library errors are recorded and retried.
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
            continue

    record = {
        "segment_key": segment.key,
        "created_at": now_iso(),
        "dataset": segment.dataset,
        "recording_id": segment.recording_id,
        "segment_index": segment.segment_index,
        "start": segment.start,
        "end": segment.end,
        "duration": segment.duration,
        "speaker_original": segment.speaker_original,
        "model": None,
        "text": "",
        "raw_response": None,
        "errors": errors,
    }
    cache[segment.key] = record
    append_jsonl(cache_path, record)
    return record


def build_text_window(
    row: dict[str, Any],
    candidates: list[RawSegment],
    transcriptions: dict[str, dict[str, Any]],
    eval_turns: int,
    profiles_per_speaker: int,
) -> tuple[list[TextSegment], dict[str, str]]:
    """Map original speakers to generic IDs and split segments into profile/eval."""
    nonempty: list[tuple[RawSegment, str, list[str]]] = []
    for segment in candidates:
        text = transcriptions.get(segment.key, {}).get("text", "")
        words = normalize_words(text)
        if len(words) >= 2:
            nonempty.append((segment, text.strip(), words))

    speakers_original = unique_keep_order([s.speaker_original for s, _, _ in nonempty])
    speaker_map = {speaker: f"S{i + 1}" for i, speaker in enumerate(speakers_original)}
    if len(speaker_map) < 2:
        return [], speaker_map

    profile_counts: dict[str, int] = defaultdict(int)
    profile_keys: set[str] = set()
    for segment, _, _ in nonempty:
        generic = speaker_map[segment.speaker_original]
        if profile_counts[generic] < profiles_per_speaker:
            profile_counts[generic] += 1
            profile_keys.add(segment.key)
        if all(profile_counts[f"S{i + 1}"] >= profiles_per_speaker for i in range(len(speaker_map))):
            break

    text_segments: list[TextSegment] = []
    eval_count = 0
    for segment, text, words in nonempty:
        generic = speaker_map[segment.speaker_original]
        split = "profile" if segment.key in profile_keys else "eval"
        if split == "eval":
            if eval_count >= eval_turns:
                continue
            eval_count += 1
        text_segments.append(
            TextSegment(
                dataset=segment.dataset,
                recording_id=segment.recording_id,
                row_index=segment.row_index,
                segment_index=segment.segment_index,
                segment_id=f"{segment.dataset}_{segment.recording_id}_{segment.segment_index}",
                speaker=generic,
                speaker_original=segment.speaker_original,
                start=segment.start,
                end=segment.end,
                duration=segment.duration,
                overlap=segment.overlap,
                boundary_change=segment.boundary_change,
                text=text,
                words=words,
                split=split,
            )
        )

    eval_segments = [s for s in text_segments if s.split == "eval"]
    if len(eval_segments) < min(6, eval_turns):
        return [], speaker_map
    return text_segments, speaker_map


def corrupt_speaker_labels(
    eval_segments: list[TextSegment],
    speakers: list[str],
    base_noise: float,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    labels: list[str] = []
    for i, segment in enumerate(eval_segments):
        p = base_noise
        if segment.overlap:
            p += 0.12
        if segment.duration < 1.8:
            p += 0.08
        if segment.boundary_change:
            p += 0.04
        p = min(0.85, p)
        if rng.random() < p:
            alternatives = [s for s in speakers if s != segment.speaker]
            if i > 0 and labels[-1] != segment.speaker and labels[-1] in alternatives and rng.random() < 0.55:
                labels.append(labels[-1])
            else:
                labels.append(rng.choice(alternatives))
        else:
            labels.append(segment.speaker)

    # Add a short burst error to mimic diarization clustering drift.
    if len(eval_segments) >= 5 and rng.random() < min(0.75, base_noise + 0.25):
        start = rng.randrange(0, len(eval_segments) - 2)
        length = rng.choice([2, 3])
        true_speaker = eval_segments[start].speaker
        alternatives = [s for s in speakers if s != true_speaker]
        burst_label = rng.choice(alternatives)
        for j in range(start, min(len(labels), start + length)):
            labels[j] = burst_label
    return labels


def smooth_labels(labels: list[str]) -> list[str]:
    smoothed = labels[:]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            smoothed[i] = labels[i - 1]
    return smoothed


def build_cases(
    windows: dict[str, list[TextSegment]],
    noise_levels: dict[str, float],
    seeds: list[int],
) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    for window_id, segments in windows.items():
        eval_segments = [s for s in segments if s.split == "eval"]
        speakers = sorted(set(s.speaker for s in segments))
        for noise_name, base_noise in noise_levels.items():
            for seed in seeds:
                noisy = corrupt_speaker_labels(eval_segments, speakers, base_noise, seed)
                smoothed = smooth_labels(noisy)
                case_id = stable_hash(
                    {
                        "window_id": window_id,
                        "noise": noise_name,
                        "base": base_noise,
                        "seed": seed,
                    }
                )
                cases.append(
                    ExperimentCase(
                        case_id=case_id,
                        window_id=window_id,
                        dataset=eval_segments[0].dataset,
                        recording_id=eval_segments[0].recording_id,
                        num_speakers=len(speakers),
                        noise_level=noise_name,
                        base_noise=base_noise,
                        seed=seed,
                        eval_segment_ids=[s.segment_id for s in eval_segments],
                        true_speakers=[s.speaker for s in eval_segments],
                        noisy_speakers=noisy,
                        smoothed_speakers=smoothed,
                    )
                )
    return cases


def prompt_for_case(
    case: ExperimentCase,
    segments: list[TextSegment],
    use_profiles: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    speakers = sorted(set(s.speaker for s in segments))
    profile_segments = [s for s in segments if s.split == "profile"]
    eval_segments = [s for s in segments if s.split == "eval"]
    noisy_by_id = dict(zip(case.eval_segment_ids, case.noisy_speakers))

    profiles: dict[str, list[str]] = defaultdict(list)
    if use_profiles:
        for segment in profile_segments:
            if len(profiles[segment.speaker]) < 3:
                profiles[segment.speaker].append(segment.text)

    eval_payload = [
        {
            "id": segment.segment_id,
            "order": i,
            "noisy_speaker": noisy_by_id[segment.segment_id],
            "duration_seconds": round(segment.duration, 2),
            "overlap_flag": segment.overlap,
            "text": segment.text,
        }
        for i, segment in enumerate(eval_segments)
    ]

    user_payload: dict[str, Any] = {
        "task": "Correct speaker IDs for ASR transcript segments. Noisy speaker labels may be wrong.",
        "allowed_speakers": speakers,
        "rules": [
            "Return JSON only.",
            "Use only speaker IDs from allowed_speakers.",
            "Return exactly one assignment for every segment id.",
            "Do not rewrite or summarize transcript text.",
            "Change a noisy speaker only when dialogue context or speaker profile evidence supports it.",
        ],
        "segments": eval_payload,
        "output_schema": {"assignments": [{"id": "segment id", "speaker": "one allowed speaker id"}]},
    }
    if use_profiles:
        user_payload["speaker_profiles_from_high_confidence_context"] = profiles
    else:
        user_payload["speaker_profiles_from_high_confidence_context"] = "not provided"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful speaker-attribution correction system for "
                "multi-speaker ASR transcripts. Your job is only to choose "
                "speaker IDs, not to edit words."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, indent=2)},
    ]
    metadata = {
        "case_id": case.case_id,
        "window_id": case.window_id,
        "condition": "llm_profile" if use_profiles else "llm_no_profile",
        "prompt_hash": stable_hash(messages),
        "allowed_speakers": speakers,
    }
    return messages, metadata


def call_llm(
    client: OpenAI,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    sleep_s: float,
) -> dict[str, Any]:
    cache_key = metadata["prompt_hash"]
    if cache_key in cache:
        return cache[cache_key]

    errors: list[str] = []
    models = unique_keep_order([m for m in LLM_FALLBACK_MODELS if m])
    for model in models:
        attempts = [
            {"temperature": 0, "response_format": {"type": "json_object"}},
            {"response_format": {"type": "json_object"}},
            {"temperature": 0},
            {},
        ]
        for kwargs in attempts:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                usage = model_dump_safe(getattr(response, "usage", None))
                record = {
                    "prompt_hash": cache_key,
                    "created_at": now_iso(),
                    "model": model,
                    "metadata": metadata,
                    "content": content,
                    "usage": usage,
                    "raw_response_id": getattr(response, "id", None),
                    "errors": errors,
                }
                cache[cache_key] = record
                append_jsonl(cache_path, record)
                if sleep_s:
                    time.sleep(sleep_s)
                return record
            except Exception as exc:
                errors.append(f"{model} {kwargs}: {type(exc).__name__}: {exc}")
                continue

    record = {
        "prompt_hash": cache_key,
        "created_at": now_iso(),
        "model": None,
        "metadata": metadata,
        "content": "",
        "usage": None,
        "raw_response_id": None,
        "errors": errors,
    }
    cache[cache_key] = record
    append_jsonl(cache_path, record)
    return record


def parse_llm_assignments(
    record: dict[str, Any],
    case: ExperimentCase,
    allowed_speakers: list[str],
) -> tuple[list[str], dict[str, Any]]:
    noisy_by_id = dict(zip(case.eval_segment_ids, case.noisy_speakers))
    diagnostics = {"parse_ok": False, "invalid_ids": [], "invalid_speakers": [], "missing_ids": []}
    predicted = noisy_by_id.copy()
    try:
        parsed = json.loads(record.get("content", "") or "{}")
        assignments = parsed.get("assignments", [])
        if not isinstance(assignments, list):
            raise ValueError("assignments is not a list")
        for item in assignments:
            if not isinstance(item, dict):
                continue
            seg_id = str(item.get("id", ""))
            speaker = str(item.get("speaker", ""))
            if seg_id not in predicted:
                diagnostics["invalid_ids"].append(seg_id)
                continue
            if speaker not in allowed_speakers:
                diagnostics["invalid_speakers"].append({"id": seg_id, "speaker": speaker})
                continue
            predicted[seg_id] = speaker
        diagnostics["missing_ids"] = [sid for sid in case.eval_segment_ids if sid not in predicted]
        diagnostics["parse_ok"] = not diagnostics["missing_ids"] and not diagnostics["invalid_speakers"]
    except Exception as exc:
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"

    return [predicted[sid] for sid in case.eval_segment_ids], diagnostics


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def expand_words(eval_segments: list[TextSegment], labels: list[str]) -> tuple[list[str], list[str]]:
    words: list[str] = []
    speakers: list[str] = []
    for segment, speaker in zip(eval_segments, labels):
        words.extend(segment.words)
        speakers.extend([speaker] * len(segment.words))
    return words, speakers


def wder(eval_segments: list[TextSegment], pred_labels: list[str]) -> float:
    true_words, true_speakers = expand_words(eval_segments, [s.speaker for s in eval_segments])
    _, pred_speakers = expand_words(eval_segments, pred_labels)
    if not true_words:
        return float("nan")
    errors = sum(t != p for t, p in zip(true_speakers, pred_speakers))
    return errors / len(true_words)


def cpwer(eval_segments: list[TextSegment], pred_labels: list[str]) -> float:
    ref_by_speaker: dict[str, list[str]] = defaultdict(list)
    hyp_by_speaker: dict[str, list[str]] = defaultdict(list)
    for segment, pred_speaker in zip(eval_segments, pred_labels):
        ref_by_speaker[segment.speaker].extend(segment.words)
        hyp_by_speaker[pred_speaker].extend(segment.words)

    ref_speakers = sorted(ref_by_speaker)
    hyp_speakers = sorted(hyp_by_speaker)
    denom = sum(len(ref_by_speaker[s]) for s in ref_speakers)
    if denom == 0:
        return float("nan")
    if len(hyp_speakers) == 0:
        return 1.0

    # Speaker counts are small in this pilot, so exact permutation is simple.
    if len(hyp_speakers) <= len(ref_speakers):
        hyp_candidates = hyp_speakers + [f"__empty_{i}" for i in range(len(ref_speakers) - len(hyp_speakers))]
        best = math.inf
        for perm in itertools.permutations(hyp_candidates, len(ref_speakers)):
            dist = 0
            used = set()
            for ref_speaker, hyp_speaker in zip(ref_speakers, perm):
                used.add(hyp_speaker)
                dist += edit_distance(ref_by_speaker[ref_speaker], hyp_by_speaker.get(hyp_speaker, []))
            best = min(best, dist)
        return best / denom

    best = math.inf
    for perm in itertools.permutations(hyp_speakers, len(ref_speakers)):
        used = set(perm)
        dist = 0
        for ref_speaker, hyp_speaker in zip(ref_speakers, perm):
            dist += edit_distance(ref_by_speaker[ref_speaker], hyp_by_speaker[hyp_speaker])
        for hyp_speaker in hyp_speakers:
            if hyp_speaker not in used:
                dist += len(hyp_by_speaker[hyp_speaker])
        best = min(best, dist)
    return best / denom


def compute_metrics_for_case(
    case: ExperimentCase,
    segments: list[TextSegment],
    method_predictions: dict[str, tuple[list[str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    eval_segments = [s for s in segments if s.split == "eval"]
    true_labels = [s.speaker for s in eval_segments]
    true_words, _ = expand_words(eval_segments, true_labels)
    rows: list[dict[str, Any]] = []
    for method, (pred_labels, diagnostics) in method_predictions.items():
        rows.append(
            {
                "case_id": case.case_id,
                "window_id": case.window_id,
                "dataset": case.dataset,
                "recording_id": case.recording_id,
                "num_speakers": case.num_speakers,
                "noise_level": case.noise_level,
                "base_noise": case.base_noise,
                "seed": case.seed,
                "method": method,
                "wder": wder(eval_segments, pred_labels),
                "cpwer": cpwer(eval_segments, pred_labels),
                "speaker_count_mae": abs(len(set(pred_labels)) - len(set(true_labels))),
                "segment_error_rate": sum(t != p for t, p in zip(true_labels, pred_labels)) / len(true_labels),
                "changed_from_noisy_rate": sum(n != p for n, p in zip(case.noisy_speakers, pred_labels)) / len(pred_labels),
                "n_eval_segments": len(eval_segments),
                "n_words": len(true_words),
                "parse_ok": diagnostics.get("parse_ok", True),
                "parse_error": diagnostics.get("error", ""),
                "invalid_speaker_count": len(diagnostics.get("invalid_speakers", [])),
                "model": diagnostics.get("model", ""),
                "prompt_tokens": diagnostics.get("prompt_tokens", 0),
                "completion_tokens": diagnostics.get("completion_tokens", 0),
            }
        )
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def statistical_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    noisy = metrics[metrics["method"] == "noisy_labels"][
        ["case_id", "wder", "dataset", "num_speakers", "noise_level"]
    ].rename(columns={"wder": "wder_noisy"})
    for method in sorted(set(metrics["method"]) - {"noisy_labels"}):
        current = metrics[metrics["method"] == method][["case_id", "wder"]].rename(columns={"wder": "wder_method"})
        paired = noisy.merge(current, on="case_id")
        diff = (paired["wder_noisy"] - paired["wder_method"]).to_numpy(dtype=float)
        if len(diff) == 0:
            continue
        normal_p = stats.shapiro(diff).pvalue if 3 <= len(diff) <= 5000 and np.std(diff) > 0 else float("nan")
        if len(diff) >= 3 and np.std(diff) > 0 and (math.isnan(normal_p) or normal_p >= 0.05):
            test_name = "paired_t"
            stat, p_value = stats.ttest_rel(paired["wder_noisy"], paired["wder_method"])
        elif len(diff) >= 1 and np.any(diff != 0):
            test_name = "wilcoxon"
            stat, p_value = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        else:
            test_name = "no_variance"
            stat, p_value = 0.0, 1.0
        ci_low, ci_high = bootstrap_ci(diff)
        dz = float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else float("nan")
        rows.append(
            {
                "comparison": f"noisy_labels_vs_{method}",
                "method": method,
                "n_pairs": len(diff),
                "mean_wder_noisy": float(paired["wder_noisy"].mean()),
                "mean_wder_method": float(paired["wder_method"].mean()),
                "mean_wder_reduction": float(np.mean(diff)),
                "relative_wder_reduction": float(np.mean(diff) / paired["wder_noisy"].mean())
                if paired["wder_noisy"].mean() > 0
                else float("nan"),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "test": test_name,
                "statistic": float(stat),
                "p_value": float(p_value),
                "normality_p": float(normal_p) if not math.isnan(normal_p) else float("nan"),
                "cohens_dz": dz,
            }
        )
    if rows:
        adjusted = holm_adjust([r["p_value"] for r in rows])
        for row, adj in zip(rows, adjusted):
            row["holm_p_value"] = adj
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["dataset", "num_speakers", "noise_level", "method"], as_index=False)
        .agg(
            mean_wder=("wder", "mean"),
            std_wder=("wder", "std"),
            mean_cpwer=("cpwer", "mean"),
            mean_segment_error=("segment_error_rate", "mean"),
            mean_speaker_count_mae=("speaker_count_mae", "mean"),
            parse_ok_rate=("parse_ok", "mean"),
            n_cases=("case_id", "nunique"),
            n_words=("n_words", "sum"),
        )
        .sort_values(["dataset", "noise_level", "mean_wder"])
    )


def write_plots(metrics: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    method_order = ["noisy_labels", "continuity_smoothing", "llm_no_profile", "llm_profile"]

    plt.figure(figsize=(11, 5.8))
    sns.barplot(
        data=metrics,
        x="noise_level",
        y="wder",
        hue="method",
        order=["low", "medium", "high"],
        hue_order=[m for m in method_order if m in set(metrics["method"])],
        errorbar=("ci", 95),
    )
    plt.ylabel("Word diarization error rate")
    plt.xlabel("Injected diarization noise level")
    plt.title("Speaker-attribution error by correction method")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "wder_by_method.png", dpi=180)
    plt.close()

    noisy = metrics[metrics["method"] == "noisy_labels"][["case_id", "wder"]].rename(columns={"wder": "noisy_wder"})
    delta = metrics.merge(noisy, on="case_id")
    delta = delta[delta["method"] != "noisy_labels"].copy()
    delta["wder_reduction"] = delta["noisy_wder"] - delta["wder"]
    plt.figure(figsize=(10, 5.8))
    sns.barplot(
        data=delta,
        x="dataset",
        y="wder_reduction",
        hue="method",
        hue_order=[m for m in method_order if m != "noisy_labels" and m in set(delta["method"])],
        errorbar=("ci", 95),
    )
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("WDER reduction vs noisy labels")
    plt.xlabel("Dataset family")
    plt.title("Improvement by dataset and speaker count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "wder_reduction_by_dataset.png", dpi=180)
    plt.close()

    parse = aggregate[aggregate["method"].str.startswith("llm")].copy()
    if not parse.empty:
        plt.figure(figsize=(8, 4.8))
        sns.barplot(data=parse, x="method", y="parse_ok_rate", hue="dataset", errorbar=None)
        plt.ylim(0, 1.05)
        plt.ylabel("Parse success rate")
        plt.xlabel("LLM condition")
        plt.title("LLM JSON output validity")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "llm_parse_success.png", dpi=180)
        plt.close()


def collect_environment(config: dict[str, Any]) -> dict[str, Any]:
    packages = {}
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "openai", "matplotlib", "seaborn", "datasets", "pyarrow"]:
        try:
            import importlib.metadata as metadata

            packages[pkg] = metadata.version(pkg)
        except Exception:
            packages[pkg] = "unknown"

    gpu_cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"]
    try:
        gpu = subprocess.run(gpu_cmd, capture_output=True, text=True, check=False).stdout.strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"

    return {
        "created_at": now_iso(),
        "python": sys.version,
        "packages": packages,
        "gpu": gpu,
        "workspace": str(ROOT),
        "config": config,
    }


def save_dataset_summary(rows: list[dict[str, Any]], windows: dict[str, list[TextSegment]]) -> None:
    summary_rows = []
    for row in rows:
        segments = row["segments"]
        durations = [s.duration for s in segments]
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "recording_id": row["recording_id"],
                "n_segments": len(segments),
                "n_speakers": len(set(s.speaker_original for s in segments)),
                "mean_duration": float(np.mean(durations)) if durations else float("nan"),
                "overlap_segment_rate": float(np.mean([s.overlap for s in segments])) if segments else float("nan"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    with (RESULTS_DIR / "selected_windows.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                window_id: [asdict(segment) for segment in segments]
                for window_id, segments in windows.items()
            },
            f,
            indent=2,
            ensure_ascii=True,
        )


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for real ASR and LLM API calls.")

    client = OpenAI()
    config = vars(args).copy()
    config["asr_model_fallbacks"] = unique_keep_order([m for m in ASR_FALLBACK_MODELS if m])
    config["llm_model_fallbacks"] = unique_keep_order([m for m in LLM_FALLBACK_MODELS if m])

    env = collect_environment(config)
    with (RESULTS_DIR / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=True)

    rows = load_dataset_rows(args.max_ami_windows, args.max_simsamu_windows)
    asr_cache_path = CACHE_DIR / "asr_transcriptions.jsonl"
    asr_cache = load_jsonl_cache(asr_cache_path, "segment_key")

    selected_raw: dict[str, list[RawSegment]] = {}
    transcription_records: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidates = select_candidate_segments(
            row["segments"],
            args.min_duration,
            args.max_duration,
            args.pool_size,
        )
        if not candidates:
            continue
        window_id = f"{row['dataset']}_{row['recording_id']}"
        selected_raw[window_id] = candidates
        audio_bytes = row["audio"]["bytes"]
        for segment in candidates:
            record = transcribe_segment(
                client=client,
                audio_bytes=audio_bytes,
                segment=segment,
                cache=asr_cache,
                cache_path=asr_cache_path,
                sleep_s=args.api_sleep,
            )
            transcription_records[segment.key] = record
            print(
                f"ASR {window_id} seg={segment.segment_index} "
                f"model={record.get('model')} words={len(normalize_words(record.get('text', '')))}",
                flush=True,
            )

    windows: dict[str, list[TextSegment]] = {}
    speaker_maps: dict[str, dict[str, str]] = {}
    row_by_window = {f"{row['dataset']}_{row['recording_id']}": row for row in rows}
    for window_id, candidates in selected_raw.items():
        text_segments, speaker_map = build_text_window(
            row_by_window[window_id],
            candidates,
            transcription_records,
            eval_turns=args.eval_turns,
            profiles_per_speaker=args.profiles_per_speaker,
        )
        if text_segments:
            windows[window_id] = text_segments
            speaker_maps[window_id] = speaker_map
            n_eval = sum(1 for s in text_segments if s.split == "eval")
            print(f"WINDOW {window_id}: speakers={len(speaker_map)} eval_turns={n_eval}", flush=True)
        else:
            print(f"WINDOW {window_id}: skipped after transcription filtering", flush=True)

    if not windows:
        raise RuntimeError("No usable windows after ASR transcription.")

    save_dataset_summary(rows, windows)
    with (RESULTS_DIR / "speaker_maps.json").open("w", encoding="utf-8") as f:
        json.dump(speaker_maps, f, indent=2, ensure_ascii=True)

    cases = build_cases(
        windows,
        noise_levels={"low": 0.10, "medium": 0.25, "high": 0.40},
        seeds=args.noise_seeds,
    )
    with (RESULTS_DIR / "experiment_cases.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(case) for case in cases], f, indent=2, ensure_ascii=True)

    llm_cache_path = MODEL_OUTPUT_DIR / "llm_outputs.jsonl"
    llm_cache = load_jsonl_cache(llm_cache_path, "prompt_hash")
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for case in cases:
        segments = windows[case.window_id]
        allowed_speakers = sorted(set(s.speaker for s in segments))
        method_predictions: dict[str, tuple[list[str], dict[str, Any]]] = {
            "noisy_labels": (case.noisy_speakers, {"parse_ok": True}),
            "continuity_smoothing": (case.smoothed_speakers, {"parse_ok": True}),
        }

        for use_profiles in [False, True]:
            condition = "llm_profile" if use_profiles else "llm_no_profile"
            messages, metadata = prompt_for_case(case, segments, use_profiles)
            record = call_llm(
                client=client,
                messages=messages,
                metadata=metadata,
                cache=llm_cache,
                cache_path=llm_cache_path,
                sleep_s=args.api_sleep,
            )
            pred, diagnostics = parse_llm_assignments(record, case, allowed_speakers)
            diagnostics["model"] = record.get("model") or ""
            usage = record.get("usage") or {}
            if isinstance(usage, dict):
                diagnostics["prompt_tokens"] = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                diagnostics["completion_tokens"] = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            method_predictions[condition] = (pred, diagnostics)
            prediction_rows.append(
                {
                    "case_id": case.case_id,
                    "window_id": case.window_id,
                    "condition": condition,
                    "model": record.get("model"),
                    "prompt_hash": record.get("prompt_hash"),
                    "parse_diagnostics": diagnostics,
                    "predicted_speakers": pred,
                }
            )
            print(
                f"LLM {condition} case={case.case_id} model={record.get('model')} "
                f"parse_ok={diagnostics.get('parse_ok')}",
                flush=True,
            )

        metric_rows.extend(compute_metrics_for_case(case, segments, method_predictions))

    with (RESULTS_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "metrics_by_case.csv", index=False)

    aggregate = aggregate_metrics(metrics)
    aggregate.to_csv(RESULTS_DIR / "aggregate_metrics.csv", index=False)

    tests = statistical_tests(metrics)
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    write_plots(metrics, aggregate)

    output_hashes = {}
    for path in [
        RESULTS_DIR / "metrics_by_case.csv",
        RESULTS_DIR / "aggregate_metrics.csv",
        RESULTS_DIR / "statistical_tests.csv",
        RESULTS_DIR / "experiment_cases.json",
        RESULTS_DIR / "selected_windows.json",
        MODEL_OUTPUT_DIR / "llm_outputs.jsonl",
        CACHE_DIR / "asr_transcriptions.jsonl",
    ]:
        if path.exists():
            output_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (RESULTS_DIR / "reproducibility_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(output_hashes, f, indent=2, ensure_ascii=True)

    print("\nAggregate metrics:")
    print(aggregate.to_string(index=False))
    print("\nStatistical tests:")
    print(tests.to_string(index=False) if not tests.empty else "No tests computed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ami-windows", type=int, default=3)
    parser.add_argument("--max-simsamu-windows", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=18)
    parser.add_argument("--eval-turns", type=int, default=10)
    parser.add_argument("--profiles-per-speaker", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=1.2)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--api-sleep", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
