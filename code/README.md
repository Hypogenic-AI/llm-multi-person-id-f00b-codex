# Cloned Repositories

All repositories were cloned as shallow copies under `code/`.

## google_speaker_id

- URL: https://github.com/google/speaker-id
- Location: `code/google_speaker_id/`
- Main relevant path: `code/google_speaker_id/DiarizationLM/`
- Purpose: official DiarizationLM utilities for transcript-preserving speaker transfer, prompt/completion preparation, completion parsing, and WER/WDER/cpWER metrics.
- Key files:
  - `DiarizationLM/train_data_prep.py`
  - `DiarizationLM/postprocess_completions.py`
  - `DiarizationLM/compute_metrics_on_json.py`
  - `DiarizationLM/diarizationlm/utils.py`
  - `DiarizationLM/diarizationlm/metrics.py`
- Validation:

```bash
PYTHONPATH=code/google_speaker_id/DiarizationLM \
python code/google_speaker_id/DiarizationLM/compute_metrics_on_json.py \
  --input=code/google_speaker_id/DiarizationLM/testdata/example_data.json \
  --output=artifacts/diarizationlm_example_metrics.json
```

This completed successfully. Example aggregate metrics: WER 0.2363, WDER 0.0437, cpWER 0.2794.

## llm_diarize_asr_agnostic

- URL: https://github.com/GeorgeEfstathiadis/LLM-Diarize-ASR-Agnostic
- Location: `code/llm_diarize_asr_agnostic/`
- Purpose: code for LLM-based speaker diarization correction across ASR providers.
- Key files:
  - `preprocess/*.ipynb`
  - `fine_tuning/train.ipynb`
  - `evaluation/*.ipynb`
  - `utils/metrics.py`
  - `diarization_utils/utils.py`
- Notes: expects ASR transcripts with word-level timestamps and speaker IDs. Supports AWS Transcribe, Azure Speech to Text, Google Speech to Text, and WhisperX-style outputs.

## diarizers

- URL: https://github.com/huggingface/diarizers
- Location: `code/diarizers/`
- Purpose: fine-tuning pyannote speaker segmentation models on HF diarization datasets.
- Key files:
  - `train_segmentation.py`
  - `test_segmentation.py`
  - `datasets/spd_datasets.py`
  - `datasets/README.md`
- Notes: useful for adapting pyannote segmentation to AMI, CallHome, Simsamu, or other HF diarization datasets. Requires HF access for gated pyannote/CallHome assets.

## pyannote_audio

- URL: https://github.com/pyannote/pyannote-audio
- Location: `code/pyannote_audio/`
- Purpose: speaker diarization baseline toolkit.
- Notes: current README points to `pyannote/speaker-diarization-community-1`; requires accepting HF model terms, an HF token, and ffmpeg/torchcodec for audio decoding.

## whisperx

- URL: https://github.com/m-bain/whisperX
- Location: `code/whisperx/`
- Purpose: ASR with word-level timestamps, forced alignment, VAD batching, and pyannote-based speaker labels.
- Key use: produces the kind of ASR+word-timestamp inputs needed by DiarizationLM and LLM correction methods.
- Notes: diarization requires HF token and model terms for pyannote. CPU usage is possible but slow.

## meeteval

- URL: https://github.com/fgnt/meeteval
- Location: `code/meeteval/`
- Purpose: cpWER, tcpWER, ORC-WER, MIMO-WER, and DER utilities for meeting transcription evaluation.
- Key files:
  - `example_files/`
  - `meeteval/wer/`
  - `meeteval/der/`
- Validation status: not installed in the local venv because the package requires compiled Cython extensions and this environment has no `c++`, `g++`, or `gcc` available. The cloned code and examples are available; installing a C++ compiler should allow `uv add meeteval` or `pip install -e code/meeteval`.
