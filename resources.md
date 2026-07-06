# Resources Catalog

## Summary

This document catalogs the resources gathered for the project: papers, datasets, code repositories, and practical notes for the experiment runner.

## Papers

Total papers downloaded: 15

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| DiarizationLM | Wang et al. | 2024 | `papers/2401.03506_diarizationlm.pdf` | LLM post-processing; WDER reductions on Fisher/CallHome. |
| Enhancing Speaker Diarization with LLMs | Park et al. | 2023 | `papers/2309.05248_enhancing_speaker_diarization_llms.pdf` | Contextual beam search with LLM lexical cues. |
| Advancing Multi-talker ASR with LLMs | Shi et al. | 2024 | `papers/2408.17431_multi_talker_asr_llms.pdf` | LLM decoder for SOT multi-talker ASR. |
| Diarization-Aware MS-ASR via LLMs | Lin et al. | 2025 | `papers/2506.05796_diarization_aware_ms_asr_llms.pdf` | Speaker/time triplet conditioning. |
| DM-ASR | Li et al. | 2026 | `papers/2604.22467_dm_asr.pdf` | Multi-turn diarization-aware LLM ASR. |
| Balancing ASR and diarization | Zheng et al. | 2026 | `papers/2606.13095_balancing_asr_diarization_llms.pdf` | Compact end-to-end LLM with ASR/speaker losses. |
| SpeakerLM | Yin et al. | 2025 | `papers/2508.06372_speakerlm.pdf` | Unified SDR with speaker registration. |
| JEDIS-LLM | Shi et al. | 2025 | `papers/2511.16046_jedis_llm.pdf` | Streamable joint ASR/diarization with speaker cache. |
| Language Modelling for Speaker Diarization | India et al. | 2025 | `papers/2501.17893_language_modelling_diarization.pdf` | Linguistic plus acoustic speaker diarization. |
| Pretraining Multi-Speaker Identification | Horiguchi et al. | 2025 | `papers/2505.24545_pretraining_multi_speaker_identification.pdf` | Speaker-ID pretraining for neural diarization. |
| Long Conversation ASR and Diarization | Mao et al. | 2020 | `papers/2005.08072_long_conversation_asr_diarization.pdf` | Long-form benchmark framing. |
| MeetEval | von Neumann et al. | 2023 | `papers/2307.11394_meeteval.pdf` | cpWER/tcpWER evaluation. |
| pyannote.audio | Bredin et al. | 2019 | `papers/1911.01255_pyannote_audio.pdf` | Open diarization toolkit. |
| Pretrained LMs for Speaker Identification | Zamana et al. | 2024 | `papers/zamana_2024_pretrained_lms_speaker_identification.pdf` | Text-based named speaker identification. |
| pyannote.audio 2.1 | Bredin | 2023 | `papers/bredin_2023_pyannote_audio_2_1.pdf` | Practical pyannote pipeline recipe. |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Simsamu | HF `diarizers-community/simsamu` | 171 MB | Speaker diarization | `datasets/simsamu/` | Full small dataset; 61 rows. |
| AMI SDM sample shard | HF `diarizers-community/ami` | 313 MB | Meeting diarization/ASR | `datasets/ami_sdm_sample/` | One English SDM test shard; 6 meetings. |

See `datasets/README.md` for loading and full-download instructions.

## Code Repositories

Total repositories cloned: 6

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| google_speaker_id | https://github.com/google/speaker-id | DiarizationLM official code | `code/google_speaker_id/` | Metrics smoke test passed. |
| LLM-Diarize-ASR-Agnostic | https://github.com/GeorgeEfstathiadis/LLM-Diarize-ASR-Agnostic | ASR-agnostic LLM correction | `code/llm_diarize_asr_agnostic/` | Notebook workflow for preprocessing/fine-tuning/eval. |
| diarizers | https://github.com/huggingface/diarizers | Fine-tune pyannote segmentation | `code/diarizers/` | Needs HF tokens for gated models/datasets. |
| pyannote-audio | https://github.com/pyannote/pyannote-audio | Diarization baseline | `code/pyannote_audio/` | Needs HF terms/token and ffmpeg. |
| WhisperX | https://github.com/m-bain/whisperX | ASR, word timestamps, diarization | `code/whisperx/` | Good cascade front-end for DiarizationLM inputs. |
| MeetEval | https://github.com/fgnt/meeteval | cpWER/tcpWER/DER evaluation | `code/meeteval/` | Local install blocked by missing C++ compiler. |

See `code/README.md` for detailed notes.

## Resource Gathering Notes

### Search Strategy

The local paper-finder service was attempted first with a diligent query but did not return output. Manual search then used arXiv, Semantic Scholar, Papers with Code/GitHub, Hugging Face datasets, and paper references. Searches focused on "LLM speaker diarization", "multi-talker ASR LLM", "speaker-attributed ASR", "diarization-aware ASR", and "meeting transcription evaluation".

### Selection Criteria

- Direct relevance to "who spoke what/when" in multi-speaker audio.
- LLM, pretrained language model, or text-context contribution to speaker identification.
- Availability of PDF, dataset, or code.
- Preference for meeting and many-speaker benchmarks over only two-speaker telephone settings.

### Challenges Encountered

- Paper-finder client stalled and was terminated; manual search was used.
- CallHome on Hugging Face is gated and requires authentication.
- Full AMI, VoxConverse, and ICSI processed datasets are multi-GB; only a representative AMI shard was downloaded.
- MeetEval installation failed because this environment lacks `c++`, `g++`, and `gcc`.
- `word-levenshtein` failed to build when attempted, but DiarizationLM's example metrics script ran successfully without it after installing `absl-py`, `scipy`, and `numba`.

### Gaps and Workarounds

- Full many-speaker English training/evaluation should use AMI SDM/IHM and ICSI once storage/time are approved.
- For immediate local experiments, use `datasets/ami_sdm_sample` and `datasets/simsamu`.
- Use DiarizationLM metrics now; install a compiler before using MeetEval tcpWER.
- If HF credentials are available, accept pyannote and CallHome terms before running pyannote or CallHome downloads.

## Recommendations for Experiment Design

1. Primary dataset: AMI SDM for English many-speaker meeting audio; start with the downloaded sample shard, then scale to full AMI.
2. Secondary datasets: ICSI for more speakers, CallHome for comparison with DiarizationLM, Simsamu for quick loader sanity checks.
3. Baselines: WhisperX + pyannote timestamp assignment, DiarizationLM post-processing, and optionally pyannote fine-tuned with `diarizers`.
4. Metrics: WDER and cpWER immediately via DiarizationLM code; add DER and tcpWER via MeetEval after installing a C++ compiler.
5. Main experiment: test whether LLM correction reduces speaker-word assignment errors as number of speakers, overlap ratio, and ASR timestamp noise increase.
