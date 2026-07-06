# Downloaded Papers

This directory contains the PDF papers gathered for the project. Original PDFs are at the top level. `papers/pages/` contains three-page chunks for the deepest-read LLM diarization and multi-speaker ASR papers.

| # | Paper | Authors | Year | File | Why relevant |
|---|-------|---------|------|------|--------------|
| 1 | [DiarizationLM: Speaker Diarization Post-Processing with Large Language Models](https://arxiv.org/abs/2401.03506) | Wang, Huang, Zhao, Clark, Xia, Liao | 2024 | `2401.03506_diarizationlm.pdf` | Directly tests LLM post-processing of ASR plus diarization outputs; reports large WDER reductions. |
| 2 | [Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach](https://arxiv.org/abs/2309.05248) | Park, Dhawan, Koluguri, Balam | 2023 | `2309.05248_enhancing_speaker_diarization_llms.pdf` | Uses LLM lexical probabilities during diarization decoding. |
| 3 | [Advancing Multi-talker ASR Performance with Large Language Models](https://arxiv.org/abs/2408.17431) | Shi et al. | 2024 | `2408.17431_multi_talker_asr_llms.pdf` | LLM decoder for serialized-output multi-talker ASR on LibriMix and AMI. |
| 4 | [Diarization-Aware Multi-Speaker Automatic Speech Recognition via Large Language Models](https://arxiv.org/abs/2506.05796) | Lin, Cheng, Li, Tang, Li | 2025 | `2506.05796_diarization_aware_ms_asr_llms.pdf` | Uses diarization-derived speaker/time triplets with an LLM ASR backend. |
| 5 | [DM-ASR: Diarization-aware Multi-speaker ASR with Large Language Models](https://arxiv.org/abs/2604.22467) | Li, Cheng, Zhu, Wang, Liu, Li | 2026 | `2604.22467_dm_asr.pdf` | Reformulates diarization-guided transcription as multi-turn LLM generation. |
| 6 | [Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition](https://arxiv.org/abs/2606.13095) | Zheng et al. | 2026 | `2606.13095_balancing_asr_diarization_llms.pdf` | Latest compact end-to-end Speech-LLM work balancing ASR and speaker ID losses. |
| 7 | [SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](https://arxiv.org/abs/2508.06372) | Yin et al. | 2025 | `2508.06372_speakerlm.pdf` | Unified end-to-end "who spoke when and what" model with speaker registration modes. |
| 8 | [Train Short, Infer Long: Speech-LLM Enables Zero-Shot Streamable Joint ASR and Diarization on Long Audio](https://arxiv.org/abs/2511.16046) | Shi, Xiao, Fan, Ling, Li | 2025 | `2511.16046_jedis_llm.pdf` | Streamable joint ASR and diarization using a Speaker Prompt Cache. |
| 9 | [Language Modelling for Speaker Diarization in Telephonic Interviews](https://arxiv.org/abs/2501.17893) | India, Hernando, Fonollosa | 2025 | `2501.17893_language_modelling_diarization.pdf` | Shows linguistic features can outperform acoustic-only diarization in structured calls. |
| 10 | [Pretraining Multi-Speaker Identification for Neural Speaker Diarization](https://arxiv.org/abs/2505.24545) | Horiguchi, Ando, Delcroix, Tawara | 2025 | `2505.24545_pretraining_multi_speaker_identification.pdf` | Relevant speaker-identification pretraining for robust neural diarization. |
| 11 | [Speech Recognition and Multi-Speaker Diarization of Long Conversations](https://arxiv.org/abs/2005.08072) | Mao, Li, McAuley, Cottrell | 2020 | `2005.08072_long_conversation_asr_diarization.pdf` | Long-form multi-speaker benchmark framing and joint ASR/diarization baseline. |
| 12 | [MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems](https://arxiv.org/abs/2307.11394) | von Neumann, Boeddeker, Delcroix, Haeb-Umbach | 2023 | `2307.11394_meeteval.pdf` | Defines/reports cpWER and tcpWER tooling used in recent meeting ASR papers. |
| 13 | [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255) | Bredin et al. | 2019 | `1911.01255_pyannote_audio.pdf` | Open-source diarization baseline toolkit. |
| 14 | [Using Pretrained Language Models for Improved Speaker Identification](https://www.isca-archive.org/odyssey_2024/zamana24_odyssey.html) | Zamana, Kaard, Alumae | 2024 | `zamana_2024_pretrained_lms_speaker_identification.pdf` | Text/LLM-based named speaker identification from transcripts. |
| 15 | [pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html) | Bredin | 2023 | `bredin_2023_pyannote_audio_2_1.pdf` | Practical baseline recipe and benchmark numbers for pyannote 2.1. |

## Deep-Read Chunks

Chunked PDFs were generated with:

```bash
python .claude/skills/paper-finder/scripts/pdf_chunker.py <paper.pdf> --pages-per-chunk 3 --output-dir papers/pages
```

Deep-read chunk manifests exist for DiarizationLM, contextual beam search, LLM SOT, diarization-aware MS-ASR, DM-ASR, SpeakerLM, and JEDIS-LLM.
