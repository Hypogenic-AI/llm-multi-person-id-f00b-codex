# Literature Review: LLMs for Multi-Person Identification in Multi-Speaker ASR

## Review Scope

### Research Question

Can large language models improve multi-person or multi-speaker identification when ASR and diarization systems struggle in conversations with many speakers, overlap, poor timestamps, or long context?

### Inclusion Criteria

- Papers on speaker diarization, speaker-attributed ASR, multi-talker ASR, or speaker identification.
- Work using LLMs, pretrained language models, or language-model context for speaker assignment.
- Benchmarks and tools relevant to AMI, CallHome, Fisher, AliMeeting, AISHELL-4, ICSI, or meeting transcription.
- Recent work from 2023-2026 plus foundational baselines/tools.

### Exclusion Criteria

- Single-speaker ASR papers without speaker attribution.
- Speaker verification papers that do not connect to conversational identification or diarization.
- Non-reproducible resources with no accessible paper, code, or dataset path.

### Search Log

| Date | Query/Source | Results | Notes |
|------|--------------|---------|-------|
| 2026-07-06 | paper-finder diligent query | stalled | Local service process did not return output; manual search used. |
| 2026-07-06 | arXiv / web: LLM speaker diarization ASR | 15 papers downloaded | Found DiarizationLM, DM-ASR, SpeakerLM, JEDIS-LLM, LLM SOT, pyannote, MeetEval. |
| 2026-07-06 | Hugging Face dataset metadata | 6 datasets screened | AMI, VoxConverse, CallHome, ICSI, Simsamu, AVA-AVD. |
| 2026-07-06 | GitHub/code search | 6 repos cloned | DiarizationLM, LLM-Diarize-ASR-Agnostic, diarizers, pyannote, WhisperX, MeetEval. |

## Key Papers

### DiarizationLM

- Authors: Quan Wang, Yiling Huang, Guanlong Zhao, Evan Clark, Wei Xia, Hank Liao
- Year: 2024
- Source: arXiv/Interspeech
- Methodology: Represents ASR plus diarization outputs as compact speaker-token text, prompts or fine-tunes an LLM to correct word-speaker assignments, then uses Transcript-Preserving Speaker Transfer (TPST) to keep the original ASR words while transferring speaker labels.
- Datasets: Fisher and CallHome English.
- Results: A fine-tuned PaLM 2-S reduced WDER from 5.32 to 2.37 on Fisher and from 7.72 to 4.25 on CallHome. Zero-shot/one-shot models often degraded results, mainly through transcript deletion or hallucination.
- Code: `code/google_speaker_id/DiarizationLM/`
- Relevance: Strongest directly applicable baseline for LLM post-processing of multi-person identification.

### Enhancing Speaker Diarization with LLMs

- Authors: Tae Jin Park, Kunal Dhawan, Nithin Koluguri, Jagadeesh Balam
- Year: 2023/2024
- Methodology: Adds lexical probabilities from an n-gram LM or GPT-style LLM into a joint acoustic and lexical beam search over speaker assignments.
- Datasets: AMI-MH and CallHome English subsets.
- Results: Up to 39.8 percent relative delta-SA-WER improvement over timestamp matching. LLM context was more useful for estimating speaker probability than n-gram context.
- Relevance: Shows language context can be used during decoding, not only as post-processing.

### LLM-Based Serialized Output Training for Multi-Talker ASR

- Authors: Mohan Shi et al.
- Year: 2024
- Methodology: Uses WavLM speech encoder plus Vicuna-7B decoder to generate serialized-output multi-speaker transcripts separated by speaker-change symbols.
- Datasets: LibriMix and AMI-SDM.
- Results: On AMI-SDM, LLM architecture achieved 21.0 cpWER, improved to 20.4 with beam search, outperforming an AED model trained on much more supervised data.
- Relevance: Useful for overlap-heavy conditions where speaker changes and transcript content are learned jointly.

### Diarization-Aware Multi-Speaker ASR via LLMs

- Authors: Yuke Lin, Ming Cheng, Ze Li, Beilong Tang, Ming Li
- Year: 2025
- Methodology: Uses structured triplets of speaker embedding, start time, and end time to condition Qwen2.5-3B for segment-level transcription.
- Datasets: AliMeeting and MLC-SLM multilingual challenge data.
- Results: Strong tcpWER improvements over the official MLC-SLM baseline, reducing average dev tcpWER from 76.12 to 24.95.
- Relevance: Supports the hypothesis that explicit diarization priors plus LLM generation can outperform pure pipelines.

### DM-ASR

- Authors: Li Li, Ming Cheng, Weixin Zhu, Yannan Wang, Juan Liu, Ming Li
- Year: 2026
- Methodology: Reformulates multi-speaker ASR as multi-turn dialogue generation. Each turn asks the LLM to transcribe one diarization-provided speaker/time segment, with optional word-level timestamp tokens.
- Datasets: AliMeeting, AISHELL-4, AMI-IHM, AMI-SDM, Fisher.
- Results: DM-ASR with S2SND and 1.7B LLM reached AMI-IHM cpWER 16.40 and Fisher cpWER 15.91; word-level timestamp supervision improved both structured prediction and transcription quality.
- Relevance: Best template for a small-model experiment using diarization as structural prior.

### SpeakerLM

- Authors: Han Yin et al.
- Year: 2025/2026
- Methodology: Unified multimodal LLM for speaker diarization and recognition with flexible speaker registration modes.
- Datasets: AliMeeting, AISHELL4, AISHELL5, large in-house data, simulated mixtures.
- Results: With 7,638.95 hours of training data, SpeakerLM reached AliMeeting cpCER 16.05 and AISHELL4 cpCER 18.37, outperforming cascaded baselines.
- Relevance: Shows the high-scale end-to-end direction, but requires far more data and compute than a resource-constrained project.

### JEDIS-LLM

- Authors: Mohan Shi, Xiong Xiao, Ruchao Fan, Shaoshi Ling, Jinyu Li
- Year: 2025
- Methodology: Streamable joint ASR and diarization with a Speaker Prompt Cache for global speaker consistency across chunks.
- Datasets: AMI, ICSI, Fisher, internal data, VoxCeleb simulations.
- Results: In long-form settings, streaming JEDIS-LLM with cache updates outperformed DiarizationLM on WDER/cpWER for CH109 and Fisher.
- Relevance: Important for long recordings and many chunks, but training setup is heavy.

### Language Modeling and Speaker Identification Papers

- India et al. (2025) show that combining linguistic content with acoustic scores can reduce word-level DER sharply in structured telephonic interviews.
- Zamana et al. (2024) show pretrained LMs and GPT-4 can identify named speakers from transcript context when names are introduced, improving recall on some conversational data.
- These papers support adding text-only speaker role/name inference as an auxiliary task, but they are not complete diarization systems.

## Common Methodologies

- LLM post-processing: convert diarized ASR output into speaker-tagged text and ask/fine-tune an LLM to correct speaker assignments. Strong when fine-tuned; risky zero-shot.
- Lexical beam search: combine acoustic diarization probabilities with LM probabilities over likely speakers for each word.
- Serialized-output multi-talker ASR: generate a single ordered transcript with speaker-change markers.
- Diarization-aware prompting: feed speaker/time cues from a diarizer into the LLM, allowing it to focus on "what" while the diarizer supplies "who/when".
- Unified Speech-LLM: jointly predict speech content and speaker attribution from audio, often requiring large models and large training data.

## Standard Baselines

- Cascaded ASR plus diarization: WhisperX or Whisper plus pyannote/DiariZen, then timestamp-based speaker-word assignment.
- pyannote.audio: strong open diarization baseline; requires HF model terms/token for current pretrained pipelines.
- DiariZen / S2SND / Sortformer / Meta-CAT: strong modern diarization or speaker-aware baselines reported in recent papers.
- DiarizationLM: strongest LLM post-processing baseline with open code and model artifacts.
- MeetEval: standard cpWER/tcpWER evaluation tool, though local installation needs a C++ compiler.

## Evaluation Metrics

- DER: speaker diarization error rate; measures false alarm, miss, and speaker confusion over time.
- WDER: word diarization error rate; measures speaker assignment errors for words.
- cpWER/cpCER: minimum-permutation concatenated WER/CER across speakers; captures speaker consistency plus lexical errors.
- tcpWER/tcpCER: time-constrained cpWER/cpCER; adds timestamp plausibility and is better for "who said what when".
- SA-WER: speaker-attributed WER with stricter speaker mapping.
- Speaker count accuracy or speaker count MAE: useful when number of speakers is unknown.

## Datasets in the Literature

- AMI: primary English meeting benchmark; many papers evaluate IHM and SDM variants.
- Fisher and CallHome: two-speaker telephone benchmarks used by DiarizationLM and JEDIS-LLM.
- AliMeeting and AISHELL-4: Mandarin multi-speaker meeting benchmarks with high overlap.
- ICSI: long meeting data with up to many speakers, useful for testing "many speaker" behavior.
- VoxConverse: web video diarization, useful for out-of-domain generalization.
- LibriMix: simulated overlap benchmark, useful but less realistic than AMI/ICSI.
- Simsamu: small local sanity dataset for diarization format and loader validation.

## Gaps and Opportunities

- Many LLM correction results are strongest on two-speaker telephone data, while the hypothesis emphasizes many-speaker ASR limitations.
- Zero-shot LLM correction can hallucinate, delete words, or alter transcript content; transcript-preserving transfer is important.
- Full end-to-end Speech-LLMs work well but usually require large proprietary data and substantial GPU budgets.
- Explicit timestamps are under-modeled in older LLM speaker-attribution papers; tcpWER/tcpCER should be included.
- Speaker names/roles are promising but lack standardized datasets and metrics.

## Recommendations for the Experiment Runner

- Start with a practical cascade baseline: WhisperX for ASR/word timestamps and pyannote for diarization, using AMI SDM sample first and full AMI/ICSI later.
- Add DiarizationLM-style post-processing using the open Google code. Preserve ASR text with TPST and evaluate WDER/cpWER.
- For the core hypothesis, compare three conditions: raw timestamp assignment, LLM-corrected speaker labels, and diarization-aware prompting with speaker/time segments.
- Use AMI SDM or ICSI as the primary many-speaker benchmark. Use CallHome only as a secondary two-speaker comparison if credentials are available.
- Report WDER, cpWER, and speaker count error at minimum. Add tcpWER after installing MeetEval with a C++ compiler.
- Avoid relying on zero-shot LLM correction for final claims; include it only as an ablation because the literature shows it can degrade results.
