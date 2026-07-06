# Outline

## Title
- Prompted LLMs Are Conservative Speaker Relabelers: A Transcript-Preserving Pilot on AMI and Simsamu.

## Abstract
- Problem: meeting transcripts need accurate "who said what" attribution.
- Approach: real ASR text from local AMI-SDM and Simsamu segments, seeded noisy labels, and real GPT-4.1-mini relabeling with and without speaker profiles.
- Evidence: 30 window/noise cases, 60 LLM calls, 100% JSON parse success, mean WDER from 0.390 to 0.375 for no-profile relabeling, Holm-adjusted p=0.410.
- Takeaway: prompted LLMs can be a transcript-preserving correction layer, but naive prompting is not reliable for four-speaker meetings.

## Introduction
- Hook: speaker attribution is the part of ASR that turns transcript text into accountable dialogue.
- Gap: LLM post-processing has promising results, but lightweight API prompting in many-speaker meetings is under-tested.
- Approach: controlled pilot with real ASR, seeded noise, profile and no-profile prompts, WDER/cpWER evaluation.
- Contributions: benchmark construction, comparison of baselines, dataset split analysis, reproducible cost-aware harness.

## Related Work
- LLM post-processing and transcript preservation: DiarizationLM.
- Language-assisted diarization and decoding: contextual beam search and language-model diarization.
- Speech-LLM and diarization-aware ASR: serialized output, DM-ASR, SpeakerLM, JEDIS-LLM.
- Evaluation and baselines: MeetEval, pyannote, AMI.

## Methodology
- Inputs: AMI-SDM and Simsamu windows; local diarization annotations as speaker truth; OpenAI ASR text as preserved words.
- Noise: seeded label corruption higher under overlap, short turns, and speaker-change boundaries.
- Methods: noisy labels, continuity smoothing, LLM no-profile, LLM profile.
- Metrics: WDER, local cpWER-style exact-permutation score, segment error, speaker count MAE, parse success.

## Results
- Main table: no-profile LLM has the best mean WDER/cpWER but only small improvement.
- Dataset split: Simsamu improves; AMI is unchanged for both LLM conditions.
- Statistics: no method significant after Holm correction; no-profile p=0.137, Holm p=0.410.
- Error analysis: no-profile unchanged in 26/30 cases, improved 3, worsened 1.

## Discussion and Conclusion
- Interpretation: text-only prompts are conservative when lexical identity cues are weak.
- Limitations: small pilot, ASR pseudo-reference text for AMI, simulated noise, text-only profiles, local cpWER implementation.
- Future work: full AMI/ICSI, real WhisperX+pyannote pipeline, acoustic confidence/features, fine-tuned DiarizationLM-style models, MeetEval installation.
