# Research Plan: LLMs for Multi-Person Identification

## Motivation & Novelty Assessment

### Why This Research Matters
Multi-speaker meetings remain difficult for ASR pipelines because word recognition, timestamps, speaker turns, and speaker identity are coupled: when speakers overlap or many people participate, small timing or clustering errors turn into "who said what" errors. Improving speaker attribution benefits meeting transcription, clinical or legal conversation review, contact-center analytics, and any downstream system that relies on accurate participant accountability.

### Gap in Existing Work
The gathered literature shows strong LLM gains for diarization post-processing, especially DiarizationLM on Fisher and CallHome, but many published gains are strongest on two-speaker telephone conversations or require large proprietary speech-LLM training. The local resources include an AMI SDM shard with four-speaker meeting timelines but no reference words, so a realistic lightweight experiment should isolate whether real LLMs can improve text-level speaker attribution under controlled many-speaker diarization noise before scaling to full cpWER/tcpWER benchmarks.

### Our Novel Contribution
This study builds a reproducible pilot benchmark from local AMI and Simsamu audio: segments are transcribed with a real ASR API, assigned ground-truth segment speakers from downloaded diarization annotations, then corrupted with timestamp- and overlap-aware diarization noise. We compare noisy speaker labels, a deterministic continuity smoother, zero-profile LLM relabeling, and profile-conditioned LLM relabeling while preserving ASR words and evaluating word-level speaker attribution.

### Experiment Justification
- Experiment 1: Transcribe local AMI/Simsamu segments with real ASR. This grounds the benchmark in actual audio instead of invented transcripts.
- Experiment 2: Inject speaker-label noise that increases with overlap, short turns, and speaker changes. This approximates the known failure surface of multi-person ASR/diarization pipelines.
- Experiment 3: Compare noisy labels, heuristic smoothing, and LLM relabeling. This tests whether language context adds value beyond simple temporal continuity.
- Experiment 4: Compare two-speaker Simsamu windows against four-speaker AMI windows. This directly probes the user concern that performance degrades as speakers become many.

## Research Question
Can real large language models improve multi-person speaker identification from ASR transcripts when baseline speaker labels are noisy, especially in four-speaker meeting-style conversations?

## Background and Motivation
Prior work indicates that language context can correct speaker-attribution errors, but zero-shot LLM post-processing can also hallucinate or degrade transcripts. The practical question is not whether an LLM can replace diarization from audio, but whether it can act as a transcript-preserving correction layer over ASR plus diarization outputs. This plan follows the DiarizationLM principle of preserving words while changing speaker labels.

## Hypothesis Decomposition
- H1: LLM relabeling lowers word diarization error rate (WDER) relative to noisy ASR/diarization labels.
- H2: Profile-conditioned LLM relabeling outperforms zero-profile LLM relabeling because speaker-specific lexical/context cues provide identity anchors.
- H3: Gains are smaller or less stable in four-speaker AMI windows than two-speaker Simsamu windows because candidate-speaker ambiguity grows.
- H4: LLM relabeling must preserve transcripts; a method that lowers WDER by changing or dropping words is not acceptable.

Independent variables are dataset family (AMI four-speaker vs Simsamu two-speaker), noise level (low/medium/high), and correction method. Dependent variables are WDER, cpWER-style speaker-attributed WER, speaker count MAE, parse failure rate, and transcript preservation rate.

## Proposed Methodology

### Approach
Use local diarization annotations as speaker ground truth, real ASR transcripts for selected audio segments, and real LLM API calls for speaker relabeling. The experiment is segment-level for LLM output but word-weighted for evaluation: every word in a segment inherits the segment speaker label, allowing WDER-style comparison while avoiding artificial word edits.

### Experimental Steps
1. Load AMI SDM sample parquet and Simsamu saved dataset; summarize speaker counts, turn durations, and overlap indicators.
2. Select a small set of windows with enough speaker diversity and segment durations suitable for API transcription.
3. Transcribe selected audio segments with OpenAI ASR, caching all responses for reproducibility and cost control.
4. Build evaluation windows with unscored speaker profile turns and scored test turns.
5. Corrupt test-turn speaker labels with seeded noise that depends on base noise level, overlap, short duration, and speaker-change boundaries.
6. Run baselines: raw noisy labels and deterministic continuity smoothing.
7. Run LLM conditions: no-profile relabeling and profile-conditioned relabeling using JSON-only speaker outputs.
8. Compute metrics and paired statistical tests; generate result tables and figures.

### Baselines
- Noisy labels: the simulated ASR/diarization output.
- Continuity smoother: fixes isolated single-turn speaker changes when neighboring turns agree.
- LLM no-profile: real LLM relabeling using only the transcript and noisy speaker labels.
- LLM profile: real LLM relabeling using the same input plus clean profile turns per speaker.

### Evaluation Metrics
- WDER: word-weighted speaker-label error rate on scored turns.
- cpWER-style score: minimum-permutation concatenated word error rate by speaker.
- Speaker count MAE: absolute error in the number of speakers used in a window.
- Transcript edit rate / preservation: checks whether the LLM condition changed words; expected to be zero because JSON speaker-only output is requested.
- Parse failure rate: fraction of LLM outputs that could not be parsed or had missing segment IDs.

### Statistical Analysis Plan
Use paired tests by window and noise condition. For WDER, compare each correction method against noisy labels using Wilcoxon signed-rank tests if normality is not supported; otherwise use paired t-tests. Report bootstrap 95% confidence intervals for mean WDER differences and Cohen's d for paired differences. Apply Holm correction across the main method comparisons.

## Expected Outcomes
Results support the hypothesis if LLM profile relabeling significantly reduces WDER versus noisy labels without transcript changes, and if the effect remains positive in AMI four-speaker windows. Results refute or weaken the hypothesis if LLM relabeling fails to beat continuity smoothing, increases parse failures, or only helps in two-speaker windows.

## Timeline and Milestones
- Setup and dataset validation: 10-20 minutes.
- Script implementation: 60-90 minutes.
- ASR and LLM experiment runs: 60-90 minutes, depending on API latency.
- Analysis and visualization: 30-45 minutes.
- Final report and validation: 20-30 minutes.

## Potential Challenges
- Local AMI annotations lack reference transcripts; ASR-derived text is a pseudo-reference for words, so conclusions focus on speaker attribution rather than ASR word accuracy.
- API transcription may fail on very short or noisy segments; the loader filters short segments and caches failures.
- Zero-shot LLM outputs may be malformed; the harness validates JSON and falls back to noisy labels while counting parse failures.
- Small sample size limits external validity; this is a pilot, not a full AMI benchmark.

## Success Criteria
The research session succeeds if it produces a runnable experiment harness, cached real ASR/LLM outputs, statistical comparisons across baselines, figures/tables, and a final report that clearly states whether LLM speaker relabeling improved multi-person identification under the tested conditions.
