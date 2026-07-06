# LLMs for Multi-Person Identification

This workspace tests whether large language models can improve speaker identification in multi-person ASR-style transcripts. The pilot uses local AMI SDM and Simsamu audio annotations, real OpenAI ASR for transcript text, and real `gpt-4.1-mini` calls for transcript-preserving speaker relabeling.

## Key Findings

- LLM relabeling was safe to parse: 60/60 LLM responses returned valid JSON speaker assignments.
- Overall WDER changed from 0.390 for noisy labels to 0.375 with zero-profile LLM relabeling, but this was not statistically significant after Holm correction.
- Four-speaker AMI windows did not improve: the LLM generally copied the noisy labels.
- Two-speaker Simsamu windows improved modestly with the no-profile LLM condition, reducing mean WDER from 0.310 to 0.273.
- Simple continuity smoothing was worse than noisy labels overall, showing that naive temporal smoothing is not a reliable fix.

See [REPORT.md](REPORT.md) for full methodology, results, figures, and limitations.

## Reproduce

```bash
source .venv/bin/activate
python src/run_experiment.py
```

The script reuses cached ASR/LLM outputs in `results/cache/` and `results/model_outputs/`, so reruns should not make additional API calls unless caches are removed.

## File Structure

- `planning.md`: motivation, novelty, and experiment plan.
- `src/run_experiment.py`: experiment, API calls, metrics, statistics, and figures.
- `results/metrics_by_case.csv`: per-case WDER/cpWER metrics.
- `results/aggregate_metrics.csv`: aggregate method comparisons.
- `results/statistical_tests.csv`: paired tests and effect sizes.
- `results/usage_summary.json`: token/audio usage and estimated API cost.
- `figures/`: WDER and parse-success plots.
- `literature_review.md`, `resources.md`, `papers/`, `datasets/`, `code/`: pre-gathered research materials.
