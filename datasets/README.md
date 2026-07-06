# Downloaded Datasets

Data files are kept locally for experimentation but excluded from git by `datasets/.gitignore`.

## Dataset 1: Simsamu diarization dataset

### Overview

- Source: `diarizers-community/simsamu` on Hugging Face
- Location: `datasets/simsamu/`
- Size on disk: about 171 MB
- Format: Hugging Face `DatasetDict` saved with `save_to_disk`
- Splits: `train`, 61 recordings
- Schema: `audio`, `timestamps_start`, `timestamps_end`, `speakers`
- License: MIT according to the Hugging Face dataset card
- Task: speaker diarization sanity checks and pipeline validation

### Download Instructions

```python
from datasets import load_dataset, Audio

dataset = load_dataset("diarizers-community/simsamu")
dataset = dataset.cast_column("audio", Audio(decode=False))
dataset.save_to_disk("datasets/simsamu")
```

### Loading

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/simsamu")
sample = dataset["train"][0]
```

### Sample Data

A small annotation preview is saved at:

```text
datasets/simsamu/samples/sample_annotations.json
```

## Dataset 2: AMI SDM sample shard

### Overview

- Source: `diarizers-community/ami`, config `sdm`, test split
- Location: `datasets/ami_sdm_sample/sdm/test-00000-of-00003.parquet`
- Size on disk: about 313 MB
- Local rows: 6 meeting recordings
- Format: Parquet with embedded audio bytes and diarization annotations
- Schema: `audio`, `timestamps_start`, `timestamps_end`, `speakers`
- License: CC BY 4.0 according to the Hugging Face dataset card
- Task: English meeting diarization/ASR sanity sample

Only one AMI shard was downloaded because the full processed HF dataset is about 20 GB. This local shard is enough for validating loaders, diarization output formatting, and small end-to-end experiments.

### Download Instructions

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="diarizers-community/ami",
    repo_type="dataset",
    filename="sdm/test-00000-of-00003.parquet",
    local_dir="datasets/ami_sdm_sample",
)
```

### Loading

```python
import pyarrow.parquet as pq

table = pq.read_table("datasets/ami_sdm_sample/sdm/test-00000-of-00003.parquet")
row = table.slice(0, 1).to_pylist()[0]
```

### Sample Data

A small annotation preview is saved at:

```text
datasets/ami_sdm_sample/samples/sample_annotations.json
```

## Recommended Full Datasets for Main Experiments

### AMI Meeting Corpus

- HF source: `diarizers-community/ami`
- Configs: `ihm`, `sdm`
- Full processed size: about 20 GB
- Best use: primary English meeting benchmark, especially SDM for difficult far-field audio

```python
from datasets import load_dataset

dataset = load_dataset("diarizers-community/ami", "sdm")
dataset.save_to_disk("datasets/ami_sdm_full")
```

### VoxConverse

- HF source: `diarizers-community/voxconverse`
- Full processed size: about 7 GB
- Best use: out-of-domain English diarization, web video speech

```python
from datasets import load_dataset

dataset = load_dataset("diarizers-community/voxconverse")
dataset.save_to_disk("datasets/voxconverse_full")
```

### ICSI Meetings

- HF source: `argmaxinc/icsi-meetings`
- Full processed size: about 7.8 GB
- Best use: multi-speaker meetings with up to many participants

```python
from datasets import load_dataset

dataset = load_dataset("argmaxinc/icsi-meetings")
dataset.save_to_disk("datasets/icsi_meetings_full")
```

### CallHome

- HF source: `talkbank/callhome`
- Access: gated; requires accepting terms and authenticating with `HF_TOKEN`
- Best use: two-speaker telephone benchmark used by DiarizationLM and JEDIS-LLM

```python
from datasets import load_dataset

dataset = load_dataset("talkbank/callhome", "eng")
dataset.save_to_disk("datasets/callhome_eng_full")
```

## Validation Notes

- `datasets/simsamu` loads with `load_from_disk` and has 61 rows.
- `datasets/ami_sdm_sample` parquet loads with `pyarrow` and has 6 rows.
- CallHome was not downloaded because it is gated without an authenticated token.
- Full AMI/VoxConverse/ICSI downloads were not pulled to avoid unnecessary 7-20 GB transfers during resource gathering.
