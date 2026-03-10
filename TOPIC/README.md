# TOPIC: Topic-based Conditional Generation Module

This module implements topic-based conditional text generation using BERTopic for keyword extraction, inspired by the CTCL (ConTrollability and CLustering) framework.

## Overview

The TOPIC module provides a lightweight alternative to schema-based conditional generation. Instead of extracting complex JSON schemas using LLMs, it uses BERTopic to:

1. **Cluster documents** into semantic topics (default: 1000 topics)
2. **Extract keywords** for each topic (10 keywords per topic)
3. **Save topic distributions** for sampling and analysis
4. **Generate text** conditioned on sampled topics

## Key Features

- **Lower computational cost**: Uses pretrained sentence transformers instead of LLM API calls
- **Semantic clustering**: Automatically discovers latent topics in your data
- **Simple conditions**: Uses keyword lists instead of complex JSON structures
- **Topic distribution**: Saves topic frequencies for sampling and analysis
- **Flexible sampling**: Sample topics by frequency, uniformly, or from top-k
- **Compatible with existing pipeline**: Fully integrates with DPSFT training/generation

## Directory Structure

```
TOPIC/
├── extract_topics.py           # Extract topics and keywords (all-in-one)
├── sample_topics.py            # Sample topics from distribution
├── scripts/
│   ├── extract_topics.sh          # Multi-dataset extraction pipeline
│   └── sample_topics.sh           # Topic sampling script
├── models/                     # Saved BERTopic models (created after training)
├── distribution/               # Topic frequency distributions (JSON)
├── results/                    # Sampled topics for generation
└── README.md                   # This file
```

## Installation

The TOPIC module requires the following dependencies (add to your conda environment):

```bash
conda activate syn
pip install bertopic sentence-transformers gdown
```

## Usage

### Step 0: Download Pretrained CTCL-Topic Model

First, download the pretrained CTCL-Topic model (trained on Wikipedia with ~1K topics):

```bash
cd TOPIC
bash download_model.sh
```

This downloads the model to `./models/ctcl_topic/`.

**Note**: The CTCL-Topic model is pretrained on Wikipedia and can be directly applied to your documents without training.

### Step 1: Extract Topics and Keywords

Run the extraction script for your dataset:

```bash
cd TOPIC/scripts

# For BioRxiv dataset
bash extract_topics.sh biorxiv

# For OpenReview dataset
bash extract_topics.sh openreview
```

**Supported datasets**: `biorxiv`, `openreview`

This will:
- Load the pretrained CTCL-Topic model
- Extract topic IDs and keywords for both training and validation sets using the pretrained model
- Save topic frequency distribution for training set to `./distribution/biorxiv_topic_distribution_train.json`
- Prepare data in the format required for DPSFT training
- Save outputs to `data/biorxiv/clean_biorxiv_topic_train.csv` and `clean_biorxiv_topic_valid.csv`

**Note**: Only the training set distribution is saved, as it's used for sampling during generation.

**Output Format:**
```csv
topic,schema,abstract
42,"kidney, glomerulus, filtration, ...",<abstract_text>
15,"covid, virus, infection, ...",<abstract_text>
```

**Topic Distribution Format** (`distribution/biorxiv_topic_distribution_train.json`):
```json
{
  "total_documents": 50000,
  "n_topics": 987,
  "topics": {
    "42": {
      "count": 523,
      "frequency": 0.01046,
      "keywords": ["kidney", "glomerulus", "filtration", ...],
      "name": "kidney_glomerulus_filtration"
    },
    ...
  }
}
```

### Step 2: Train Conditional Generation Model

Use the existing DPSFT training pipeline:

```bash
cd ../DPSFT/scripts
bash train.sh biorxiv_condgen_topic <other_args>
```

The training script will automatically:
- Load data from `clean_biorxiv_topic_train.csv`
- Use the `biorxiv_condgen_topic_generation` prompt template
- Format prompts with topic keywords (schema column) as conditions

### Step 3: Sample Topics for Generation

Sample topics from the distribution:

```bash
cd ../TOPIC
python sample_topics.py \
    -d ./distribution/biorxiv_topic_distribution_train.json \
    -o ./results/biorxiv_sampled_topics.csv \
    -n 1000 \
    --sampling_mode frequency
```

Or use the shell script:

```bash
cd scripts
bash sample_topics.sh
```

This generates a CSV file with sampled topics and their keywords:
```csv
topic,schema
42,"kidney, glomerulus, filtration, ..."
15,"covid, virus, infection, ..."
...
```

### Step 4: Generate Synthetic Data

Use the existing generation pipeline with sampled topics:

```bash
cd ../DPSFT
python generation_biorxiv_condgen.py \
    --prompt_file ../TOPIC/results/biorxiv_sampled_topics.csv \
    --schema_column schema \
    --model_name_or_path <trained_model> \
    --output_dir biorxiv_sampled \
    --n_gen 1000
```

## Configuration Options

### extract_topics.py

```bash
python extract_topics.py \
    --input_file <input_csv> \
    --output_file <output_csv> \
    --dataset_name biorxiv \
    --text_column abstract \
    --model_path ./models/ctcl_topic \         # Path to pretrained CTCL-Topic model
    --distribution_path <dist_json> \          # Optional: custom distribution path
    --embedding_model all-MiniLM-L6-v2 \       # Sentence transformer model (default)
    --keep_empty                               # Keep documents without topics (optional)
```

**Output columns**: `topic`, `schema`, `<text_column>`, `<other_columns>`

### sample_topics.py

```bash
python sample_topics.py \
    --distribution_path <dist_json> \
    --output_file <output_csv> \
    --n_samples 1000 \                         # Number of samples
    --sampling_mode frequency \                # Options: frequency, uniform, top_100, top_500
    --format csv \                             # Options: csv, jsonl
    --seed 42                                  # Random seed
```

**Sampling modes**:
- `frequency`: Sample proportional to topic frequency (default)
- `uniform`: Sample uniformly across all topics
- `top_100`: Sample only from top 100 most frequent topics
- `top_500`: Sample only from top 500 most frequent topics

## Integration with DPSFT Pipeline

The TOPIC module integrates seamlessly with the existing DPSFT pipeline through modifications to `data_utils.py`:

1. **Dataset Loading** (`line ~704`): Loads `clean_biorxiv_topic_train.csv` and `clean_biorxiv_topic_valid.csv`
2. **Prompt Template** (`line ~461`): Uses `biorxiv_condgen_topic_generation` template
3. **Data Preprocessing** (`line ~251`): Formats schema (keywords) as conditional features

The prompt template looks like:
```
<start_of_turn>user
Given the list of keywords below, generate a synthetic scientific abstract
that matches the keywords, in the style of a bioRxiv paper.

Keywords: <schema>
<end_of_turn>
<start_of_turn>model
```

## Comparison with Schema-based Condgen

| Aspect | Schema-based (condgen) | Topic-based (CTCL) |
|--------|------------------------|-------------------|
| **Condition Extraction** | LLM extracts JSON schema | BERTopic extracts keywords |
| **Computational Cost** | High (LLM API calls) | Low (sentence transformers) |
| **Condition Complexity** | Rich (8+ dimensions) | Simple (10 keywords) |
| **Control Granularity** | Fine-grained | Semantic/topic-level |
| **Dependency** | Requires LLM API | Self-contained |
| **Topic Distribution** | Not available | Saved for sampling |
| **Sampling** | Not supported | Frequency-based sampling |

## Workflow Example

Complete workflow from data to synthetic generation:

```bash
# 0. Download pretrained CTCL-Topic model (one-time setup)
cd TOPIC
bash download_model.sh

# 1. Extract topics from training data
cd scripts
bash extract_topics.sh biorxiv  # or: bash extract_topics.sh openreview

# 2. Train conditional generation model
cd ../../DPSFT/scripts
bash train.sh biorxiv_condgen_topic --eps 4.0

# 3. Sample topics for generation
cd ../../TOPIC
python sample_topics.py \
    -d ./distribution/biorxiv_topic_distribution_train.json \
    -o ./results/biorxiv_sampled_1000.csv \
    -n 1000 \
    --sampling_mode frequency

# 4. Generate synthetic abstracts
cd ../DPSFT
python generation_biorxiv_condgen.py \
    --prompt_file ../TOPIC/results/biorxiv_sampled_1000.csv \
    --schema_column schema \
    --model_name_or_path ./models/biorxiv_condgen_topic_eps4 \
    --output_dir biorxiv_synthetic \
    --n_gen 1000
```

## Advanced Usage

### About the Pretrained Model

The CTCL-Topic model is pretrained on Wikipedia (~6M documents) with approximately 1000 topics. It can be directly applied to various domains without retraining. The model uses:
- **Embedding**: all-MiniLM-L6-v2 sentence transformer
- **Topics**: ~1000 semantic topics
- **Keywords**: 10 keywords per topic

### Manual Model Download

If the automatic download script doesn't work, you can manually download:

```bash
# Install gdown
pip install gdown

# Download from Google Drive
cd TOPIC/models
gdown 1sbda6ROyMewThuoDA3bxP71ucihcf7qJ
unzip ctcl_pretrained.zip
```

### Analyzing Topic Distributions

After extraction, you can analyze the topic distribution:

```python
import json
import pandas as pd

# Load distribution
with open('./distribution/biorxiv_topic_distribution_train.json', 'r') as f:
    dist = json.load(f)

# Get top 10 most frequent topics
topics = dist['topics']
top_topics = sorted(topics.items(), key=lambda x: x[1]['count'], reverse=True)[:10]

for topic_id, info in top_topics:
    print(f"Topic {topic_id}: {info['name']}")
    print(f"  Count: {info['count']} ({info['frequency']:.2%})")
    print(f"  Keywords: {', '.join(info['keywords'][:5])}")
```

### Custom Sampling Strategies

You can implement custom sampling strategies:

```python
from TOPIC.sample_topics import load_topic_distribution, sample_topics

# Load distribution
dist = load_topic_distribution('./distribution/biorxiv_topic_distribution_train.json')

# Custom sampling: oversample rare topics
# ... your custom logic ...
```

## Troubleshooting

### Issue: Out of Memory during Topic Extraction

**Solution**: Reduce batch size or use smaller embedding model
```bash
# Use a smaller embedding model
--embedding_model "paraphrase-MiniLM-L3-v2"
```

### Issue: Too Many Outlier Topics

**Solution**: Adjust HDBSCAN parameters in `train_bertopic_model()`:
```python
hdbscan_model = HDBSCAN(
    min_cluster_size=100,  # Increase for fewer outliers
    min_samples=20
)
```

### Issue: Keywords Not Relevant

**Solution**:
1. Check if your documents are too short/noisy
2. Try different embedding models
3. Adjust the number of topics

### Issue: Sampling produces too few unique topics

**Solution**: Use uniform sampling or increase top-k:
```bash
python sample_topics.py ... --sampling_mode uniform
# or
python sample_topics.py ... --sampling_mode top_500
```

## File Naming Convention

The TOPIC module follows the naming convention: `clean_{dataset_name}_topic_{split}.csv`

Examples:
- `clean_biorxiv_topic_train.csv`
- `clean_biorxiv_topic_valid.csv`
- `clean_openreview_topic_train.csv`

This matches the existing `clean_{dataset_name}_schema_{variant}_{split}.csv` convention used for schema-based condgen.

## References

- **CTCL Paper**: [Synthetic Private Data with ConTrollability and CLustering](https://github.com/tanyuqian/synthetic-private-data)
- **BERTopic**: [Official Documentation](https://maartengr.github.io/BERTopic/)
- **Sentence Transformers**: [Official Documentation](https://www.sbert.net/)

## Citation

If you use this module, please cite both the CTCL paper and BERTopic:

```bibtex
@article{tan2024ctcl,
  title={Data Synthesis with ConTrollability and CLustering},
  author={Tan, Yuqian and others},
  journal={ICML},
  year={2025}
}

@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```
