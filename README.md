# ChartCF

This is the implementation of our paper "**Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding**", which is accepted to the **ACL 2026 main conference**.

## 📖 Overview

Vision-Language Models (VLMs) have made strong progress on chart understanding, largely by supervised fine-tuning (SFT) on synthetic data. However, scaling SFT data alone overlooks a key property of charts: charts are programmatically generated visual artifacts, so small, code-controlled visual modifications can drastically shift semantics and change answers. Standard SFT treats training instances independently and provides limited supervision for this counterfactual sensitivity. **ChartCF** is a data-efficient training framework that pairs counterfactual chart synthesis with multimodal preference optimization to instill this sensitivity in VLMs.

## 📁 Repository Structure

```
ChartCF/
├── prompts/                # Prompt templates for data synthesis & similarity eval
├── scripts/
│   ├── run_data_syn.sh                     # End-to-end data synthesis pipeline
│   ├── run_train_*.sh                      # Per-model DPO training entry points
│   ├── run_eval_*.sh                       # Per-model evaluation entry points
│   └── eval/                               # Per-benchmark evaluation sub-scripts
└── src/chartcf/
    ├── data_syn/           # CF data generation, post-processing, similarity, selection
    └── evaluation/         # ChartQA / ChartBench / ChartX / CharXiv / ECDBench evaluators
    └── patches/            # Runtime monkey-patches for ms-swift to relax rejected_response checks for text and image DPO training
```

## 📦 Installation

```bash
uv sync

# You may also need to install the following system dependencies for rendering of charts during data synthesis. The exact list may depend on your OS and environment, but these are commonly required.
plotly_get_chrome

sudo apt-get install libnss3 libatk-bridge2.0-0t64 libcups2t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2t64
```

## 🔑 API Setup

Several stages (data synthesis, CharXiv grading) call the OpenAI API. Create a `.env` file at the repository root with your credentials:

```bash
OPENAI_API_KEY="xxx"
OPENAI_BASE_URL="xxx"
```

## 📊 Data Preparation

> 💾 All artifacts referenced below are released in our HuggingFace dataset repo: [jianzhubao/ChartCF-Data](https://huggingface.co/datasets/jianzhubao/ChartCF-Data).

1. Download [`codes_with_seed.zip`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/codes_with_seed.zip) and extract it to `data/ECD/codes_with_seed`. These are derived from the ECD raw data ([codes.tar](https://huggingface.co/datasets/ChartFoundation/ECD-10k-Images/blob/main/codes.tar)). To eliminate randomness, we prepend a fixed random seed to every script. A small number of ECD scripts that we could not execute successfully or could not make deterministic have been removed.
2. Run every script under `data/ECD/codes_with_seed` to render the charts and save them to `data/ECD/rendered_images_with_seed`. Alternatively, you can also download our pre-rendered images: [`rendered_images_with_seed.zip`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/rendered_images_with_seed.zip).
3. Download [`ECD_QAs_one_per_image.json`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/ECD_QAs_one_per_image.json) to `data/ECD/ECD_QAs_one_per_image.json`. This file contains exactly one QA pair per image, sub-sampled from the ECD raw data ([ECD_QAs_All.json](https://huggingface.co/datasets/ChartFoundation/ECD-10k-Images/blob/main/ECD_QAs_All.json)).

## 🧪 Data Synthesis

```bash
bash scripts/run_data_syn.sh
```

This produces `data/ECD/syn_data/dpo_data_keep40.json`.

Alternatively, you can directly use the synthesized and filtered 4k-sample dataset that we used: [`dpo_data_keep40.json`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/dpo_data_keep40.json). Place it at `data/ECD/syn_data/dpo_data_keep40.json`. Also, Download [`descriptive_post.zip`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/descriptive_post.zip) and extract into `ChartCF/data/ECD/syn_data/descriptive_post`, and download [`reasoning_post.zip`](https://huggingface.co/datasets/jianzhubao/ChartCF-Data/blob/main/reasoning_post.zip) and extract into `ChartCF/data/ECD/syn_data/reasoning_post`.

## 🚀 Training

You can run the training scripts for each model as follows.

We also release the trained LoRA adapters for all three models at [jianzhubao/ChartCF-Models](https://huggingface.co/jianzhubao/ChartCF-Models). Each subfolder (`ChartCF_qwen2_5_vl_7b_instruct`, `ChartCF_qwen3_vl_8b_instruct`, `ChartCF_internvl3_5_8b_instruct`) can be plugged directly into the matching `CHECKPOINT_DIR` in the evaluation scripts below.

### Qwen2.5-VL-7B-Instruct

```bash
bash scripts/run_train_qwen2_5_vl_7b_instruct.sh
```

### Qwen3-VL-8B-Instruct

```bash
bash scripts/run_train_qwen3_vl_8b_instruct.sh
```

### InternVL3.5-8B-Instruct

```bash
bash scripts/run_train_internvl3_5_8b_instruct.sh
```

## 🧭 Evaluation

Follow the instructions under the `public_benchmarks` folder of [ECD](https://github.com/yuweiyang-anu/ECD/tree/main) to download the relevant benchmarks into `data/public_benchmarks`.

🧪 The evaluation protocol is aligned with prior work [ECD/evaluation](https://github.com/yuweiyang-anu/ECD/tree/main/evaluation).

> ⚠️ Before running, edit the `CHECKPOINT_DIR` variable inside each entry script to point to the best validation checkpoint of the corresponding training run.

### Qwen2.5-VL-7B-Instruct

```bash
bash scripts/run_eval_qwen2_5_vl_7b_instruct.sh
```

### Qwen3-VL-8B-Instruct

```bash
bash scripts/run_eval_qwen3_vl_8b_instruct.sh
```

### InternVL3.5-8B-Instruct

```bash
bash scripts/run_eval_internvl3_5_8b_instruct.sh
```

## 🙏 Acknowledgments

We build upon the data, evaluation protocols, and tooling released by the following projects, and we thank their authors and maintainers:

- [ECD](https://github.com/yuweiyang-anu/ECD) — chart image/code corpus, QA data, and evaluation pipeline that ChartCF aligns with.
- [ms-swift](https://github.com/modelscope/ms-swift) — DPO training and inference framework used by all `run_train_*.sh` and `run_eval_*.sh` scripts.
- [CharXiv](https://github.com/princeton-nlp/CharXiv) — descriptive/reasoning evaluation utilities reused under `src/chartcf/evaluation/CharXiv/eval_utils/`.
- [ChartQA](https://github.com/vis-nlp/ChartQA), [ChartBench](https://github.com/IDEA-FinAI/ChartBench), [ChartX](https://github.com/InternScience/ChartVLM), and [ECDBench](https://github.com/yuweiyang-anu/ECD) — public benchmarks used for evaluation.

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{bao2026learninglessexploitingcounterfactuals,
      title={Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding}, 
      author={Jianzhu Bao and Haozhen Zhang and Kuicai Dong and Bozhi Wu and Sarthak Ketanbhai Modi and Zi Pong Lim and Yon Shin Teo and Wenya Wang},
      year={2026},
      eprint={2605.10855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.10855}, 
}
```