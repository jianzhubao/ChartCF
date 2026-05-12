#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="data/ECD/syn_data"

for SPLIT_TYPE in "descriptive" "reasoning"; do
    python src/chartcf/data_syn/cf_data_syn.py \
        --template "prompts/prompt_data_syn_${SPLIT_TYPE}.md" \
        --output-dir "${OUTPUT_DIR}/${SPLIT_TYPE}" \
        --split-type "${SPLIT_TYPE}" \
        --workers 16

    python src/chartcf/data_syn/post_process.py --split-type "${SPLIT_TYPE}"
done

python src/chartcf/data_syn/compute_similarity.py --max-workers 16

python src/chartcf/data_syn/merge_dpo_data.py \
    --output data/ECD/syn_data/dpo_data.json

python src/chartcf/data_syn/data_selection.py \
    --input-file ${OUTPUT_DIR}/dpo_data.json \
    --output-dir ${OUTPUT_DIR} \
    --keep-percent 40
