#!/bin/bash
#
# run_train_model.sh \
#  backbone \
#  input_size \
#  input_crop \
#  batch_size \
#  embedding_dim \
#  epochs \
#  'rand-config' \
#  'outdir' \
#  'indir' \
#  'output-bucket' \
#  'toplevel-output' \
#  'output-tag'

backbone=${BACKBONE:-"tf_efficientnetv2_m"}
input_size=${INPUT_SIZE:-320}
input_crop=${INPUT_CROP:-300}

batch_size=${BATCH_SIZE:-32}
embedding_dim=${EMBEDDING_DIM:-512}
epochs=${EPOCHS:-30}
rand_config=${RAND_CONFIG:-"rand-mstd1"}

# input / output data
results_dir=${RESULTS_DIR:-"/mnt/data/results/triplet-learning"}
data_dir=${DATA_DIR:-"/mnt/data/triplet-learning"}
output_bucket=${OUTPUT_BUCKET:-"gs://ljcv-model-artifacts"}
top_level_output=${TOP_LEVEL_OUTPUT:-"pytorch-metric-learning"}
result_output_dir=${RESULT_OUTPUT_DIR:-"20240621"}
output_tag=${OUTPUT_TAG:-"pymetric.effv2_m.b${batch_size}.dim${embedding_dim}.im${input_size}.defs"}

output_target=${output_bucket}/${top_level_output}
output_path=${results_dir}/${top_level_output}/results/${result_output_dir}/${output_tag}
echo "backbone ${backbone} input-size ${input_size} input-crop ${input_crop} batch-size ${batch_size}"
echo "embedding-dim ${embedding_dim} epochs ${epochs} rand-config '${rand_config}'"
echo "input-dir '${data_dir}' results-dir '${results_dir}' output-path '${output_path}'"
echo "output-bucket '${output_bucket}' output-tag '${output_tag}' target '${output_target}'"

mkdir -p "${output_path}"
python lj_train_model.py \
    --backbone "${backbone}" \
    --input-size "${input_size}" --input-crop "${input_crop}" \
    --batch-size "${batch_size}" \
    --dim "${embedding_dim}" \
    --epochs "${epochs}" \
    --rand_config "${rand_config}" \
    --base-log-dir "${output_path}" \
    "${data_dir}"

echo "uploading '${top_level_output}' to '${output_target}'"
cd "${results_dir}" || exit 1
gsutil -m rsync -r "${top_level_output}/" "${output_target}/"
echo "Completed upload"
