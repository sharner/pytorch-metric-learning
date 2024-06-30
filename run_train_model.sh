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

backbone=${1}
input_size=$2
input_crop=$3

batch_size=$4
embedding_dim=$5
epochs=$6
rand_config=${7}

# input / output data
results_dir=${8}
data_dir=${9}
output_bucket=${10}
top_level_output=${11}
result_output_dir=${12}
output_tag=${13}

output_target=${output_bucket}/${top_level_output}
output_path=${results_dir}/${top_level_output}/results/${result_output_dir}/${output_tag}
echo "backbone $backbone input-size $input_size input-crop $input_crop batch-size $batch_size"
echo "embedding-dim $embedding_dim epochs $epochs rand-config '$rand_config'"
echo "input-dir '$data_dir' results-dir '$results_dir' output-path '$output_path'"
echo "output-bucket '$output_bucket' output-tag '$output_tag' target '$output_target'"

mkdir -p $output_path
python lj_train_model.py \
    --backbone ${backbone} \
    --input-size ${input_size} --input-crop ${input_crop} \
    --batch-size $batch_size \
    --dim $embedding_dim \
    --epochs $epochs \
    --rand_config "$rand_config" \
    --base-log-dir ${output_path} \
    $data_dir

echo "uploading '$top_level_output' to '$output_target'"
cd $results_dir
gsutil -m rsync -r ${top_level_output}/ ${output_target}/
echo "Completed upload"
