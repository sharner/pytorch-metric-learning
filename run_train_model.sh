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
#  'output-tag'

backbone=$1
input_size=$2
input_crop=$3

batch_size=$4
embedding_dim=$5
epochs=$6
rand_config=$7

# input / output data
results_dir=$8
data_dir=$9
output_bucket=${10}
output_tag=${11}
output_target=${output_bucket}${output_tag}

echo "backbone $backbone input-size $input_size input-crop $input_crop batch-size $batch_size"
echo "embedding-dim $embedding_dim epochs $epochs rand-config '$rand_config'"
echo "results-dir '$results_dir' input-dir '$data_dir'"
echo "output-bucket '$output_bucket' output-tag '$output_tag' target '$output_target'"

python lj_train_model.py \
    --backbone ${backbone} \
    --input-size ${input_size} --input-crop ${input_crop} \
    --batch-size $batch_size \
    --dim $embedding_dim \
    --epochs $epochs \
    --rand_config $rand_config \
    --base-log-dir $results_dir \
    $data_dir

echo "uploading '$results_dir' to '$output_target'"
gsutil -m rsync -r $results_dir $output_target
