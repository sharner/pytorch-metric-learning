# Script version of LJCV-MetricLossOnly.ipynb

import os
import lj_inference as ljinf

parser = ljinf.create_inference_parser()
parser.add_argument('--query-image', default='all', type=str, choices=['random', 'first', 'all'],
                    help='What query (or set of query) images to use for evaluation')
    
args = parser.parse_args()
embedding_checkpoint_file = os.path.join(args.base_output_dir, args.embedder)
trunk_checkpoint_file = os.path.join(args.base_output_dir, args.trunk)

inference_model, dataset_val, labels_to_indices, indices_to_labels = \
    ljinf.build_inference_model(args.data, args.backbone, args.output_dim, args.eval_device,
                                args.input_size, args.input_crop,
                                args.dim, args.similarity_threshold,
                                trunk_checkpoint_file, embedding_checkpoint_file)
print("len labels_to_indices {}".format(len(labels_to_indices)))

query_option = ljinf.NN_query_image.ALL
precision_at_k, recall_at_k, d_for_match, d_for_mismatch = ljinf.nearest_neighbors(inference_model, dataset_val, labels_to_indices, indices_to_labels, query_image=query_option)
print("precisions@k {} recall@k {}".format(precision_at_k, recall_at_k))
# For sheet
print("recall@1 recall@3 recall@5 recall@10 precision@10 {}".format(recall_at_k[0], recall_at_k[2], recall_at_k[4], recall_at_k[9], precision_at_k[9]))
print("{}, {}, {}, {}, {}".format(recall_at_k[0], recall_at_k[2], recall_at_k[4], recall_at_k[9], precision_at_k[9]))

