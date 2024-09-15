# lj_inference.py
import os
import collections
import random

from efficientnet_pytorch import EfficientNet
import lj_common_model as lj_com
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pytorch_metric_learning.utils import common_functions as c_f
from timm.models import create_model, list_models
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def create_embedder(trunk_output_size, embedding_dim, embedder_checkpoint):
    embedder = lj_com.MLP([trunk_output_size, trunk_output_size/2, embedding_dim])
    checkpoint = torch.load(embedder_checkpoint)
    embedder.load_state_dict(checkpoint)
    return embedder

def create_trunk(backbone, nclasses, trunk_checkpoint):
    """
    Given a backbone and a checkpoint file, return a trunk
    """
    # For using TIMM models.  We need to switch over to timm
    # trunk = create_model(effnet, num_classes=num_classes, pretrained=pretrained)
    # trunk.reset_classifier(0)
    try:
        trunk = create_model(backbone, num_classes=nclasses, pretrained=True)
        # trunk = EfficientNet.from_pretrained(backbone, nclasses)
        # num_ftrs = trunk._fc.in_features
        # trunk._fc = nn.Linear(num_ftrs, nclasses)
        # trunk.set_swish(memory_efficient=False)
    except:
        print("Model {} not found!".format(backbone))
        print("List of models:\n{}".format(list_models()))
        raise

    trunk_output_size = trunk.classifier.in_features
    trunk.classifier = nn.Identity()
    checkpoint = torch.load(trunk_checkpoint)
    trunk.load_state_dict(checkpoint)
    return trunk, trunk_output_size

def create_inference_model(trunk, embedder, match_finder, val_dataset):
    inference_model = InferenceModel(trunk=trunk, embedder=embedder, match_finder=match_finder)
    inference_model.train_knn(val_dataset)
    return inference_model

def build_inference_model_from_cli(args):
    toplevel_data_dir = args.data
    nclasses = args.output_dim # number of classes
    input_dim_resize = args.input_size
    input_dim_crop = args.input_crop
    embedding_dim = args.dim
    eval_device = torch.device(args.eval_device)

    embedding_checkpoint_file = os.path.join(args.base_output_dir, args.embedder)
    trunk_checkpoint_file = os.path.join(args.base_output_dir, args.trunk)

    return build_inference_model(toplevel_data_dir, args.backbone, nclasses, eval_device,
                                 input_dim_resize, input_dim_crop,
                                 embedding_dim, args.similarity_threshold,
                                 trunk_checkpoint_file, embedding_checkpoint_file)

def build_inference_model(toplevel_data_dir, backbone, nclasses, eval_device,
                          input_dim_resize, input_dim_crop,
                          embedding_dim, similarity_threshold,
                          trunk_checkpoint_file, embedding_checkpoint_file):
    """
    Return an inference model and a dataset for given backbone, trunk and
    embedding checkpoint files.
    """

    if nclasses == 0:
        # use training dataset to determine number of classes
        traindir = os.path.join(toplevel_data_dir, "train")
        train_dataset = lj_com.TrainLoader(
            traindir,
            transforms.Compose([
                transforms.Resize(input_dim_resize), # Remove this when using EfficientNet - or test with it
                transforms.RandomResizedCrop(input_dim_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lj_com.normalize
                ]),
        )
        # Retrieve the number of classes from the loader
        nclasses = train_dataset.n_classes
        print("nclasses {}".format(nclasses))

    testdir = os.path.join(toplevel_data_dir, "test")
    val_dataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(input_dim_resize),
            transforms.CenterCrop(input_dim_crop),
            transforms.ToTensor(),
            lj_com.normalize,
        ]))

    trunk, trunk_output_size  = create_trunk(backbone, nclasses, trunk_checkpoint_file)
    print("trunk output size {} trunk file {}".format(trunk_output_size, trunk_checkpoint_file))
    trunk = torch.nn.DataParallel(trunk.to(eval_device))

    print("nclasses {} embedding file {}".format(nclasses, embedding_checkpoint_file))
    embedder = create_embedder(trunk_output_size, embedding_dim, embedding_checkpoint_file)
    embedder = torch.nn.DataParallel(embedder).to(eval_device)

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=similarity_threshold)
    inference = create_inference_model(trunk, embedder, match_finder, val_dataset)
    labels_to_indices = c_f.get_labels_to_indices(val_dataset.targets)
    indices_to_labels = collections.defaultdict(list)
    for k, v in labels_to_indices.items():
        for i in v:
            indices_to_labels[int(i)] = k

    return inference, val_dataset, labels_to_indices, indices_to_labels

class NN_query_image:
    FIRST = 1
    RANDOM = 2
    ALL = 3

def query_image_set(n_images, query_image):
    if query_image == NN_query_image.ALL:
        return range(0, n_images)
    query_index = random.randint(0, n_images-1) if query_image == NN_query_image.RANDOM else 0
    return [query_index]

def nearest_neighbors(inference_model, dataset_val, labels_to_indices, indices_to_labels, top_N=10,
                      query_image=NN_query_image.RANDOM):
    """
    Choose query image from each label and determine how many images match query in top-N images
    """
    match_top_n = [0] * top_N
    at_least_one = [0] * top_N
    total_count = [0] * top_N
    precision_at_k = [-1] * top_N
    recall_at_k = [-1] * top_N

    distances_for_matches = []
    distances_for_mismatches = []
    for k in labels_to_indices:
        indices_of_label = labels_to_indices[k]
        n_images = len(indices_of_label)
        if n_images == 0:
            continue
        indices = query_image_set(n_images, query_image)
        for query_index in indices:
            index = indices_of_label[query_index]
            # print("Next label {} n_images {} index {}".format(k, n_images, index))
            dmatch, dmiss = nearest_neighbors_to_index(inference_model, dataset_val, index, indices_to_labels, match_top_n, at_least_one, total_count)
            distances_for_matches.extend(dmatch)
            distances_for_mismatches.extend(dmiss)
        for k in range(top_N):
            tq_at_k = total_count[k]
            if tq_at_k > 0:
                precision_at_k[k] = float(match_top_n[k])/float(tq_at_k*(k+1))
                recall_at_k[k] = float(at_least_one[k]/float(tq_at_k))

    return precision_at_k, recall_at_k, distances_for_matches, distances_for_mismatches

def nearest_neighbors_to_index(inference_model, dataset_val, class_index, indices_to_labels, match_count, at_least_one, total_count):
    top_N = len(match_count)
    img = dataset_val[class_index][0].unsqueeze(0)
    class_label = dataset_val[class_index][1]
    class_label_from_index = indices_to_labels[class_index]
    if class_label_from_index != class_label:
        print("Error: dataset class label {} != label from index {}".format(class_label, class_label_from_index))
    distances, indices = inference_model.get_nearest_neighbors(img, k=top_N+1)
    # print("class label {} class index {}".format(class_label, class_index))

    correct = []
    one_match = []
    distances_by_match = []
    distances_by_mismatch = []
    at_least_one_match_v = False
    flattened_indices = indices.cpu()[0]
    flattened_distances = distances.cpu()[0]
    top_i = 1
    for result_index, result_distance in zip(flattened_indices, flattened_distances):
        if class_index == result_index or top_i > top_N:
            continue
        result_label = indices_to_labels[int(result_index)]
        is_match = result_label == class_label
        correct.append(is_match)

        at_least_one_match_v |= is_match
        one_match.append(at_least_one_match_v)

        desc = (float(result_distance), top_i, int(class_index), int(result_index), class_label, result_label)
        if is_match:
            distances_by_match.append(desc)
        else:
            distances_by_mismatch.append(desc)

        top_i += 1

    # count how many labels are correct within top-N
    # the only reason we are taking the total count is because
    # classes are not guaranteed to have at least top-N images so
    # we have to keep track of the total images in top-N separately.
    current_correct_count = 0
    for top_i in range(top_N):
        if top_i >= len(correct):
            break

        # Count how many images in top-N are correct
        if correct[top_i]:
            current_correct_count += 1
        match_count[top_i] += current_correct_count

        # Count how many queries have at least one match in top-N
        if one_match[top_i]:
            at_least_one[top_i] += 1
        
        # Count how many total targets in top-N
        total_count[top_i] += 1
    return distances_by_match, distances_by_mismatch

def create_inference_parser():
    parser = lj_com.create_parser()
    parser.add_argument('--base-output-dir', default=".", type=str,
                        help = 'base directory for model output')
    parser.add_argument('--embedder', required=True, type=str,
                        help="embedder checkpoint file")
    parser.add_argument('--trunk', required=True, type=str,
                        help="trunk checkpoint file")
    parser.add_argument('--similarity-threshold', default=0.7, type=float,
                        help="Cosine similarity threshold")
    parser.add_argument('--eval-device', choices=["cpu", "gpu", "cuda"], default="gpu")
    return parser

def main():
    parser = create_inference_parser()
    args = parser.parse_args()

    inference, val_dataset, labels_to_indices, indices_to_labels = build_inference_model(args)

    # create list of class indices from training directory
    traindir = os.path.join(args.data, "train")


if __name__ == '__main__':
    main()