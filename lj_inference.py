# lj_inference.py
import os

from efficientnet_pytorch import EfficientNet
import lj_common_model as lj_com
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from timm.models import create_model, list_models
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def create_embedder(trunk_output_size, embedding_dim, embedder_checkpoint):
    embedder = lj_com.MLP([trunk_output_size, trunk_output_size/2, embedding_dim])
    checkpoint = torch.load(embedder_checkpoint)
    embedder.load_state_dict(checkpoint)
    return embedder

def create_trunk(backbone, output_dim, trunk_checkpoint):
    """
    Given a backbone and a checkpoint file, return a trunk
    """
    # For using TIMM models.  We need to switch over to timm
    # trunk = create_model(effnet, num_classes=num_classes, pretrained=pretrained)
    # trunk.reset_classifier(0)
    try:
        trunk = EfficientNet.from_pretrained(backbone, output_dim)
    except:
        print("Model {} not found!".format(backbone))
        print("List of models:\n{}".format(list_models()))
        raise

    num_ftrs = trunk._fc.in_features
    trunk._fc = nn.Linear(num_ftrs, output_dim)
    trunk.set_swish(memory_efficient=False)

    checkpoint = torch.load(trunk_checkpoint)
    trunk.load_state_dict(checkpoint)
    return trunk

def create_inference_model(trunk, embedder, match_finder, val_dataset):
    inference_model = InferenceModel(trunk=trunk, embedder=embedder, match_finder=match_finder)
    inference_model.train_knn(val_dataset)
    return inference_model

def build_inference_model(args):
    toplevel_dir = args.data
    output_dim = args.output_dim
    input_dim_resize = args.input_size
    input_dim_crop = args.input_crop
    embedding_dim = args.dim
    eval_device = torch.device(args.eval_device)
    testdir = os.path.join(toplevel_dir, "test")

    embedding_checkpoint_file = os.path.join(args.base_output_dir, args.embedder)
    trunk_checkpoint_file = os.path.join(args.base_output_dir, args.trunk)

    val_dataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(input_dim_resize),
            transforms.CenterCrop(input_dim_crop),
            transforms.ToTensor(),
            lj_com.normalize,
        ]))

    embedder = create_embedder(output_dim, embedding_dim, embedding_checkpoint_file)
    embedder = torch.nn.DataParallel(embedder).to(eval_device)

    trunk = create_trunk(args.backbone, output_dim, trunk_checkpoint_file)
    trunk = torch.nn.DataParallel(trunk.to(eval_device))

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=args.similarity_threshold)
    inference = create_inference_model(trunk, embedder, match_finder, val_dataset)
    return inference

def main():
    parser = lj_com.create_parser()
    parser.add_argument('--base-output-dir', default=".", type=str,
                        help = 'base directory for model output')
    parser.add_argument('--embedder', required=True, type=str,
                        help="embedder checkpoint file")
    parser.add_argument('--trunk', required=True, type=str,
                        help="trunk checkpoint file")
    parser.add_argument('--similarity-threshold', default=0.7, type=float,
                        help="Cosine similarity threshold")
    parser.add_argument('--eval-device', choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    inference = build_inference_model(args)

    # create list of class indices from training directory
    traindir = os.path.join(args.data, "train")
    _, class_to_idx, image_name_to_idx = lj_com.image_name_to_index_dictionaries(traindir)


if __name__ == '__main__':
    main()