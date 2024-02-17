import argparse, os
import lj_common_model as lj_com
import lj_triplet_sampling as lj_trip
import numpy as np
import pytorch_metric_learning
from pytorch_metric_learning import trainers
import sys
import torch
import torch.nn as nn
import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

# common globals

# NOTE: I don't think these params are going to do the right thing - revisit this choice
# ??????
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
normalize = torchvision.transforms.Normalize(mean=mean, std=std)

# common classes

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class TrainLoader(torchvision.datasets.ImageFolder):
    """
    Allow external specification of triplet samples
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        triplet_json_file : Optional[str] = None,
        compute_triplet : Optional[bool] = False,
        batch_size : Optional[int] = 8,
        allow_copies : Optional[bool] = False,
        weights : Optional[Dict[str, Dict[str, float]]] = None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        if triplet_json_file:
            self.samples = lj_trip.lj_triplet_read(triplet_json_file)
        elif compute_triplet:
            self.samples = lj_trip.lj_triplet_sampling(root, batch_size, allow_copies, weights=weights)
        self.n_classes, self.n_images, _, _, _ = lj_com.lj_analyze(self.samples)

class MetricLossAccumGrad(trainers.MetricLossOnly):
    def __init__(self, *args, accumulation_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def forward_and_backward(self):
        self.zero_losses()
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        self.loss_tracker.update(self.loss_weights)
        self.backward()
        self.clip_gradients()
        if ((self.iteration + 1) % self.accumulation_steps == 0) or ((self.iteration + 1) == np.ceil(len(self.dataset) / self.batch_size)):
            self.step_optimizers()
            self.zero_grad()

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        with torch.cuda.amp.autocast():
            embeddings = self.compute_embeddings(data)
            indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
            self.losses["metric_loss"] = self.maybe_get_metric_loss(
                embeddings, labels, indices_tuple
            )

# Common methods

def lj_count_classes(triplets : List[Tuple[str, int]]) -> Dict[int, int]:
    """
    Return dictionary of class index to count of images
    """
    # dictionary of class -> n images
    class_count = {}
    for _, clidx in triplets:
        if clidx not in class_count:
            class_count[clidx] = 0
        class_count[clidx] += 1
    return class_count

def lj_analyze(triplets : List[Tuple[str, int]]) -> Tuple[int, int, float, int, int]:
    """
    Return tuple of (nclasses, nimages, avg_images_per_class, min_images_in_class, max_images_in_class)
    """
    class_count = lj_count_classes(triplets)

    n_classes = len(class_count)
    total_images = 0
    max_images = 0
    min_images = sys.maxsize
    for clidx in class_count:
        n_images = class_count[clidx]
        max_images = max(max_images, n_images)
        min_images = min(min_images, n_images)
        total_images += n_images

    avg = float(total_images)/float(n_classes)
    return (n_classes, total_images, avg, min_images, max_images)
     
def create_parser():
    # SETTINGS
    parser = argparse.ArgumentParser(description='PyTorch Metric Training')
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('--triplet-json', type=str, default=None,
                        help="triplet samples as a json file")
    parser.add_argument('--compute-triplet', default=False,
                        action='store_true', help="compute the set of triplets")
    parser.add_argument('--allow-copies', default=False,
                        action='store_true', help="if compute triplets, then allow image copies")
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size')
    parser.add_argument('--eval-batch-size', default=1, type=int,
                        help='evaluation batch size')
    parser.add_argument('--modellr', default=0.00001, type=float,
                        help='initial model learning rate')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='optimizer learning rate')
    parser.add_argument('--dim', default=128, type=int,
                        help='dimensionality of embeddings')
    parser.add_argument('--output-dim', default=0, type=int,
                        help='n trained classes; if 0 use training set (for eval only)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay', dest='weight_decay')
    parser.add_argument('--margin', default=0.01, type=float, help='margin')
    parser.add_argument('--backbone', default='tf_efficientnet_b7', type=str,
                        help='type of model to use: "resnet" for Resnet152, "mobilenet" for Mobilenet_v2, "efficientb7" + "efficientb0" for Efficient Net B0 and B7, "efficientlite" for Efficient Net Lite')
    parser.add_argument('--rand_config', default='rand-mstd1',
                        help='Random augment configuration string')
    parser.add_argument('--resume', default=None, help='resume from given file')
    parser.add_argument('--eval-only', default=False, action='store_true',
                        help='evaluate model only')
    parser.add_argument('--base-log-dir', default=".", type=str,
                        help = 'base directory for all logs')
    parser.add_argument('--input-size', default=650, type=int,
                        help='resize images to this size for input')
    parser.add_argument('--input-crop', default=600, type=int,
                        help='random crop images to this size')
    return parser
