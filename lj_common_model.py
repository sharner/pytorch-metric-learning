import argparse, os
import pytorch_metric_learning
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
            self.samples = lj.lj_triplet_read(triplet_json_file)
        elif compute_triplet:
            self.samples = lj.lj_triplet_sampling(root, batch_size, allow_copies, weights=weights)
        self.n_classes, self.n_images, _, _, _ = lj.lj_analyze(self.samples)

class MetricLossAccumGrad(pytorch_metric_learning.trainers.MetricLossOnly):
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
            
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    # classes = [int(x) for x in classes] # not sure about this.  Training doesn't assume integer labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def image_name_to_class(directory: str) -> Dict[str, str]:
    img_clz = dict()
    for clz in os.listdir(directory):
        for img_name in os.listdir(os.path.join(directory, clz)):
            img_clz[img_name] = clz
    return img_clz

def image_name_to_index_dictionaries(directory: str) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Return list of classes, class to index and image to class index dictionaries
    """
    classes, class_to_idx = find_classes(directory)
    img_clz = image_name_to_class(directory)

    img_to_idx = dict()
    for img in img_clz:
        img_to_idx[img] = class_to_idx[img_clz[img]]
    return classes, class_to_idx, img_to_idx

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
    parser.add_argument('--output-dim', default=1792, type=int,
                        help='dimensionality of model output')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay', dest='weight_decay')
    parser.add_argument('--margin', default=0.01, type=float, help='margin')
    parser.add_argument('--backbone', default='efficientnet-b7', type=str,
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
