# Script version of LJCV-MetricLossOnly.ipynb

import logging
import matplotlib.pyplot as plt
import numpy as np
# import record_keeper
import torch
import sys, os
import torch.nn as nn
import argparse
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import lj_triplet_sampling as lj
# import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.append(os.path.join('/layerjot', 'pytorch-image-models'))
from timm.models import create_model, list_models
from efficientnet_pytorch import EfficientNet

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

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

args = parser.parse_args()
models_with_backbone = list_models(args.backbone)
toplevel_dir = args.data
output_dim = args.output_dim
input_dim_resize = args.input_size
input_dim_crop = args.input_crop
embedding_dim = args.dim

# Set other training parameters
batch_size = args.batch_size
num_epochs = args.epochs
margin = args.margin
wd = args.weight_decay
m_per_class = 2
eval_batch_size = args.eval_batch_size
eval_k="max_bin_count"
patience=3
lr=args.lr
model_lr=args.modellr
# eval_k=10
# Need to run eval on the CPU because training holds onto GPU memory
eval_device = torch.device("cpu")

# NOTE: I don't think these params are going to do the right thing - revisit this choice
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
traindir = os.path.join(toplevel_dir, "train")
testdir = os.path.join(toplevel_dir, "test")
pretrained=True

# EMBEDDER

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

# TRUNK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function

# For using TIMM models
# trunk = create_model(effnet, num_classes=num_classes, pretrained=pretrained)
# trunk.reset_classifier(0)

# Using large pretrained efficientnet
#trunk = EfficientNet.from_pretrained("efficientnet-b7", output_dim)
try:
    trunk = EfficientNet.from_pretrained(args.backbone, output_dim)
except:
    print("Model {} not found!".format(args.backbone))
    print("List of models:\n{}".format(list_models()))
    raise

num_ftrs = trunk._fc.in_features
trunk._fc = nn.Linear(num_ftrs, output_dim)
trunk.set_swish(memory_efficient=False)

# checkpoint = torch.load(args.resume)
# trunk.load_state_dict(checkpoint)
# Set classification head to identity
trunk_output_size = output_dim
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs the embedding dimension
embedder = torch.nn.DataParallel(MLP([trunk_output_size, trunk_output_size/2, embedding_dim]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=lr, weight_decay=wd
)
loss_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)

class TrainLoader(datasets.ImageFolder):
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

train_dataset = TrainLoader(
    traindir,
    transforms.Compose([
        transforms.Resize(input_dim_resize), # Remove this when using EfficientNet - or test with it
        transforms.RandomResizedCrop(input_dim_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ]),
    triplet_json_file=args.triplet_json,
    compute_triplet=args.compute_triplet,
    batch_size = batch_size,
    allow_copies = args.allow_copies
)

# Retrieve the number of classes from the loader
num_classes = train_dataset.n_classes

# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.Resize(input_dim_resize), # Remove this when using EfficientNet - or test with it
#         transforms.RandomResizedCrop(input_dim_crop),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))

# If use external triplet specification, do not use sampler.
if args.triplet_json or args.compute_triplet:
    sampler = None
else:
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets,
                                        m=m_per_class,
                                        length_before_new_iter=len(train_dataset))

val_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(input_dim_resize),
        transforms.CenterCrop(input_dim_crop),
        transforms.ToTensor(),
        normalize,
    ]))

# Loss, miner, sampler

# Set the loss function
loss = losses.TripletMarginLoss(margin=margin)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=margin)

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
    # 'metric_loss_optimizer': loss_optimizer,
}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

# HOOKS

log_dir = os.path.join(args.base_log_dir, "lj_logs")
tensor_dir = os.path.join(args.base_log_dir, "lj_tensorboard")
model_dir = os.path.join(args.base_log_dir, "lj_saved_models")

record_keeper, _, _ = logging_presets.get_record_keeper(
    log_dir, tensor_dir
)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = model_dir

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()


# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    # visualizer=umap.UMAP(),
    # visualizer_hook=visualizer_hook,
    dataloader_num_workers=1,
    batch_size=eval_batch_size,
    data_device=eval_device,
    accuracy_calculator=AccuracyCalculator(k=eval_k, device=eval_device),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, model_folder, test_interval=1, patience=patience
)

# TRAINER

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

trainer = MetricLossAccumGrad(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    train_dataset,
    mining_funcs=mining_funcs,
    sampler=sampler,
    dataloader_num_workers=2,
    # end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

trainer.train(num_epochs=num_epochs)
