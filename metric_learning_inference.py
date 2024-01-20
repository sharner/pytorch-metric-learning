import logging
import matplotlib.pyplot as plt
import numpy as np
import record_keeper
import torch
import sys, os
import torch.nn as nn
import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms
import lj_common_model as lj_com

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.append(os.path.join('/layerjot', 'pytorch-image-models'))
from timm.models import create_model, list_models
from efficientnet_pytorch import EfficientNet

parser = lj_com.create_parser()
args = parser.parse_args()

odels_with_backbone = list_models(args.backbone)
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

# EMBEDDER is lj_com.MLP

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
embedder = torch.nn.DataParallel(lj_com.MLP([trunk_output_size, trunk_output_size/2, embedding_dim]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=lr, weight_decay=wd
)
loss_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)
