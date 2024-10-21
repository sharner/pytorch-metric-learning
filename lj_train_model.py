# Script version of LJCV-MetricLossOnly.ipynb

import logging
import matplotlib.pyplot as plt
import numpy as np
# import record_keeper
import torch
import os
import torch.nn as nn

import lj_common_model as lj_com

# import umap
from cycler import cycler
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers  # , trainers
# from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from timm.data.auto_augment import rand_augment_transform
from timm.models import create_model, list_models
from timm.data.transforms import RandomResizedCropAndInterpolation

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

# SETTINGS
parser = lj_com.create_parser()
args = parser.parse_args()

models_with_backbone = list_models(args.backbone)
toplevel_dir = args.data
input_dim_resize = args.input_size
input_dim_crop = args.input_crop
embedding_dim = args.dim

# Set other training parameters
batch_size = args.batch_size
num_epochs = args.epochs
margin = args.margin
epsilon = args.epsilon
wd = args.weight_decay
m_per_class = 2
eval_batch_size = args.eval_batch_size
eval_k = "max_bin_count"
patience = 3
lr = args.lr
model_lr = args.modellr
# eval_k = 10

# Loss function parameters
alpha = 2  # just for test for now
beta = 50
base = 0.5

# Need to run eval on the CPU because training holds onto GPU memory
eval_device = torch.device("cpu")

normalize = lj_com.normalize
traindir = os.path.join(toplevel_dir, "train")
testdir = os.path.join(toplevel_dir, "test")
pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset.

if args.rand_config:
    rand_tfm = rand_augment_transform(config_str=args.rand_config, hparams={'img_mean': (104, 117, 128)})
    transform = transforms.Compose([
            RandomResizedCropAndInterpolation(input_dim_crop),
            transforms.RandomHorizontalFlip(),
            rand_tfm,
            transforms.ToTensor(),
            normalize,
    ])
else:
    transform = transforms.Compose([
            normalize,
    ])

train_dataset = lj_com.TrainLoader(
    traindir,
    transform,
    triplet_json_file=args.triplet_json,
    compute_triplet=args.compute_triplet,
    batch_size=batch_size,
    allow_copies=args.allow_copies
)

# Retrieve the number of classes from the loader
num_classes = train_dataset.n_classes

# TRUNK: Set trunk model and replace the softmax layer with an identity function

try:
    trunk = create_model(args.backbone, num_classes=num_classes, pretrained=pretrained)
    # trunk.reset_classifier(0)
    # trunk = EfficientNet.from_pretrained(args.backbone, num_classes)
    # num_ftrs = trunk._fc.in_features
    # trunk._fc = nn.Linear(num_ftrs, num_classes)
    # trunk.set_swish(memory_efficient=False)
except:
    print("Model {} not found!".format(args.backbone))
    print("List of models:\n{}".format(list_models()))
    raise

# checkpoint = torch.load(args.resume)
# trunk.load_state_dict(checkpoint)
# Set classification head to identity
trunk_output_size = trunk.classifier.in_features
trunk.classifier = nn.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# EMBEDDER is lj_com.MLP

# Set embedder model. This takes in the output of the trunk and outputs the embedding dimension
embedder = torch.nn.DataParallel(lj_com.MLP([trunk_output_size, trunk_output_size/2, embedding_dim]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=lr, weight_decay=wd
)
loss_optimizer = torch.optim.Adam(trunk.parameters(), lr=model_lr, weight_decay=wd)

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
# loss = losses.TripletMarginLoss(margin=margin)
loss = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=epsilon)
# miner = miners.BatchEasyHardMiner(
#     pos_strategy=miners.BatchEasyHardMiner.EASY,
#     neg_strategy=miners.BatchEasyHardMiner.HARD)

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

def create_dir_with_permissions(dir_path):
    """
    Create specified directory with permission 0o775.
    Note have to override umask of 0o0022 in order to set
    proper group permissions.  Could use umask of 0o0002,
    but that seems confusing.
    """
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            raise Exception("{} is not a directory!".format(dir_path))
        return
    old_umask = os.umask(0)
    try:
        os.mkdir(dir_path, 0o775)
    finally:
        os.umask(old_umask)

for d in [args.base_log_dir, log_dir, tensor_dir, model_dir]:
    create_dir_with_permissions(d)

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
    # fig = plt.figure(figsize=(20, 15))
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
measures = ("precision_at_1",
            "NMI",
            "AMI",
            "r_precision",
            "mean_average_precision_at_r")
accuracy_calculator = AccuracyCalculator(include=measures,
                                         k=eval_k,
                                         device=eval_device)
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    # visualizer=umap.UMAP(),
    # visualizer_hook=visualizer_hook,
    dataloader_num_workers=1,
    batch_size=eval_batch_size,
    data_device=eval_device,
    accuracy_calculator=accuracy_calculator
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, model_folder, test_interval=1, patience=patience
)

# TRAINER

trainer = lj_com.MetricLossAccumGrad(
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
