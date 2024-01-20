import torch.nn as nn
import argparse

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
