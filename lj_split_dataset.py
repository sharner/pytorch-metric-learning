#
# split a dataset for triplet learning
#
import lj_common_model as lj_com
import argparse, os

# SETTINGS
parser = argparse.ArgumentParser(description='lj_split_dataset')
parser.add_argument('data', help='path to complete dataset')
parser.add_argument('-t', '--target', type=str, required=True,
                    help="target data directory")
parser.add_argument('-s', '--split-num', default=-1, type=int,
                    help='number of data loading workers')
parser.add_argument('--prob', default=0.90, type=float,
                    help='Split using probability instead of number of directories')
parser.add_argument('-f', '--force', default=False, action='store_true',
                    help='force overwrite of train/test')

args = parser.parse_args()

train_dir = os.path.join(args.target, "train")
test_dir = os.path.join(args.target, "test")
lj_com.lj_split_dataset(args.data, args.force, train_dir, test_dir, args.prob, args.split_num)