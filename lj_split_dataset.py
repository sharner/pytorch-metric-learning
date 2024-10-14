#
# split a dataset for triplet learning
#
import argparse, os, random, shutil


# Split dataset
def lj_split_dataset(dataset_dir, force_write, train_dir, test_dir, split_prob, split_number):
    # if we are forcing writing data, remove the current train and test directories.
    # if we are NOT forcing writing data AND the training directory exists, return.
    if os.path.exists(train_dir):
        if not force_write:
            return
        shutil.rmtree(train_dir)

    # If the test directory exists remove it because we are going to
    # re-write both the training and test datasets
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    parent_dir = os.path.dirname(train_dir)
    os.makedirs(parent_dir, exist_ok=True)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    classes = []
    for filename in os.listdir(dataset_dir):
        fpath = os.path.join(dataset_dir, filename)
        if os.path.isdir(fpath):
            classes.append(filename)

    if split_number <= 0.0:
        assert split_prob > 0, "split prob {} must be > 0 if split number < 0!".format(split_prob)
        split_number = int(split_prob * len(classes))
    
    # Random shuffle takes the first split_number as test classes for N+1 with the
    # remaining classes for training

    random.shuffle(classes)
    n_classes = 1
    for inst_class in classes:
        if n_classes <= split_number:
            # write into test_dir
            dst_class_dir = os.path.join(test_dir, inst_class)
        else:
            # write into train_dir
            dst_class_dir = os.path.join(train_dir, inst_class)
        n_classes += 1

        os.mkdir(dst_class_dir)
        # loop over dat and write into class_dir
        class_data_dir = os.path.join(dataset_dir, inst_class)
        for image_name in os.listdir(class_data_dir):
            src_image_path = os.path.join(class_data_dir, image_name)
            base_image_name, fext = os.path.splitext(image_name)
            if not fext:
                img_file_name = "{}.jpeg".format(base_image_name)
            else:
                img_file_name = image_name

            dst_image_path = os.path.join(dst_class_dir, img_file_name)
            # print("idx {} cp {} -> {}".format(inst_class, src_image_path, dst_image_path))
            shutil.copy(src_image_path, dst_image_path)
        if n_classes % 100 == 0:
            print("processed {} iids".format(n_classes))

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
lj_split_dataset(args.data, args.force, train_dir, test_dir, args.prob, args.split_num)