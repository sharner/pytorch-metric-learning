# lj_triplet_main.py: Manage creation of triplet samples and writing them to json.

import argparse
import lj_triplet_sampling as lj
import os

parser = argparse.ArgumentParser(description='Create/Check Triplet sample file')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--read-json', default=None,
                    help='Read given json file')
parser.add_argument('--batch', default=8, help='batch size', type=int)
parser.add_argument('--anal-avail', action='store_true',
                    help='Analyze and report available data')
parser.add_argument('--anal-triplet', action='store_true',
                    help='Analyze and report available data')
parser.add_argument('--write-json', default=None,
                    help="Write triplets to json file")
parser.add_argument('--allow-copies', action='store_true',
                    help='Allow repeats of images in batches until all images are used at least once')

def dump_analysis(analysis):
    nc, ni, avg, mini, maxi = analysis
    print("nclasses {} nimages {} images-per-class {} min images in class {} max images in class {}"
          .format(nc, ni, avg, mini, maxi))

def main():
    args = parser.parse_args()
    train_dir = os.path.join(args.data, 'train')
    if args.read_json:
        json_file = os.path.join(args.data, args.read_json)
        triplet_samples = lj.lj_triplet_read(json_file)
    else:
        triplet_samples = lj.lj_triplet_sampling(train_dir, args.batch, args.allow_copies, weights=None)

    if args.anal_avail:
        samples, _ = lj.lj_list_available(train_dir)
        dump_analysis(lj.lj_analyze(samples))

    if args.anal_triplet:
        dump_analysis(lj.lj_analyze(triplet_samples))

    if args.write_json:
        json_file = os.path.join(args.data, args.write_json)
        lj.lj_triplet_write(json_file, triplet_samples)

if __name__ == '__main__':
    main()
