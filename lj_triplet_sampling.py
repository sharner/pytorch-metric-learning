# Take root directory and shuffle into batches

import os
import os.path
import json
import random
import sys
import copy
from typing import Dict, List, Optional, Tuple

# from PIL import Image

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def lj_list_available(
        directory: str,
) -> Tuple[List[Tuple[str, int]], Dict[str,int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.

    Note:  This is compatible with list of samples produced folder.make_dataset
    """
    directory = os.path.expanduser(directory)

    _, class_to_idx = find_classes(directory)

    samples = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                samples.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        raise FileNotFoundError(msg)

    return samples, class_to_idx

def lj_available_images_per_class(
        directory: str,
) -> Tuple[Dict[int, List[str]], List[Tuple[str, int]], set[int], Dict[str,int]]:
    samples, class_to_idx = lj_list_available(directory)

    # make list of class indices
    class_idx_list = [class_to_idx[c] for c in class_to_idx]
    available_images_per_class = {}
    for path, class_idx in samples:
        if not class_idx in available_images_per_class:
            available_images_per_class[class_idx] = set()
        available_images_per_class[class_idx].add(path)
    return available_images_per_class, set(class_idx_list), class_to_idx

def lj_next_anchor_set(class_idx_list : set,
                       n_anchor_class: int,
                       available_images : Dict[int, set],
                       allow_copies) -> Tuple[set[int], set[int]]:
    """Choose the next set of n_ref_classes classes
    """
    next_class_idx_list = [c for c in class_idx_list]
    random.shuffle(next_class_idx_list)
    
    # Make sure that there are at least 2 images for any returned if we don't allow copies
    min_anchor_images = 1 if allow_copies else 2

    anchor_class_idx_list = set()
    for candidate in next_class_idx_list:
        if len(available_images[candidate]) >= min_anchor_images:
            anchor_class_idx_list.add(candidate)
            if len(anchor_class_idx_list) == n_anchor_class:
                break
    
    # If we have no more candidate classes, we are done.
    if len(anchor_class_idx_list) == 0:
        return [], class_idx_list

    # if we allow copies of images and we don't yet have enough
    # anchor classes, randomly choose the missing classes
    if allow_copies and len(anchor_class_idx_list) < n_anchor_class:
        n_missing = n_anchor_class - len(anchor_class_idx_list)
        remaining_classes = list(class_idx_list - anchor_class_idx_list)
        random.shuffle(remaining_classes)

        # take the first n_missing classes to pad out our anchor class list
        # extra_anchor = set(remaining_classes[:n_missing])
        anchor_class_idx_list.update(remaining_classes[:n_missing])

    # if we allow copies, allow using classes without any available images as query classes
    min_query_images = 0 if allow_copies else 2
    query_class_idx_list = class_idx_list - anchor_class_idx_list
    avail_query_classes = filter(lambda x: len(available_images[x]) >= min_query_images, query_class_idx_list)
    return anchor_class_idx_list, avail_query_classes

def lj_image_samples(class_set : set[int],
                     avail_images_per_class : Dict[int, set],
                     all_images_per_class: Dict[int, set],
                     allow_copies) -> List[Tuple[str, int]]:
    """
    Return list of 2 randomly selected images per specified class.
    Remove images from avail_images_per_class.
    Note: MUST be at least 2 images for every class index
    specified.
    """
    next_samples = []
    for cidx in class_set:
        avail_images_list = [p for p in avail_images_per_class[cidx]]
        all_images_list = [p for p in all_images_per_class[cidx]]
        if len(avail_images_list) > 1:
            random.shuffle(avail_images_list)
            next_images = avail_images_list[:2]
        elif allow_copies and len(all_images_list) > 0:
            # randomly select the number of missing images from
            # all available images
            next_images = avail_images_list
            n_missing = 2 - len(avail_images_list)
            for _ in range(n_missing):
                nextidx = random.randint(0, len(all_images_list)-1)
                next_images.append(all_images_list[nextidx])
        else:
            continue
        next_samples +=  [(p, cidx) for p in next_images]

        # remove the sampled images from the available list
        # Since we allow copies, it is possible the path is not in the
        # remaining available images
        for path in next_images:
            if path in avail_images_per_class[cidx]:
                avail_images_per_class[cidx].remove(path)
    return next_samples

def lj_random_class_sample(class_set : set[int],
                           n_anchor_class : int) -> List[int]:
    """
    Return randomly selected set of n_anchor_class out of
    given class_set
    """
    next_class_idx_list = [ci for ci in class_set]
    random.shuffle(next_class_idx_list)
    return next_class_idx_list[:n_anchor_class]

def lj_triplet_sampling(
    directory: str,
    batch_size: int,
    allow_copies,
    weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Tuple[str, int]]:
    """
    Creates list of triplet samples that can fit within one batch.

    * Pick batch/4 classes randomly.  If randomly picked class does not have
      2 or more images remaining, pick another class.  If none available,
      terminate algorithm.
    * Pick 2 images per class
    * For each picked class
        * pick an available class outside of 4 chosen.
            * if weights available, use weights to determine which class
            * else randomly pick a class
        * Again pick 2 images per class.  If a class doesn't have 2 available images
          pick another class.  If no other class available, terminate algorithm
    """
    all_images_per_class, all_classes, class_to_idx = lj_available_images_per_class(directory)
    available_images_per_class = copy.deepcopy(all_images_per_class)

    weights_by_class_idx = None
    if weights:
        for cl in weights:
            cl_idx = class_to_idx[cl]
            weights_by_class_idx[cl_idx] = {}
            for ocl, weight in weights[cl]:
                ocl_idx = class_to_idx[ocl]
                weights_by_class_idx[cl_idx][ocl_idx] = weight

    n_anchor_class = batch_size // 4
    triplet_samples = []

    # continue allocating until we have consumed as much as we possibly can
    while True:
        next_anchor_set, next_query_set = lj_next_anchor_set(all_classes, n_anchor_class,
                                                             available_images_per_class,
                                                             allow_copies)
        if len(next_anchor_set) != n_anchor_class:
            break

        # reference class set is next_anchor_set; disjoint set of
        # classes we will pair with them is next_query_set.  If weights are
        # provided, order by weights and once weights are exhausted, randomly
        # order rest of classes.  If no weights provided, just randomly
        # order disjoint set.

        candidate_samples = lj_image_samples(next_anchor_set,
                                             available_images_per_class,
                                             all_images_per_class,
                                             allow_copies)

        # return n_anchor_set query classes.  If don't have enough
        # then we terminate the algorithm.
        # TODO - handle weights
        next_sample_query_classes = lj_random_class_sample(next_query_set, n_anchor_class)
        if len(next_sample_query_classes) != n_anchor_class:
            break

        # have enough to add a batch.
        triplet_samples += candidate_samples
        triplet_samples += lj_image_samples(next_sample_query_classes,
                                            available_images_per_class,
                                            all_images_per_class,
                                            allow_copies)
    return triplet_samples

def lj_triplet_read(path : str) -> List[Tuple[str, int]]:
    """
    Read json file written by lj_triplet_write
    """
    with open(path, "r") as infile:
        samples = json.load(infile)

    # Manually convert list of list to list of tuples.  Could create decoder
    # but that seems more trouble than it is worth
    triplet_samples = []
    for sample in samples:
        triplet_samples.append(tuple(sample))
    return triplet_samples

def lj_triplet_write(path : str, triplets : List[Tuple[str, int]]):
    """
    Write triplet samples to given json file
    """ 
    with open(path, "w") as outfile:
        json.dump(triplets, outfile)

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
