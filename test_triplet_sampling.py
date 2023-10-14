import lj_triplet_sampling as lj
import os
import copy
    
class TestSampleUtils:
    test_class_dir = "./test_classes"
    exp_class_idx_list = set([0, 1, 2, 3])
    exp_class_to_idx = {'1' : 0, '10' : 1, '20': 2, '5' : 3}
    exp_samples = [
        ('./test_classes/1/1.jpg', 0),
        ('./test_classes/1/2.jpg', 0),
        ('./test_classes/1/3.jpg', 0),
        ('./test_classes/10/103.jpg', 1),
        ('./test_classes/10/105.jpg', 1),
        ('./test_classes/20/205.jpg', 2),
        ('./test_classes/20/210.jpg', 2),
        ('./test_classes/5/51.jpg', 3),
        ('./test_classes/5/53.jpg', 3)
    ]
    exp_avail_per_class = {
        0 : set(['./test_classes/1/1.jpg',
                 './test_classes/1/2.jpg',
                 './test_classes/1/3.jpg']),
        1 : set(['./test_classes/10/103.jpg',
                 './test_classes/10/105.jpg']),
        2 : set(['./test_classes/20/205.jpg',
                 './test_classes/20/210.jpg']),
        3 : set(['./test_classes/5/51.jpg',
                 './test_classes/5/53.jpg'])
    }

    @staticmethod
    def check_image_samples(clidx,
                            exp_remain_length,
                            avail_per_class,
                            all_per_class,
                            allow_copies):
        im_cl = lj.lj_image_samples(set([clidx]), avail_per_class, all_per_class, allow_copies)
        assert len(im_cl) == 2
        assert len(avail_per_class[clidx]) == exp_remain_length

        # check that paths are members of expected paths for class
        sample_path = set([sample[0] for sample in im_cl])
        assert(sample_path.issubset(TestSampleUtils.exp_avail_per_class[clidx]))
    
    def test_find_classes(self):
        exp_classes = [cl for cl in TestSampleUtils.exp_class_to_idx]
        exp_classes = sorted(exp_classes)
        classes, class_to_idx = lj.find_classes(TestSampleUtils.test_class_dir)
        assert classes == exp_classes
        assert class_to_idx == TestSampleUtils.exp_class_to_idx

    def test_available_classes(self):
        samples, class_to_idx = lj.lj_list_available(TestSampleUtils.test_class_dir)
        assert class_to_idx == TestSampleUtils.exp_class_to_idx
        assert samples == TestSampleUtils.exp_samples

    def test_available_images_per_class(self):
        avail_per_class, class_idx_list, class_to_idx = lj.lj_available_images_per_class(TestSampleUtils.test_class_dir)
        assert avail_per_class == TestSampleUtils.exp_avail_per_class
        assert class_to_idx == TestSampleUtils.exp_class_to_idx
        assert class_idx_list == TestSampleUtils.exp_class_idx_list

    def test_next_anchor_set(self):
        avail_per_class, class_idx_list, _ = lj.lj_available_images_per_class(TestSampleUtils.test_class_dir)

        # partition all classes into a 'reference' set and a non-reference set of classes
        ref_class_list, not_ref_class = lj.lj_next_anchor_set(class_idx_list, 2, avail_per_class, False)
        assert len(ref_class_list) == 2
        all_classes = ref_class_list.union(not_ref_class)
        assert all_classes == class_idx_list

    def test_image_samples(self):
        all_per_class, _, _ = lj.lj_available_images_per_class(TestSampleUtils.test_class_dir)
        avail_per_class = copy.deepcopy(all_per_class)
        TestSampleUtils.check_image_samples(0, 1, avail_per_class, all_per_class, False)
        TestSampleUtils.check_image_samples(1, 0, avail_per_class, all_per_class, False)
        TestSampleUtils.check_image_samples(2, 0, avail_per_class, all_per_class, False)
        TestSampleUtils.check_image_samples(3, 0, avail_per_class, all_per_class, False)

    def test_triplet_sampling_no_weight(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, False)
        assert len(samples) == 8 # only enough for 1 batch

        # count the number of images in each class.  Should have
        # 2 from each class
        class_count = lj.lj_count_classes(samples)        
        for c in class_count:
            assert class_count[c] == 2

        assert set(class_count.keys()) == TestSampleUtils.exp_class_idx_list
    
    def test_triplet_sampling_no_weight_2batch(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 4, False)
        assert len(samples) == 8 # enough for 2 batches

        # count the number of images in each class.  Should have
        # 2 from each class
        class_count = lj.lj_count_classes(samples)        
        
        for c in class_count:
            assert class_count[c] == 2

        assert set(class_count.keys()) == TestSampleUtils.exp_class_idx_list

    def test_io(self):
        test_file_path = "./test_classes.json"
        if os.path.isfile(test_file_path):
            os.remove(test_file_path)
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, False)
        lj.lj_triplet_write(test_file_path, samples)

        # read it back
        read_samples = lj.lj_triplet_read(test_file_path)
        assert samples == read_samples

    def test_analyze_triplet(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, False)
        triplet_anal = lj.lj_analyze(samples)
        exp_triplet_anal = (4, 8, 2.0, 2, 2)
        assert triplet_anal == exp_triplet_anal

    def test_analyze_avail(self):
        samples, _ = lj.lj_list_available(TestSampleUtils.test_class_dir)
        avail_anal = lj.lj_analyze(samples)
        exp_avail_anal = (4, 9, 2.25, 2, 3)
        assert avail_anal == exp_avail_anal

    # ------------ Allow image copies ------------------

    def test_next_anchor_set_allow_copies(self):
        avail_per_class, class_idx_list, _ = lj.lj_available_images_per_class(TestSampleUtils.test_class_dir)

        # partition all classes into a 'reference' set and a non-reference set of classes
        ref_class_list, not_ref_class = lj.lj_next_anchor_set(class_idx_list, 2, avail_per_class, True)
        assert len(ref_class_list) == 2
        all_classes = ref_class_list.union(not_ref_class)
        assert all_classes == class_idx_list

    def test_image_samples_allow_copies(self):
        all_per_class, _, _ = lj.lj_available_images_per_class(TestSampleUtils.test_class_dir)
        avail_per_class = copy.deepcopy(all_per_class)
        TestSampleUtils.check_image_samples(0, 1, avail_per_class, all_per_class, True)
        TestSampleUtils.check_image_samples(1, 0, avail_per_class, all_per_class, True)
        TestSampleUtils.check_image_samples(2, 0, avail_per_class, all_per_class, True)
        TestSampleUtils.check_image_samples(3, 0, avail_per_class, all_per_class, True)

    def test_triplet_sampling_no_weight_allow_copies(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, True)
        assert len(samples) == 16 # Should be 2 batches

        # count the number of images in each class.  Should have
        # 4 from each class in order to fill out the 2 batches
        class_count = lj.lj_count_classes(samples)        
        for c in class_count:
            assert class_count[c] == 4

        assert set(class_count.keys()) == TestSampleUtils.exp_class_idx_list
    
    def test_triplet_sampling_no_weight_2batch_allow_copies(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 4, True)

        # we can end up with either 3 or 4 batches
        assert len(samples) == 12 or len(samples) == 16

        # count the number of images in each class.  Should have
        # 2 from each class
        class_count = lj.lj_count_classes(samples)        
        
        # with copies, count will just be a multiple of 2 in this case
        for c in class_count:
            assert (class_count[c] % 2 == 0) and (class_count[c] > 0)

        assert set(class_count.keys()) == TestSampleUtils.exp_class_idx_list

    def test_analyze_triplet_allow_copies(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, True)
        triplet_anal = lj.lj_analyze(samples)
        exp_triplet_anal = (4, 16, 4.0, 4, 4)
        assert triplet_anal == exp_triplet_anal

    # ----------------- Test I/O ------------------------

    def test_io(self):
        test_file_path = "./test_classes.json"
        if os.path.isfile(test_file_path):
            os.remove(test_file_path)
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, False)
        lj.lj_triplet_write(test_file_path, samples)

        # read it back
        read_samples = lj.lj_triplet_read(test_file_path)
        assert samples == read_samples

    def test_analyze_triplet(self):
        samples = lj.lj_triplet_sampling(TestSampleUtils.test_class_dir, 8, False)
        triplet_anal = lj.lj_analyze(samples)
        exp_triplet_anal = (4, 8, 2.0, 2, 2)
        assert triplet_anal == exp_triplet_anal

