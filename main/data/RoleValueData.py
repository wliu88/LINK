import os
import pickle
import copy

import numpy as np
from collections import defaultdict
import tqdm


# important: when perturbing values for generating negative samples, we are currently using values that have the correct
#            type. We can also use all possible values.


def convert_to_canonical_instance(instance):
    """
    Convert the original instance to canonical form by sorting the roles and values.

    e.g., [(A, a), (A, b), ..., (Z, z)]

    :param instance:
    :return:
    """
    instance = sorted(instance)
    return instance


def convert_to_role_value_format(object_data, object_instance_data, subsample_object_instance_ratio,
                                 exclude_iids_file=None):
    """
    This function converts object instance data and object data to the role-value pair format, where each instance is
    a list of role-value pairs. The data after conversion are called flattened_data.

    The order of role-value pairs in each instance are sorted.

    e.g., instance1 = [(role1, value1), (role2, value2), ..., (roleN, vlaueN))]

    # Important: weight, price, and size are the 3 roles that are handled separately

    :param object_data:
    :param object_instance_data:
    :param subsample_object_instance_ratio: if less than 1.0, for each object, only a subset of object instance will be
                                            maintained
    :param exclude_iids_file: a txt file storing a list of iids that need to be excluded
    :return:
    """
    assert 0 < subsample_object_instance_ratio <= 1.0, "subsample_object_instance_ratio not in range (0, 1.0]"

    excluding_iids = []
    if exclude_iids_file is not None:
        assert ".txt" in exclude_iids_file
        with open(exclude_iids_file, "r") as fh:
            excluding_iids = eval(fh.readline())
        print("Excluding iids: {}".format(excluding_iids))

    oid_to_flattened_data = defaultdict(list)
    # the following 3 lists are used to discretize continuous values
    weights = []
    sizes = []
    prices = []
    for oiid in object_instance_data:

        if oiid in excluding_iids:
            continue

        object_instance = set()

        # flatten object data
        oid = object_instance_data[oiid]["id"]
        for role in object_data[oid]:
            value = object_data[oid][role]
            # 3 special roles
            if role == "size":
                # compute volume from WxHxL
                value = np.product([float(v) for v in value])
                sizes.append(value)
            if role == "weight":
                value = float(value)
                weights.append(value)
            if role == "price":
                value = float(value)
                prices.append(value)

            if type(value) != list:
                value = [value]
            for v in value:
                rv_pair = tuple([role, v])
                object_instance.add(rv_pair)

        # flatten object instance data
        for role in object_instance_data[oiid]:
            value = object_instance_data[oiid][role]
            if type(value) != list:
                value = [value]
            for v in value:
                rv_pair = tuple([role, v])
                object_instance.add(rv_pair)

        # sort role-value pairs
        object_instance = list(object_instance)
        object_instance = convert_to_canonical_instance(object_instance)
        oid_to_flattened_data[oid].append(object_instance)

    # subsample
    flattened_data = []
    for oid in oid_to_flattened_data:
        if subsample_object_instance_ratio >= 1.0:
            flattened_data.extend(oid_to_flattened_data[oid])
        else:
            candidates = oid_to_flattened_data[oid]
            subsample_number = int(len(candidates) * subsample_object_instance_ratio)
            if subsample_number > 0:
                candidate_idxs = list(range(len(candidates)))
                subset_idxs = np.random.choice(candidate_idxs, subsample_number, replace=False)
                flattened_data.extend([candidates[si] for si in subset_idxs])

    # Important: discretize weight, size, and price based on percentile
    for i, d in enumerate(flattened_data):
        for pi, rv_pair in enumerate(d):
            role, value = rv_pair
            if role == "size":
                if value < np.percentile(sizes, 33):
                    value = "small"
                elif value < np.percentile(sizes, 66):
                    value = "medium"
                else:
                    value = "large"
            elif role == "weight":
                if value < np.percentile(weights, 33):
                    value = "light"
                elif value < np.percentile(weights, 66):
                    value = "medium"
                else:
                    value = "heavy"
            elif role == "price":
                if value < np.percentile(prices, 33):
                    value = "cheap"
                elif value < np.percentile(prices, 66):
                    value = "medium"
                else:
                    value = "expensive"
            d[pi] = (role, value)
        flattened_data[i] = convert_to_canonical_instance(d)

    return flattened_data


def remove_roles(flattened_data, keep_roles):
    """
    This function removes role-value pairs from each instance by keeping only keep_roles

    :param flattened_data:
    :param keep_roles: a list of roles that will be kept
    :return:
    """

    abridged_data = []
    for d in flattened_data:
        instance = []
        for rv_pair in d:
            if rv_pair[0] in keep_roles:
                instance.append(rv_pair)
        abridged_data.append(instance)

    return abridged_data


def check_duplicate_instances(flattened_data):
    """
    This function checks whether the same instance exist multiple times in the data

    :param flattened_data:
    :return:
    """

    # the number of times that an instance has appeared in data
    instance_to_num_app = {}
    for d in flattened_data:

        # Important: must sort here, since the order of role-value pairs doesn't matter for identifying unique instances
        d = tuple(convert_to_canonical_instance(d))
        if d not in instance_to_num_app:
            instance_to_num_app[d] = 0
        instance_to_num_app[d] += 1

    # the number instances for each duplicate number
    duplicate_num = {}
    for instance in instance_to_num_app:
        num = instance_to_num_app[instance]
        if num not in duplicate_num:
            duplicate_num[num] = 0
        duplicate_num[num] += 1

    print("\nChecking duplicate instances in the data...")
    for duplicate in duplicate_num:
        ratio = duplicate_num[duplicate] * 1.0 / len(flattened_data) * 100
        print("There are {}% ({}) of instances that have {} duplicate".format(ratio, duplicate_num[duplicate], duplicate-1))

    return instance_to_num_app


def split_data_simple(flattened_data, split_ratio):
    """
    This function splits flattened_data into train/validation/test based on the given ratio.

    Important: this split function allows for known instances in testing

    :param split_ratio: [train_ratio, val_ratio, test_ratio]
    :param flattened_data:
    :return:
    """

    train_ratio, val_ratio, test_ratio = split_ratio
    assert train_ratio + test_ratio + val_ratio == 1.0

    object_instance_ids = list(range(len(flattened_data)))

    num_train = int(len(object_instance_ids) * train_ratio)
    num_test = int(len(object_instance_ids) * test_ratio)
    num_val = int(len(object_instance_ids) - num_train - num_test)

    idxes = object_instance_ids
    np.random.shuffle(idxes)
    train_idxs = idxes[:num_train]
    test_idxs = idxes[num_train:num_train + num_test]
    val_idxs = idxes[num_train + num_test:]

    return train_idxs, val_idxs, test_idxs


def split_data_no_duplicates_for_base_properties(flattened_data, split_ratio, base_properties):
    """
    This function splits flattened_data into train/validation/test based ROUGHLY on the given ratio. Instead of directly
    splitting the data. A non-repeating set of data will be constructed based on the given base_properties. This set
    will be split into train/validation/test first, which are then used to split the flattened_data.

    Important: this split function ensures no test leakage.

    :param flattened_data:
    :param split_ratio:
    :param base_properties:
    :return:
    """

    train_ratio, val_ratio, test_ratio = split_ratio
    assert train_ratio + test_ratio + val_ratio == 1.0

    # build a dictionary based on base_properties
    base_data_to_flattened_data_idxs = defaultdict(list)
    for idx, d in enumerate(flattened_data):
        d_base = []
        for r, v in d:
            if r in base_properties:
                d_base.append((r, v))
        d_base = tuple(sorted(d_base))
        base_data_to_flattened_data_idxs[d_base].append(idx)

    print("\nSplitting data...")
    print("Considering only the given base properties: {}".format(base_properties))
    print("Number of unique instances is {}".format(len(base_data_to_flattened_data_idxs)))

    # split base data
    base_objects = list(base_data_to_flattened_data_idxs.keys())
    base_object_ids = list(range(len(base_objects)))

    num_train = int(len(base_object_ids) * train_ratio)
    num_test = int(len(base_object_ids) * test_ratio)
    num_val = int(len(base_object_ids) - num_train - num_test)

    np.random.shuffle(base_object_ids)
    base_train_idxs = base_object_ids[:num_train]
    base_test_idxs = base_object_ids[num_train:num_train + num_test]
    base_val_idxs = base_object_ids[num_train + num_test:]

    train_idxs = []
    test_idxs = []
    val_idxs = []

    for idx in base_train_idxs:
        train_idxs.extend(base_data_to_flattened_data_idxs[base_objects[idx]])
    for idx in base_test_idxs:
        test_idxs.extend(base_data_to_flattened_data_idxs[base_objects[idx]])
    for idx in base_val_idxs:
        val_idxs.extend(base_data_to_flattened_data_idxs[base_objects[idx]])

    return train_idxs, val_idxs, test_idxs



def check_data_splits(flattened_data, train_idxs, val_idxs, test_idxs):
    """
    This function checks if an instance exists multiple times in testing set and if an instance in testing set also
    exists in training set.

    :param flattened_data:
    :param train_idxs:
    :param val_idxs:
    :param test_idxs:
    :return:
    """

    instance_duplicate_in_train = {}
    instance_duplicate_in_test = {}

    for test_idx in test_idxs:
        test_instance = flattened_data[test_idx]
        test_instance = tuple(convert_to_canonical_instance(test_instance))
        if test_instance not in instance_duplicate_in_test:
            instance_duplicate_in_test[test_instance] = 0
        if test_instance not in instance_duplicate_in_train:
            instance_duplicate_in_train[test_instance] = 0
        instance_duplicate_in_test[test_instance] += 1

    for train_idx in train_idxs:
        train_instance = flattened_data[train_idx]
        train_instance = tuple(convert_to_canonical_instance(train_instance))
        if train_instance in instance_duplicate_in_train:
            instance_duplicate_in_train[train_instance] += 1

    # for test_instance in instance_duplicate_in_test:
    #     print("{}, {} duplicates in train and test for instance {}".format(instance_duplicate_in_train[test_instance],
    #                                                                        instance_duplicate_in_test[test_instance],
    #                                                                        test_instance))

    train_duplicate_num = {}
    test_duplicate_num = {}
    for instance in instance_duplicate_in_test:
        num = instance_duplicate_in_test[instance]
        if num not in test_duplicate_num:
            test_duplicate_num[num] = 0
        test_duplicate_num[num] += 1

        num = instance_duplicate_in_train[instance]
        if num not in train_duplicate_num:
            train_duplicate_num[num] = 0
        train_duplicate_num[num] += 1

    print("\nChecking if testing data is in train...")
    print("There are {} testing instances and {} unique testing instances".format(len(test_idxs), len(instance_duplicate_in_test)))

    for duplicate in train_duplicate_num:
        ratio = train_duplicate_num[duplicate] * 1.0 / len(instance_duplicate_in_test) * 100
        print("There are {}% ({}) of unique testing instances that have {} duplicates in train".format(ratio, train_duplicate_num[duplicate], duplicate))

    for duplicate in test_duplicate_num:
        ratio = test_duplicate_num[duplicate] * 1.0 / len(instance_duplicate_in_test) * 100
        print("There are {}% ({}) of unique testing instances that have {} duplicates in test".format(ratio, test_duplicate_num[duplicate], duplicate - 1))


def build_dicts(flattened_data, add_reverse):
    """

    :param flattened_data:
    :param add_reverse:
    :return:
        - role2idx - a dictionary from each role to its unique id
        - value2idx - a dictionary from each value to its unique id
        - role_to_values - a dictionary from each role to matching values in a list
    """

    value_set = set()
    role_to_values = defaultdict(set)

    for instance in flattened_data:
        for role, value in instance:
            value_set.add(value)
            role_to_values[role].add(value)

    value_list = sorted(list(value_set))
    roles_list = sorted(list(role_to_values.keys()))
    role_to_values = {role: list(role_to_values[role]) for role in role_to_values}

    print("\nBuilding dictionaries...")
    print("{} roles: {}".format(len(roles_list), roles_list))
    print("{} values: {}".format(len(value_list), value_list))
    for role in role_to_values:
        print("{} values for {}: {}".format(len(role_to_values[role]), role, role_to_values[role]))

    if add_reverse:
        print("Reverse roles will be added!")

    # build dicts
    role2idx = {role: idx for idx, role in enumerate(["#PAD_TOKEN"] + roles_list)}
    value2idx = {value: idx for idx, value in enumerate(["#PAD_TOKEN"] + value_list)}

    if add_reverse:
        # reverse relation id is: idx+len(rel2id)
        role2idx.update({role + '_reverse': idx + len(role2idx) for idx, role in enumerate(roles_list)})

    # idx2role = {idx: role for role, idx in role2idx.items()}
    # idx2value = {idx: value for value, idx in value2idx.items()}

    return role2idx, value2idx, role_to_values


def sample_negative_examples(seed_data, check_data, role_to_values, replace_value_ratio, check_subset, negative_ratio):
    """
    This function creates negative examples from seed_data by randomly sampling roles and values for each instance in
    seed_data. The created negative examples do not exist in check_data but may have duplicates.

    :param seed_data:
    :param check_data:
    :param role_to_values:
    :param replace_value_ratio: a number between 0 and 1, indicating the probability of replacing value vs role
    :param check_subset: if set to true, any sampled negative instance is also not a subset of any instance in
                         check_data
    :param negative_ratio: how many negative instances to sample for each instance in seed_data.
    :return:
    """

    print("\nSampling negative examples...")
    print("A large negative ratio (currently 1:{}) may result in sampling never ending".format(negative_ratio))

    # convert each instance to tuple
    check_data_tuple = [tuple(convert_to_canonical_instance(d)) for d in check_data]

    # Important: we don't ensure that sampled negative instances have no duplicates
    # ToDo: we can add an argument to indicate if we want to sample non-repeating negative examples
    sampled_data = []
    for instance in tqdm.tqdm(seed_data, desc="sample for each seed instance"):

        # sample negative_ratio negative examples for each positive example
        for _ in range(negative_ratio):

            if np.random.uniform(0, 1) <= replace_value_ratio:
                # perturb value
                perturbed_instance = randomly_perturb_instance_value(instance, check_data_tuple,
                                                                     role_to_values, check_subset)
            else:
                # perturb role
                perturbed_instance = randomly_perturb_instance_role(instance, check_data_tuple,
                                                                    role_to_values, check_subset)

            sampled_data.append(perturbed_instance)

    # check if sampled data is correct
    for d in tqdm.tqdm(sampled_data, desc="verify each sampled instance"):
        d = tuple(convert_to_canonical_instance(d))
        for cd in check_data:
            cd = tuple(convert_to_canonical_instance(cd))
            assert d != cd
            if check_subset:
                assert not check_if_a_subset(d, cd)

    return sampled_data


def enumerate_negative_examples(seed_data, check_data, role_to_values, enumerate_roles, enumerate_values, check_subset):
    """
    This function creates negative examples from seed_data by enumerating roles and values from each instance in
    seed_data. The created negative examples have no duplicates and do not exist in check_data.

    :param seed_data: a list of instances
    :param check_data: a list of instances
    :param role_to_values:
    :param enumerate_roles: whether to enumerate roles
    :param enumerate_values: whether to enumerate values
    :param check_subset: if set to true, any sampled negative instance is also not a subset of any instance in
                         check_data
    :return:
        - enumerated_data - negative examples in a list
        - instance_to_enumerated_instances - a dictionary mapping from each positive instance to negative instances
                                             sampled from it
    """

    print("\nEnumerating negative examples...")
    assert enumerate_roles or enumerate_values, "neither roles or values are set to be enumerated"

    # convert each instance to tuple
    check_data_tuple = [tuple(convert_to_canonical_instance(d)) for d in check_data]

    enumerated_data = set()
    instance_to_enumerated_instances = defaultdict(set)
    for instance in tqdm.tqdm(seed_data, desc="enumerate for each seed instance"):

        enumerated_instances = []
        if enumerate_roles:
            enumerated_instances.extend(enumerate_instance_role(instance, check_data_tuple, role_to_values, check_subset))
        if enumerate_values:
            enumerated_instances.extend(enumerate_instance_value(instance, check_data_tuple, role_to_values, check_subset))

        # convert each instance to tuple to make sure no duplicate instances are enumerated
        enumerate_instances_tuple = [tuple(convert_to_canonical_instance(d)) for d in enumerated_instances]
        enumerated_data.update(enumerate_instances_tuple)

        instance_tuple = tuple(convert_to_canonical_instance(instance))
        instance_to_enumerated_instances[instance_tuple].update(enumerate_instances_tuple)

    # convert enumerated data back to a list of instances, where each instance is a list of role-value pairs
    enumerated_data = [list(d) for d in enumerated_data]

    # check if sampled data is correct
    # check if each instance is a list of tuples
    assert type(enumerated_data[0]) == list
    assert type(enumerated_data[0][0]) == tuple
    for d in tqdm.tqdm(enumerated_data, desc="verify each enumerated instance"):
        d = tuple(convert_to_canonical_instance(d))
        for cd in check_data:
            cd = tuple(convert_to_canonical_instance(cd))
            assert d != cd
            if check_subset:
                assert not check_if_a_subset(d, cd)

    return enumerated_data, instance_to_enumerated_instances


def enumerate_instance_role(instance, check_data, role_to_values, check_subset):
    """
    This function enumerate all negative example by exhaustively perturbing each role in a positive example.

    :param instance: [(role1, value1), (role2, value2), ..., (roleN, vlaueN)]
    :param check_data: a list of instances, where each instance is a tuple, in the canonical form, and positive
    :param role_to_values:
    :return:
    """

    # roughly check if each instance in checked data is a tuple
    assert type(check_data[0]) == tuple

    enumerated_instances = []
    for pair_i in range(len(instance)):
        value = instance[pair_i][1]
        candidate_roles = sorted(list(role_to_values.keys()))

        for perturbed_role in candidate_roles:
            perturbed_instance = copy.deepcopy(instance)
            perturbed_instance[pair_i] = (perturbed_role, value)

            # check the perturbed instance against check_data
            if tuple(convert_to_canonical_instance(perturbed_instance)) not in check_data:
                if check_subset:
                    if not check_if_a_subset_in_list(perturbed_instance, check_data):
                        enumerated_instances.append(perturbed_instance)
                else:
                    enumerated_instances.append(perturbed_instance)

    return enumerated_instances


def enumerate_instance_value(instance, check_data, role_to_values, check_subset):
    """
    This function enumerate all negative example by exhaustively perturbing each value in a positive example.

    :param instance: [(role1, value1), (role2, value2), ..., (roleN, vlaueN)]
    :param check_data: a list of instances, where each instance is a tuple, in the canonical form, and positive
    :param role_to_values:
    :return:
    """

    # roughly check if each instance in checked data is a tuple
    assert type(check_data[0]) == tuple

    enumerated_instances = []
    for pair_i in range(len(instance)):
        role = instance[pair_i][0]
        candidate_values = sorted(role_to_values[role])

        for perturbed_value in candidate_values:
            perturbed_instance = copy.deepcopy(instance)
            perturbed_instance[pair_i] = (role, perturbed_value)

            # check the perturbed instance against check_data
            if tuple(convert_to_canonical_instance(perturbed_instance)) not in check_data:
                if check_subset:
                    if not check_if_a_subset_in_list(perturbed_instance, check_data):
                        enumerated_instances.append(perturbed_instance)
                else:
                    enumerated_instances.append(perturbed_instance)

    return enumerated_instances


def randomly_perturb_instance_role(instance, check_data, role_to_values, check_subset):
    """
    This function samples a negative example by perturbing a role in a positive example.

    :param instance: [(role1, value1), (role2, value2), ..., (roleN, vlaueN)]
    :param check_data: a list of instances, where each instance is a tuple, in the canonical form, and positive
    :param role_to_values:
    :return:
    """

    # roughly check if each instance in checked data is a tuple
    assert type(check_data[0]) == tuple

    perturbed_instance = None
    successfully_perturbed = False
    count = 0
    while not successfully_perturbed:
        pair_i = np.random.randint(0, len(instance))
        value = instance[pair_i][1]
        candidate_roles = sorted(list(role_to_values.keys()))
        perturbed_role = candidate_roles[np.random.randint(0, len(candidate_roles))]

        perturbed_instance = copy.deepcopy(instance)
        perturbed_instance[pair_i] = (perturbed_role, value)

        # check the perturbed instance against check_data
        if tuple(convert_to_canonical_instance(perturbed_instance)) not in check_data:
            if check_subset:
                if not check_if_a_subset_in_list(perturbed_instance, check_data):
                    successfully_perturbed = True
            else:
                successfully_perturbed = True

        count += 1
        assert count < 1000, "perturbing role failed after 1000 tries"

    return perturbed_instance


def randomly_perturb_instance_value(instance, check_data, role_to_values, check_subset):
    """
    This function samples a negative example by perturbing a value in a positive example.

    :param instance: [(role1, value1), (role2, value2), ..., (roleN, vlaueN)]
    :param check_data: a list of instances, where each instance is a tuple, in the canonical form, and positive
    :param role_to_values:
    :return: perturbed_instance: as a list of tuples
    """

    # roughly check if each instance in checked data is a tuple
    assert type(check_data[0]) == tuple

    perturbed_instance = None
    successfully_perturbed = False
    count = 0
    while not successfully_perturbed:
        pair_i = np.random.randint(0, len(instance))
        role = instance[pair_i][0]
        candidate_values = sorted(role_to_values[role])
        perturbed_value = candidate_values[np.random.randint(0, len(candidate_values))]

        perturbed_instance = copy.deepcopy(instance)
        perturbed_instance[pair_i] = (role, perturbed_value)

        # check the perturbed instance against check_data
        # Important: we also need to check if the perturbed instance is a subset of any positive instance
        #            for example, if [(color, silver), (physical_property, hard), (material, metal)] is positive,
        #            then, [(color, silver), (physical_property, hard)] shouldn't be negative
        if tuple(convert_to_canonical_instance(perturbed_instance)) not in check_data:
            if check_subset:
                if not check_if_a_subset_in_list(perturbed_instance, check_data):
                    successfully_perturbed = True
            else:
                successfully_perturbed = True

        count += 1
        assert count < 1000, "perturbing value failed after 1000 tries"

    return perturbed_instance


def check_if_a_subset(instance_1, instance_2):
    """
    This function checks if instance_1 is a subset of instance_2

    for example, [(color, silver), (physical_property, hard)] is a subset of the instance
    [(color, silver), (physical_property, hard), (material, metal)]

    :param instance_1:
    :param instance_2:
    :return:
    """
    for rv_pair in instance_1:
        if rv_pair not in instance_2:
            return False
    return True


def check_if_a_subset_in_list(instance, check_instances):
    """
    This function checks if instance is a subset of any instance in instances

    :param instance:
    :param check_instances:
    :return:
    """
    for check_instance in check_instances:
        if check_if_a_subset(instance, check_instance):
            return True
    return False


def save_processed_data(save_dir,
                        train_data, val_data, test_data,
                        train_data_negative, val_data_negative, test_data_negative,
                        val_instance_to_enumerated_instances, test_instance_to_enumerated_instances,
                        role2idx, value2idx, role_to_values):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_flattened_data(os.path.join(save_dir, "train_data.txt"), train_data)
    save_flattened_data(os.path.join(save_dir, "val_data.txt"), val_data)
    save_flattened_data(os.path.join(save_dir, "test_data.txt"), test_data)
    save_flattened_data(os.path.join(save_dir, "train_data_negative.txt"), train_data_negative)
    save_flattened_data(os.path.join(save_dir, "val_data_negative.txt"), val_data_negative)
    save_flattened_data(os.path.join(save_dir, "test_data_negative.txt"), test_data_negative)

    with open(os.path.join(save_dir, "role2idx.txt"), "w") as fh:
        fh.write(str(role2idx))

    with open(os.path.join(save_dir, "value2idx.txt"), "w") as fh:
        fh.write(str(value2idx))

    with open(os.path.join(save_dir, "role_to_values.txt"), "w") as fh:
        fh.write(str(role_to_values))

    with open(os.path.join(save_dir, "val_instance_to_enumerated_instances.pkl"), "wb") as fh:
        pickle.dump(val_instance_to_enumerated_instances, fh)

    with open(os.path.join(save_dir, "test_instance_to_enumerated_instances.pkl"), "wb") as fh:
        pickle.dump(test_instance_to_enumerated_instances, fh)


def load_processed_data(save_dir):
    train_data = load_flattened_data(os.path.join(save_dir, "train_data.txt"))
    val_data = load_flattened_data(os.path.join(save_dir, "val_data.txt"))
    test_data = load_flattened_data(os.path.join(save_dir, "test_data.txt"))
    train_data_negative = load_flattened_data(os.path.join(save_dir, "train_data_negative.txt"))
    val_data_negative = load_flattened_data(os.path.join(save_dir, "val_data_negative.txt"))
    test_data_negative = load_flattened_data(os.path.join(save_dir, "test_data_negative.txt"))

    with open(os.path.join(save_dir, "role2idx.txt"), "r") as fh:
        role2idx = eval(fh.readline())

    with open(os.path.join(save_dir, "value2idx.txt"), "r") as fh:
        value2idx = eval(fh.readline())

    with open(os.path.join(save_dir, "role_to_values.txt"), "r") as fh:
        role_to_values = eval(fh.readline())

    with open(os.path.join(save_dir, "val_instance_to_enumerated_instances.pkl"), "rb") as fh:
        val_instance_to_enumerated_instances = pickle.load(fh)

    with open(os.path.join(save_dir, "test_instance_to_enumerated_instances.pkl"), "rb") as fh:
        test_instance_to_enumerated_instances = pickle.load(fh)

    return train_data, val_data, test_data, \
           train_data_negative, val_data_negative, test_data_negative, \
           val_instance_to_enumerated_instances, test_instance_to_enumerated_instances, \
           role2idx, value2idx, role_to_values


def save_flattened_data(filename, flattend_data):
    with open(filename, "w") as fh:
        for instance in flattend_data:
            fh.write(str(instance) + "\n")


def load_flattened_data(filename):
    flattened_data = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                flattened_data.append(eval(line))
    return flattened_data


def load_dictionaries(save_dir):

    with open(os.path.join(save_dir, "role2idx.txt"), "r") as fh:
        role2idx = eval(fh.readline())

    with open(os.path.join(save_dir, "value2idx.txt"), "r") as fh:
        value2idx = eval(fh.readline())

    with open(os.path.join(save_dir, "role_to_values.txt"), "r") as fh:
        role_to_values = eval(fh.readline())

    return role2idx, value2idx, role_to_values