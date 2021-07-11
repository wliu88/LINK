import os
import pickle
import numpy as np
import argparse
from omegaconf import OmegaConf

from data.RoleValueData import *


def build_role_value_data(cfg):
    np.random.seed(cfg.build_role_value_data.random_seed)

    with open(cfg.base.raw_data_file, "rb") as fh:
        object_data, object_instance_data = pickle.load(fh)

    flattened_data = convert_to_role_value_format(object_data, object_instance_data,
                                                  cfg.build_role_value_data.subsample_object_instance_ratio,
                                                  cfg.build_role_value_data.exclude_iids_file)
    print("\nNumber of instances after converting to role-value format: {}".format(len(flattened_data)))

    flattened_data = remove_roles(flattened_data, cfg.build_role_value_data.test_roles)
    check_duplicate_instances(flattened_data)

    if cfg.build_role_value_data.non_repeating_split:
        train_idxs, val_idxs, test_idxs = split_data_no_duplicates_for_base_properties(flattened_data,
                                                                                       cfg.build_role_value_data.split_ratio,
                                                                                       cfg.build_role_value_data.non_repeating_split_base_properties)
    else:
        train_idxs, val_idxs, test_idxs = split_data_simple(flattened_data, cfg.build_role_value_data.split_ratio)
    check_data_splits(flattened_data, train_idxs, val_idxs, test_idxs)

    train_data = list(np.array(flattened_data)[train_idxs])
    val_data = list(np.array(flattened_data)[val_idxs])
    test_data = list(np.array(flattened_data)[test_idxs])

    role2idx, value2idx, role_to_values = build_dicts(flattened_data,
                                                      add_reverse=cfg.build_role_value_data.dict_add_role_reverse)

    if cfg.build_role_value_data.train_subsample_k is not None and (0 < cfg.build_role_value_data.train_subsample_k < 1):
        train_data_subset = np.random.choice(train_data, int(len(train_data) * cfg.build_role_value_data.train_subsample_k),
                                             replace=False)
        train_data_negative = sample_negative_examples(train_data_subset, train_data_subset, role_to_values,
                                                       replace_value_ratio=cfg.build_role_value_data.perturb_train_replace_value_ratio,
                                                       check_subset=cfg.build_role_value_data.perturb_train_check_subset,
                                                       negative_ratio=cfg.build_role_value_data.perturb_train_negative_ratio)
    else:
        train_data_negative = sample_negative_examples(train_data, train_data, role_to_values,
                                                   replace_value_ratio=cfg.build_role_value_data.perturb_train_replace_value_ratio,
                                                   check_subset=cfg.build_role_value_data.perturb_train_check_subset,
                                                   negative_ratio=cfg.build_role_value_data.perturb_train_negative_ratio)

    # reset random seed
    np.random.seed(cfg.build_role_value_data.random_seed)

    val_data_negative, val_instance_to_enumerated_instances = enumerate_negative_examples(val_data, train_data + val_data + test_data, role_to_values,
                                                    enumerate_roles=cfg.build_role_value_data.perturb_validation_roles,
                                                    enumerate_values=cfg.build_role_value_data.perturb_validation_values,
                                                    check_subset=cfg.build_role_value_data.perturb_validation_check_subset)

    test_data_negative, test_instance_to_enumerated_instances = enumerate_negative_examples(test_data, train_data + val_data + test_data, role_to_values,
                                                     enumerate_roles=cfg.build_role_value_data.perturb_test_roles,
                                                     enumerate_values=cfg.build_role_value_data.perturb_test_values,
                                                     check_subset=cfg.build_role_value_data.perturb_test_check_subset)

    if cfg.build_role_value_data.train_subsample_k is not None and (0 < cfg.build_role_value_data.train_subsample_k < 1):
        train_data = train_data_subset

    save_processed_data(cfg.build_role_value_data.save_dir,
                        train_data, val_data, test_data, train_data_negative, val_data_negative, test_data_negative,
                        val_instance_to_enumerated_instances, test_instance_to_enumerated_instances,
                        role2idx, value2idx, role_to_values)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build Role-Value Data")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/data/data_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    cfg = OmegaConf.load(args.config_file)

    build_role_value_data(cfg)