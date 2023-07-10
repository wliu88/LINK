import os
import argparse
import itertools
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import pickle
from tqdm import tqdm

from omegaconf import OmegaConf

from data.RoleValueData import load_processed_data, load_dictionaries
from data.HyperRelationalLoader import HyperRelationalDataset
from experiments.Metrics import compute_metric_scores
from run_NaLP import save_predictions, load_predictions, convert_predictions_to_instance_predictions
from try_gpt import predict_property_with_gpt



def convert_1N_predictions_to_instance_predictions(masked_instances_dataset, instances, query_role, role_to_values):
    """
    Convert 1-N predictions for masked instances to predictions for each single complete instance

    :param n_predictions: a list of lists, has dimension [number of masked instance, number of values]
    :param masked_instances_dataset:
    :param instances:
    :return: predictions: has length [number of instances]
    """

    value2idx = masked_instances_dataset.value2idx
    idx2value = {value2idx[v]: v for v in value2idx}
    candidate_values = [idx2value[i] for i in range(len(idx2value))]
    if "#MASK" not in candidate_values:
        candidate_values.append("#MASK")

    perturbed_instance_to_score = defaultdict(list)
    for i in tqdm(range(len(masked_instances_dataset))):
        masked_instance_tuple, truth_values, mask_position = masked_instances_dataset.get_raw_item(i)

        if masked_instance_tuple[mask_position][0] != query_role:
            continue

        # Note: this may put this model in an unfair advantage
        role_constrained_candidate_values = role_to_values[query_role]

        # make prediction here
        # debug
        # query_value_to_scores = None
        query_value_to_scores = predict_property_with_gpt(query_role, masked_instance_tuple, role_constrained_candidate_values, verbose=False, num_retry=3)
        if query_value_to_scores is None:
            query_value_to_scores = {v: 0.0 for v in role_constrained_candidate_values}

        print("---")
        print(query_role)
        print(masked_instance_tuple)
        print(role_constrained_candidate_values)
        print(query_value_to_scores)

        for v in candidate_values:
            if v not in role_constrained_candidate_values:
                query_value_to_scores[v] = 0.0

        assert len(query_value_to_scores) == len(candidate_values)

        for vi, v in enumerate(candidate_values):

            v_score = query_value_to_scores[v]
            perturbed_instance = list(masked_instance_tuple)
            perturbed_instance[mask_position] = (query_role, v)
            perturbed_instance = sorted(perturbed_instance)

            perturbed_instance_to_score[tuple(perturbed_instance)].append(v_score)

    predictions = []
    for instance in instances:
        if tuple(instance) in perturbed_instance_to_score:
            # ToDo: maybe multiply?
            predictions.append(np.mean(perturbed_instance_to_score[tuple(instance)]))
        else:
            predictions.append(None)

    return predictions


def evaluate_predictions(experiment_dir, test_data, test_data_negative, role_to_values, masked_instances_dataset, query_roles, split):
    """
    This function computes metric scores of the model based on 1-N predictions

    :param experiment_dir:
    :param test_data:
    :param test_data_negative:
    :param role_to_values:
    :param query_roles:
    :param object_centric:
    :return:
    """
    assert split in ["test", "val"]

    test_dir = os.path.join(experiment_dir, split)

    query_role_dir = os.path.join(test_dir, "query_role")
    if not os.path.exists(query_role_dir):
        os.makedirs(query_role_dir)

    for query_role in query_roles:
        predictions = convert_1N_predictions_to_instance_predictions(masked_instances_dataset,
                                                                     test_data + test_data_negative, query_role, role_to_values)

        predictions_file = os.path.join(query_role_dir, "{}:predictions.txt".format(query_role))

        save_predictions(test_data, test_data_negative, predictions, predictions_file)
        instance_prediction = convert_predictions_to_instance_predictions(test_data, test_data_negative, predictions)
        compute_metric_scores(test_data, test_data_negative, instance_prediction, [query_role], role_to_values,
                              ignore_non_object=True, save_dir=query_role_dir, verbose=True)

    # summarize metric scores
    dfs = []
    for query_role in query_roles:
        results_file = os.path.join(query_role_dir, "{}:results.csv".format(query_role))
        dfs.append(pd.read_csv(results_file, index_col=0))
    summary_df = pd.concat(dfs, axis=1)
    summary_df['average'] = summary_df.mean(axis=1)
    results_file = os.path.join(test_dir, "results.csv")
    summary_df.to_csv(results_file)


def main(cfg):
    # load data
    train_data, val_data, test_data, \
    train_data_negative, val_data_negative, test_data_negative, \
    val_instance_to_enumerated_instances, test_instance_to_enumerated_instances, \
    role2idx, value2idx, role_to_values = load_processed_data(cfg.data_dir)

    test_dataset = HyperRelationalDataset(test_data, cfg.test_query_roles, cfg.max_arity, role2idx, value2idx,
                                          label_smooth=0.0, augment_class_level=False)

    evaluate_predictions(cfg.experiment_dir, test_data, test_data_negative, role_to_values,
                         test_dataset, cfg.test_query_roles, "test")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run frequency model")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/gpt/run_gpt_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not cfg.run_all:
        # only run one experiment
        if not os.path.exists(cfg.experiment_dir):
            os.makedirs(cfg.experiment_dir)

        cfg.model = "gpt"
        OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

        main(cfg)
    else:
        # run multiple experiments with using different main roles
        for test_role in copy.deepcopy(cfg.test_query_roles):
            os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
            cfg = OmegaConf.load(args.config_file)

            if not os.path.exists(cfg.experiment_dir):
                os.makedirs(cfg.experiment_dir)

            cfg.model = "gpt"
            cfg.main_role = test_role
            cfg.test_query_roles.remove(test_role)

            OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

            main(cfg)