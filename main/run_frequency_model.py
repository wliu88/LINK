import os
import argparse
import itertools
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import pickle

from omegaconf import OmegaConf

from data.RoleValueData import load_processed_data, load_dictionaries
from experiments.Metrics import compute_metric_scores
from run_NaLP import save_predictions, load_predictions, convert_predictions_to_instance_predictions


def evaluate_predictions(frequency_table, experiment_dir, test_data, test_data_negative, role_to_values, query_roles, main_role):
    """
    This function computes metric scores of the model based on triple predictions

    :param experiment_dir:
    :param test_data:
    :param test_data_negative:
    :param role_to_values:
    :param query_roles:
    :return:
    """

    test_dir = os.path.join(experiment_dir, "test")

    query_role_dir = os.path.join(test_dir, "query_role")
    if not os.path.exists(query_role_dir):
        os.makedirs(query_role_dir)

    for query_role in query_roles:

        # evaluate model
        predictions = []
        for d in test_data + test_data_negative:
            main_role_value = None
            query_values = []
            for r, v in d:
                if r == main_role:
                    main_role_value = v
                if r == query_role:
                    query_values.append(v)

            query_scores = []
            if main_role_value is None or len(query_values) == 0 or main_role_value not in frequency_table[query_role]:
                query_scores.append(0)
            else:
                for query_value in query_values:
                    if query_value in frequency_table[query_role][main_role_value]:
                        query_scores.append(frequency_table[query_role][main_role_value][query_value])
                    else:
                        # if the value is not in the frequency table, assign 0
                        query_scores.append(0)

            predictions.append(float(np.mean(query_scores)))

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
    # results = summary_df.to_dict()


def save_frequency_model(model_dir, cfg, model):

    with open(os.path.join(model_dir, "model.pkl"), "wb") as fh:
        pickle.dump(model, fh)

    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_frequency_model(model_dir):

    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as fh:
        model = pickle.load(fh)

    return cfg, model


def run_frequency_model(cfg):
    # load data
    train_data, val_data, test_data, \
    train_data_negative, val_data_negative, test_data_negative, \
    val_instance_to_enumerated_instances, test_instance_to_enumerated_instances, \
    role2idx, value2idx, role_to_values = load_processed_data(cfg.data_dir)

    main_role = cfg.main_role

    # build the frequency table, query_role -> main_role_value (e.g., object class)-> query_role_value -> count
    # initialize the table
    frequency_table = {}
    for r in role_to_values:
        if r != main_role:
            frequency_table[r] = {}
            for m in role_to_values[main_role]:
                frequency_table[r][m] = {}
                for v in role_to_values[r]:
                    frequency_table[r][m][v] = 0

    for d in train_data:
        main_role_value = None
        for r, v in d:
            if r == main_role:
                main_role_value = v
                break

        if main_role_value is None:
            continue

        for r, v in d:
            if r != main_role:
                frequency_table[r][main_role_value][v] += 1

    # normalize counts for each object class
    for r in frequency_table:
        for m in frequency_table[r]:
            count_sum = 0
            count_num = 0
            for v in frequency_table[r][m]:
                count = frequency_table[r][m][v]
                count_sum += count
                count_num += 1

            for v in frequency_table[r][m]:
                if count_sum:
                    frequency_table[r][m][v] = frequency_table[r][m][v] * 1.0 / count_sum
                else:
                    assert frequency_table[r][m][v] == 0
                    frequency_table[r][m][v] = 1.0 / count_num

            # assert sum([frequency_table[r][o][v] for v in frequency_table[r][o]]) - 1 < 0.01, frequency_table[r][o]

    # evaluate
    if cfg.evaluate_on_validation:
        evaluate_predictions(frequency_table, cfg.experiment_dir, val_data, val_data_negative, role_to_values, cfg.test_query_roles, cfg.main_role)
    else:
        evaluate_predictions(frequency_table, cfg.experiment_dir, test_data, test_data_negative, role_to_values, cfg.test_query_roles, cfg.main_role)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "checkpoint")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_frequency_model(model_dir, cfg, frequency_table)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run frequency model")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/frequency/run_frequency_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not cfg.run_all:
        # only run one experiment
        if not os.path.exists(cfg.experiment_dir):
            os.makedirs(cfg.experiment_dir)

        cfg.model = "Frequency"
        OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

        run_frequency_model(cfg)
    else:
        # run multiple experiments with using different main roles
        for test_role in copy.deepcopy(cfg.test_query_roles):
            os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
            cfg = OmegaConf.load(args.config_file)

            if not os.path.exists(cfg.experiment_dir):
                os.makedirs(cfg.experiment_dir)

            cfg.model = "Frequency"
            cfg.main_role = test_role
            cfg.test_query_roles.remove(test_role)

            OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

            run_frequency_model(cfg)