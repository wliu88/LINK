import os
import time
import pickle
import copy
import argparse
from omegaconf import OmegaConf

import numpy as np

from data.RoleValueData import load_processed_data
from experiments.Metrics import compute_metric_scores
from run_NaLP import save_predictions, load_predictions, convert_predictions_to_instance_predictions


def run_random_model(cfg):

    np.random.seed(cfg.random_seed)

    # load data
    train_data, val_data, test_data, \
    train_data_negative, val_data_negative, test_data_negative, \
    val_instance_to_enumerated_instances, test_instance_to_enumerated_instances, \
    role2idx, value2idx, role_to_values = load_processed_data(cfg.data_dir)

    validation_dir = os.path.join(cfg.experiment_dir, "validation")
    test_dir = os.path.join(cfg.experiment_dir, "test")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # randomly generated
    print("Evaluating valid scores...")
    predictions = np.random.random(len(val_data + val_data_negative))
    predictions_file = os.path.join(validation_dir, "predictions.txt")
    save_predictions(val_data, val_data_negative, predictions, predictions_file)
    instance_prediction = convert_predictions_to_instance_predictions(val_data, val_data_negative, predictions)
    compute_metric_scores(val_data, val_data_negative, instance_prediction, cfg.validation_query_roles, role_to_values,
                          ignore_non_object=True, save_dir=validation_dir, verbose=True)

    # test
    print("Evaluating test scores...")
    predictions = np.random.random(len(test_data + test_data_negative))
    predictions_file = os.path.join(test_dir, "predictions.txt")
    save_predictions(test_data, test_data_negative, predictions, predictions_file)
    instance_prediction = convert_predictions_to_instance_predictions(test_data, test_data_negative, predictions)
    compute_metric_scores(test_data, test_data_negative, instance_prediction, cfg.test_query_roles, role_to_values,
                          ignore_non_object=True, save_dir=test_dir, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random model")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/random/run_random_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    cfg.model = "Random"
    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_random_model(cfg)
