import os
import time
import pickle
import copy
import argparse
from omegaconf import OmegaConf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from data.RoleValueData import load_processed_data, load_dictionaries
from data.RoleValueLoader import RoleValueDataset
from models.NaLP import NaLP
from experiments.Metrics import compute_metric_scores


def run_epoch(model, data_iter, optimizer, epoch, device):
    """
    helper function to run one epoch of training

    :param model:
    :param data_iter:
    :param optimizer:
    :param epoch: epoch number
    :param device:
    :return:
    """

    model.train()
    losses = []

    for step, batch in enumerate(data_iter):
        optimizer.zero_grad()
        inputs, labels = [_.to(device) for _ in batch]

        preds = model.forward(inputs)
        # print(pred)
        loss = model.criterion(preds, labels)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    loss = np.sum(losses)
    # print(losses)
    print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, loss))
    return loss


def evaluate(model, data_iter, epoch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """

    model.eval()
    losses = []

    predications = []
    with torch.no_grad():

        for step, batch in enumerate(data_iter):

            inputs, labels = [_.to(device) for _ in batch]
            preds = model.forward(inputs)
            loss = model.criterion(preds, labels)

            losses.append(loss.item())

            predications.extend(preds.clone().cpu().data.numpy().flatten().tolist())

    loss = np.sum(losses)
    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, loss))

    return predications


def save_predictions(positive_data, negative_data, predictions, predictions_file):
    with open(predictions_file, "w") as fh:
        for i, instance in enumerate(positive_data):
            fh.write("{}\t{}\t{}".format(1, predictions[i], instance) + "\n")
        for i, instance in enumerate(negative_data):
            fh.write("{}\t{}\t{}".format(0, predictions[i + len(positive_data)], instance) + "\n")


def load_predictions(predictions_file):
    positive_data = []
    negative_data = []
    instance_prediction = {}
    with open(predictions_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                gt, score, instance = line.split("\t")
                gt = int(gt)
                if score == "None" or score is None:
                    score = None
                else:
                    score = float(score)
                instance = eval(instance)
                if gt == 1:
                    positive_data.append(instance)
                elif gt == 0:
                    negative_data.append(instance)
                else:
                    raise Exception("groundtruth label is not 1 or 0")
                instance_prediction[tuple(instance)] = score
    return positive_data, negative_data, instance_prediction


def convert_predictions_to_instance_predictions(positive_data, negative_data, predictions):
    instance_prediction = {}
    for i, instance in enumerate(positive_data):
        instance_prediction[tuple(instance)] = predictions[i]
    for i, instance in enumerate(negative_data):
        instance_prediction[tuple(instance)] = predictions[i + len(positive_data)]
    return instance_prediction


def save_nalp_model(model_dir, cfg, epoch, model, optimizer):

    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(state_dict, os.path.join(model_dir, "model.tar"))

    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_nalp_model(model_dir):
    """
    Load transformer model

    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """

    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    role2idx, value2idx, role_to_values = load_dictionaries(cfg.data_dir)

    # initialize model
    model = NaLP(role2idx, value2idx,
                 embedding_size=cfg.embedding_size,
                 num_filters=cfg.num_filters,
                 fully_connected_dimension=cfg.fully_connected_dimensions)
    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return cfg, model, optimizer, epoch


def run_nalp(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    # load data
    train_data, val_data, test_data, \
    train_data_negative, val_data_negative, test_data_negative, \
    val_instance_to_enumerated_instances, test_instance_to_enumerated_instances, \
    role2idx, value2idx, role_to_values = load_processed_data(cfg.data_dir)

    train_dataset = RoleValueDataset(train_data, train_data_negative, cfg.max_arity, role2idx, value2idx,
                                     label_smooth=cfg.label_smooth)
    val_dataset = RoleValueDataset(val_data, val_data_negative, cfg.max_arity, role2idx, value2idx,
                                   label_smooth=False)
    test_dataset = RoleValueDataset(test_data, test_data_negative, cfg.max_arity, role2idx, value2idx,
                                    label_smooth=False)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                                    collate_fn=RoleValueDataset.collate_fn)
    data_iter["val"] = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=RoleValueDataset.collate_fn)
    data_iter["test"] = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                                   collate_fn=RoleValueDataset.collate_fn)

    # initialize model
    model = NaLP(role2idx, value2idx,
                 embedding_size=cfg.embedding_size,
                 num_filters=cfg.num_filters,
                 fully_connected_dimension=cfg.fully_connected_dimensions)
    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0)

    validation_dir = os.path.join(cfg.experiment_dir, "validation")
    test_dir = os.path.join(cfg.experiment_dir, "test")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # train
    predictions = None
    for epoch in range(cfg.max_epochs + 1):
        run_epoch(model, data_iter["train"], optimizer, epoch, cfg.device)

        if epoch % 50 == 0:
            predictions = evaluate(model, data_iter["val"], epoch, cfg.device)
            predictions_file = os.path.join(validation_dir, "predictions.txt")
            save_predictions(val_data, val_data_negative, predictions, predictions_file)
            instance_prediction = convert_predictions_to_instance_predictions(val_data, val_data_negative, predictions)
            val_results = compute_metric_scores(val_data, val_data_negative, instance_prediction,
                                                cfg.validation_query_roles, role_to_values,
                                                ignore_non_object=True, save_dir=validation_dir, verbose=True)

    # test
    print("Evaluating test scores...")
    predictions = evaluate(model, data_iter["test"], cfg.max_epochs, cfg.device)
    predictions_file = os.path.join(test_dir, "predictions.txt")
    save_predictions(test_data, test_data_negative, predictions, predictions_file)
    instance_prediction = convert_predictions_to_instance_predictions(test_data, test_data_negative, predictions)
    compute_metric_scores(test_data, test_data_negative, instance_prediction, cfg.test_query_roles, role_to_values,
                          ignore_non_object=True, save_dir=test_dir, verbose=True)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "checkpoint")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_nalp_model(model_dir, cfg, cfg.max_epochs, model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NaLP model")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/nalp/run_nalp_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    cfg.model = "NaLP"
    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_nalp(cfg)