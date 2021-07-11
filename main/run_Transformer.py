import os
import time
import pickle
import copy
import argparse
from omegaconf import OmegaConf
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader

from data.RoleValueData import load_processed_data, load_dictionaries
from data.HyperRelationalLoader import HyperRelationalDataset
from models.Transformer import Transformer
from experiments.Metrics import compute_metric_scores
from run_NaLP import save_predictions, load_predictions, convert_predictions_to_instance_predictions


def run_epoch(model, data_iter, optimizer, scheduler, grad_clipping, epoch, device):
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
        inputs, labels, mask_positions = [_.to(device) for _ in batch]

        preds = model.forward(inputs, mask_positions)
        loss = model.criterion(preds, labels)

        loss.backward()

        if grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

        optimizer.step()
        losses.append(loss.item())

    if scheduler is not None:
        scheduler.step()

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

    # predictions: [number of instances, number of entities]
    predications = []
    with torch.no_grad():

        for step, batch in enumerate(data_iter):

            inputs, labels, mask_positions = [_.to(device) for _ in batch]

            preds = model.forward(inputs, mask_positions)
            loss = model.criterion(preds, labels)

            losses.append(loss.item())

            predications.extend(preds.clone().cpu().data.numpy().tolist())

    loss = np.sum(losses)
    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, loss))

    return predications


def convert_1N_predictions_to_instance_predictions(n_predictions, masked_instances_dataset, instances, query_role):
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
    for i in range(len(masked_instances_dataset)):
        masked_instance_tuple, truth_values, mask_position = masked_instances_dataset.get_raw_item(i)

        if masked_instance_tuple[mask_position][0] != query_role:
            continue

        n_pred = n_predictions[i]
        assert len(n_pred) == len(candidate_values)

        for vi, v in enumerate(candidate_values):

            v_score = n_pred[vi]
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


def evaluate_predictions(experiment_dir, test_data, test_data_negative, role_to_values,
                         n_predictions, masked_instances_dataset, query_roles, split):
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
        predictions = convert_1N_predictions_to_instance_predictions(n_predictions, masked_instances_dataset,
                                                                     test_data + test_data_negative, query_role)

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


def save_transformer_model(model_dir, cfg, epoch, model, optimizer, scheduler):

    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state_dict, os.path.join(model_dir, "model.tar"))

    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_transformer_model(model_dir):
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
    model = Transformer(role2idx, value2idx, cfg.embedding_dim, cfg.max_arity, cfg.num_encoder_layer,
                        cfg.num_attention_heads, cfg.encoder_hidden_dim, cfg.encoder_dropout, cfg.encoder_activation,
                        cfg.use_output_layer_norm, cfg.use_position_embedding, cfg.pooling_method,
                        cfg.use_mask_pos_output)
    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size,
                                                    gamma=cfg.scheduler_gamma)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if cfg.use_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

    return cfg, role2idx, value2idx, role_to_values, model, optimizer, scheduler, epoch


def run_transformer(cfg):

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

    train_dataset = HyperRelationalDataset(train_data, cfg.test_query_roles, cfg.max_arity, role2idx, value2idx,
                                           label_smooth=cfg.label_smooth, augment_class_level=cfg.augment_class_level)
    val_dataset = HyperRelationalDataset(val_data, cfg.test_query_roles, cfg.max_arity, role2idx, value2idx,
                                         label_smooth=0.0, augment_class_level=False)
    test_dataset = HyperRelationalDataset(test_data, cfg.test_query_roles, cfg.max_arity, role2idx, value2idx,
                                          label_smooth=0.0, augment_class_level=False)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                                    collate_fn=HyperRelationalDataset.collate_fn)
    data_iter["val"] = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=HyperRelationalDataset.collate_fn)
    data_iter["test"] = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                                   collate_fn=HyperRelationalDataset.collate_fn)

    # initialize model
    model = Transformer(role2idx, value2idx, cfg.embedding_dim, cfg.max_arity, cfg.num_encoder_layer,
                        cfg.num_attention_heads, cfg.encoder_hidden_dim, cfg.encoder_dropout, cfg.encoder_activation,
                        cfg.use_output_layer_norm, cfg.use_position_embedding, cfg.pooling_method,
                        cfg.use_mask_pos_output)

    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size,
                                                    gamma=cfg.scheduler_gamma)

    validation_dir = os.path.join(cfg.experiment_dir, "val")
    test_dir = os.path.join(cfg.experiment_dir, "test")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # train
    predictions = None
    for epoch in range(cfg.max_epochs + 1):
        run_epoch(model, data_iter["train"], optimizer, scheduler, cfg.grad_clipping, epoch, cfg.device)

        if epoch % 50 == 0:
            n_predictions = evaluate(model, data_iter["val"], epoch, cfg.device)
            evaluate_predictions(cfg.experiment_dir, val_data, val_data_negative, role_to_values,
                                 n_predictions, val_dataset, cfg.validation_query_roles, "val")

    # test
    print("Evaluating test scores...")
    n_predictions = evaluate(model, data_iter["test"], cfg.max_epochs, cfg.device)
    evaluate_predictions(cfg.experiment_dir, test_data, test_data_negative, role_to_values,
                         n_predictions, test_dataset, cfg.test_query_roles, "test")

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "checkpoint")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_transformer_model(model_dir, cfg, cfg.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer model")
    parser.add_argument("--config_file", help='config yaml file', default='../configs/transformer/run_transformer_non_repeating_10_value_negative_expanded.yaml', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    cfg.model = "Transformer"
    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_transformer(cfg)
