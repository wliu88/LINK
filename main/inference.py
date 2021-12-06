import os
import sys
import copy
import itertools
from collections import defaultdict
import argparse

import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from tqdm import tqdm

from data.RoleValueData import check_if_a_subset, convert_to_role_value_format, save_flattened_data, load_flattened_data, load_dictionaries
# from build_object_search_data import load_object_search_data
from experiments.Metrics import compute_rank

from run_Transformer import load_transformer_model


def evaluate_transformer(input, mask_position, candidate_entities, model, model_cfg, role2idx, value2idx, role_to_values, all_entities):

    vectorized_input = []
    for r, v in input:
        vectorized_input.append(role2idx[r])
        vectorized_input.append(value2idx[v])

    for _ in range(model_cfg.max_arity - len(input)):
        vectorized_input.append(role2idx["#PAD_TOKEN"])
        vectorized_input.append(value2idx["#PAD_TOKEN"])
    input = torch.LongTensor([vectorized_input]).to(model_cfg.device)
    mask_position = torch.LongTensor([[mask_position]]).to(model_cfg.device)

    # run model
    model.eval()
    with torch.no_grad():
        pred = model.forward(input, mask_position)
        pred = pred.clone().cpu().data.numpy().tolist()[0]

    entitiy_scores = [0] * len(candidate_entities)
    for score, value in zip(pred, all_entities):
        if value in candidate_entities:
            entitiy_scores[candidate_entities.index(value)] = score

    return entitiy_scores


def evaluate(input, mask_position, candidate_entities, model, model_cfg,
             role2idx, value2idx, role_to_values, all_entities):
    """
    Evaluate a query.

    :param input: has format [(r1, v1), (r2, v2), (r3, #MASK)]
    :param mask_position:
    :param candidate_entities: candidate entities for the masked value
    :param model:
    :param model_cfg:
    :param role2idx:
    :param value2idx:
    :param role_to_values:
    :param all_entities:
    :return:
    """

    # print("evaluate: {}".format(input))

    if model_cfg.model.lower() == "transformer":
        return evaluate_transformer(input, mask_position, candidate_entities, model, model_cfg, role2idx, value2idx,
                                    role_to_values, all_entities)
    # elif model_cfg.model.lower() == "frequency":
    #     return evaluate_frequency(input, mask_position, candidate_entities, model, model_cfg)
    # elif model_cfg.model.lower() == "nalp":
    #     return evaluate_nalp(input, mask_position, candidate_entities, model, model_cfg, role2idx, value2idx)
    # elif model_cfg.model.lower() == "tucker":
    #     return evaluate_tucker(input, mask_position, candidate_entities, model, model_cfg)
    else:
        raise Exception("Model {} not supported for inference yet.".format(model_cfg.model.lower()))


def predict_one_role(data_instance, query_role, leave_roles, role2idx, value2idx, role_to_values, all_entities,
                     model, model_cfg):
    """
    Given a data instance, this function helps evaluate the query role

    :param data_instance:
    :param query_role:
    :param leave_roles:
    :return:
        - scores: a 1D numpy array
        - candidate_entities: a 1D list
    """

    candidate_entities = role_to_values[query_role]

    # 1. prepare canonical input: [(r1, v1), (r2, v2), (r3, #MASK)]
    input = []
    mask_position = -1
    for pi, rv_pair in enumerate(data_instance):
        r, v = rv_pair
        if r in leave_roles:
            continue
        if r == query_role:
            input.append((r, "#MASK"))
            mask_position = pi
        else:
            input.append((r, v))

    # 2. evaluate
    scores = evaluate(input, mask_position, candidate_entities, model, model_cfg,
                      role2idx, value2idx, role_to_values, all_entities)

    # 3. check
    scores = np.array(scores)
    assert len(scores) == len(candidate_entities)

    return scores, candidate_entities


def process_keyboard_input(role_to_values, override_query_roles=None, skip_input_roles=None):
    """
    command-line interface for taking user input

    :param role_to_values: map from property types to possible values
    :param override_query_roles: property types to query, if None, will prompt user to type in
    :param skip_input_roles: property types not to query
    :return:
    """

    print("-"*30)

    d = []
    if override_query_roles:
        query_roles = override_query_roles
    else:
        query_roles = []
    while not query_roles:
        print("1. Input the query role ({})".format(role_to_values.keys()))
        input_str = input()
        if input_str:
            candidate_query_roles = [r.strip() for r in input_str.split(",")]
            for r in candidate_query_roles:
                if r in role_to_values.keys():
                    query_roles.append(r)

    print("2. Input the known properties next")
    for ri, role in enumerate(role_to_values):
        print("role type ({}/{}): {}".format(ri+1, len(role_to_values), role))
        print("has candidate values: {}".format(", ".join(role_to_values[role])))
        if role in query_roles:
            d.append((role, "#MASK"))
            print()
            continue
        if skip_input_roles:
            if role in skip_input_roles:
                print()
                continue
        print("enter known properties (enter multiple properties with comma, press enter to skip)")
        input_str = input()
        if input_str:
            values = input_str.split(",")
            values = [v.strip() for v in values]
            if values:
                for v in values:
                    d.append((role, v))

    print("formatted query to the model: {}".format(d))
    print("-"*30)

    return d, query_roles


def command_line_query(model_dir, answers_per_question):
    """
    This function enables command line query of all property types

    :return:
    """

    model_cfg, role2idx, value2idx, role_to_values, model, _, _, _ = load_transformer_model(model_dir)

    # prepare some dictionaries
    if "#MASK" not in value2idx:
        value2idx["#MASK"] = len(value2idx)
    idx2value = {value2idx[v]: v for v in value2idx}
    all_entities = [idx2value[i] for i in range(len(idx2value))]

    # run model
    while True:

        input_str = input("press n to exit")
        if input_str == "n":
            print("Exit...")
            break

        d, query_roles = process_keyboard_input(role_to_values)

        assert len(query_roles) == 1
        scores, _ = predict_one_role(d, query_roles[0], [],
                                     role2idx, value2idx, role_to_values, all_entities,
                                     model, model_cfg)

        candidate_values = role_to_values[query_roles[0]]
        sort_idxs = np.argsort(scores)[::-1][0: min(answers_per_question, len(candidate_values))]
        top_answers = np.array(candidate_values)[sort_idxs]
        scores = np.array(scores)[sort_idxs]
        print("Query result (property and associated score):")
        for a, s in zip(top_answers, scores):
            print(a, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer inference")
    parser.add_argument("-model_dir", help='the location of the model checkpoint', type=str)
    parser.add_argument("--answers_per_question", help='the number of answers for each query', default=10, type=int)
    args = parser.parse_args()
    assert os.path.exists(args.model_dir), "Cannot find model checkpoint at {}".format(args.model_dir)

    command_line_query(args.model_dir, args.answers_per_question)