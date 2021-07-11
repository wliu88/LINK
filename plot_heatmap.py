import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


# 13
entity_property_types = ["class", "material", "color", "transparency", "dimension", "physical_property", "shape",
                         "temperature", "spatial_distribution", "dampness", "purity", "room", "specific_place"]
# 6 + 13
all_property_types = ["name", "price", "weight", "size", "state_description", "image"] + entity_property_types


def build_dicts(object_data, object_instance_data, all_property_types, entity_property_types, add_reverse=True):
    """
    Builds entity and relation dicts

    :param object_data:
    :param object_instance_data:
    :param all_property_types: all property types
    :param entity_property_types: property types that have named entities as property values
    :param add_reverse:
    :return:
    """

    print("Build entity and relation dictionaries...")

    # build entity set and a dictionary from relation to corresponding entities
    ent_set = set()
    rel_to_ents = defaultdict(set)
    for oid in object_data:
        for property_type in object_data[oid]:
            if property_type in entity_property_types:
                property_value = object_data[oid][property_type]
                assert type(property_value) == list or type(property_value) == str
                if type(property_value) == list:
                    for v in property_value:
                        assert type(v) == str
                        ent_set.add(v)
                        rel_to_ents[property_type].add(v)
                elif type(property_value) == str:
                    ent_set.add(property_value)
                    rel_to_ents[property_type].add(property_value)

    for oiid in object_instance_data:
        for property_type in object_instance_data[oiid]:
            if property_type in entity_property_types:
                property_value = object_instance_data[oiid][property_type]
                assert type(property_value) == list or type(property_value) == str
                if type(property_value) == list:
                    for v in property_value:
                        assert type(v) == str
                        ent_set.add(v)
                        rel_to_ents[property_type].add(v)
                elif type(property_value) == str:
                    ent_set.add(property_value)
                    rel_to_ents[property_type].add(property_value)

    ent_set = sorted(list(ent_set))
    rel_to_ents = dict(rel_to_ents)
    print("{} entities: {}".format(len(ent_set), ent_set))
    for rel in rel_to_ents:
        print("{} Entities for {}: {}".format(len(rel_to_ents[rel]), rel, rel_to_ents[rel]))

    rel_set = set(all_property_types)
    rel_set = sorted(list(rel_set))
    print("{} relations: {}".format(len(rel_set), rel_set))

    # build dicts
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    if add_reverse:
        # reverse relation id is: idx+len(rel2id)
        rel2id.update({rel + '_reverse': idx + len(rel2id) for idx, rel in enumerate(rel_set)})

    id2ent = {idx: ent for ent, idx in ent2id.items()}
    id2rel = {idx: rel for rel, idx in rel2id.items()}

    return ent2id, rel2id, id2ent, id2rel, rel_to_ents


def get_heatmap(object_data, object_instance_data, property1, property2, rel_to_ents, figsize=(30, 15)):

    p1s = sorted(list(rel_to_ents[property1]))
    p2s = sorted(list(rel_to_ents[property2]))
    p1_to_p2 = np.zeros([len(p1s), len(p2s)], dtype=int)
    for oiid in object_instance_data:
        if property1 in object_instance_data[oiid]:
            p1 = object_instance_data[oiid][property1]
        elif property1 in object_data[object_instance_data[oiid]["id"]]:
            p1 = object_data[object_instance_data[oiid]["id"]][property1]
        else:
            continue
        if property2 in object_instance_data[oiid]:
            p2 = object_instance_data[oiid][property2]
        elif property2 in object_data[object_instance_data[oiid]["id"]]:
            p2 = object_data[object_instance_data[oiid]["id"]][property2]
        else:
            continue
        if type(p1) != list:
            p1 = [p1]
        if type(p2) != list:
            p2 = [p2]
        for pp1 in p1:
            for pp2 in p2:
                p1_to_p2[p1s.index(pp1), p2s.index(pp2)] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(p1_to_p2)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(p2s)))
    ax.set_yticks(np.arange(len(p1s)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(p2s)
    ax.set_yticklabels(p1s)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(p1s)):
        for j in range(len(p2s)):
            if p1_to_p2[i, j] > 0:
                text = ax.text(j, i, p1_to_p2[i, j],
                               ha="center", va="center", color="w")

    ax.set_title("Co-occurrence of {} and {}".format(property1, property2))
    fig.tight_layout()
    plt.show()


def main(args):

    object_data_filename = os.path.join(args.data_dir, "object_data.pkl")
    with open(object_data_filename, "rb") as fh:
        object_data, object_instance_data = pickle.load(fh)

    ent2id, rel2id, id2ent, id2rel, rel_to_ents = build_dicts(object_data, object_instance_data,
                                                              all_property_types=all_property_types,
                                                              entity_property_types=entity_property_types,
                                                              add_reverse=True)

    assert args.property_1 in entity_property_types and \
           args.property_2 in entity_property_types, "Property type is not an entity type"
    get_heatmap(object_data, object_instance_data, args.property_1, args.property_2,
                rel_to_ents, figsize=(10, 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data")
    parser.add_argument('--data_dir', type=str, default='', help='')
    parser.add_argument('--property_1', type=str, default='class', help='')
    parser.add_argument('--property_2', type=str, default='material', help='')
    args = parser.parse_args()

    if args.data_dir == '':
        args.data_dir = "data/LINK_dataset"

    main(args)