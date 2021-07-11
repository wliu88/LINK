import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import copy


class HyperRelationalDataset(Dataset):
    """
    :ivar label_smooth
    :ivar debug
    :ivar role2idx
    :ivar value2idx
    :ivar inputs
    :ivar outputs
    """

    def __init__(self, positive_data, query_roles, max_arity, role2idx, value2idx, label_smooth, augment_class_level,
                 debug=False):
        super().__init__()

        self.debug = debug
        self.augment_class_level = augment_class_level

        # params
        self.query_roles = query_roles
        self.label_smooth = label_smooth
        assert type(self.label_smooth) == float
        self.max_arity = max_arity

        # dicts
        self.role2idx = role2idx
        self.value2idx = value2idx

        # ToDo: temporary solution, move this to build_role_value_data
        if "#MASK" not in self.value2idx:
            self.value2idx["#MASK"] = len(self.value2idx)

        # data
        self.masked_instances = []
        self.masked_instance_to_values = {}
        self.masked_instance_to_mask_position = {}

        # self.inputs = []
        # self.labels = []
        # self.mask_positions = []
        self.load_and_process_data_v2(positive_data)

    def load_and_process_data_v2(self, positive_data):

        print("Loading data...")

        # verify that data is list of instances, where each instance is a list of role-value tuples
        assert type(positive_data) == list
        assert type(positive_data[0]) == list
        assert type(positive_data[0][0]) == tuple

        max_arity = 0
        for d in positive_data:
            if len(d) > max_arity:
                max_arity = len(d)
        assert max_arity <= self.max_arity, "max arity in data is larger than the value specified"
        print("set max arity {}, max arity in data is {}".format(self.max_arity, max_arity))

        # build a dictionary from masked instance to values, which can be used for 1-N training
        # (and also just normal training)
        # A masked instance is an instance where the value of the query role has been replaced with the #MASK token
        # Important: we are using list here because we still want to maintain the frequency information of correct
        #            values for each unique masked instance
        masked_instance_to_values = defaultdict(list)
        masked_instance_to_mask_position = {}
        for query_role in self.query_roles:
            for d in positive_data:
                query_role_positions = []
                for pair_i, rv_pair in enumerate(d):
                    if rv_pair[0] == query_role:
                        query_role_positions.append(pair_i)

                for pair_i in query_role_positions:
                    masked_instance = copy.deepcopy(d)
                    masked_instance[pair_i] = (query_role, "#MASK")
                    masked_instance_tuple = tuple(masked_instance)

                    masked_instance_to_values[masked_instance_tuple].append(d[pair_i][1])
                    if masked_instance_tuple not in masked_instance_to_mask_position:
                        masked_instance_to_mask_position[masked_instance_tuple] = pair_i
                    else:
                        assert masked_instance_to_mask_position[masked_instance_tuple] == pair_i

        number_of_values = [len(masked_instance_to_values[mi]) for mi in masked_instance_to_values]
        number_of_unique_values = [len(set(masked_instance_to_values[mi])) for mi in masked_instance_to_values]
        print("\nBefore adding class-level masked instance")
        print("{} masked instances".format(len(masked_instance_to_values)))
        print("on average {} values for each unique masked instances".format(np.mean(number_of_values)))
        print("on average {} unique values for each unique masked instances".format(np.mean(number_of_unique_values)))

        if self.augment_class_level:
            # add class-level masked instance
            for query_role in self.query_roles:
                for d in positive_data:
                    query_values = []
                    class_value = None
                    for r, v in d:
                        if r == query_role:
                            query_values.append(v)
                        if r == "class":
                            class_value = v

                    if class_value:
                        for query_value in query_values:
                            masked_instance = [("class", class_value), (query_role, "#MASK")]
                            masked_instance_tuple = tuple(masked_instance)
                            masked_instance_to_values[masked_instance_tuple].append(query_value)
                            if masked_instance_tuple not in masked_instance_to_mask_position:
                                masked_instance_to_mask_position[masked_instance_tuple] = 1
                            else:
                                assert masked_instance_to_mask_position[masked_instance_tuple] == 1

            number_of_values = [len(masked_instance_to_values[mi]) for mi in masked_instance_to_values]
            number_of_unique_values = [len(set(masked_instance_to_values[mi])) for mi in masked_instance_to_values]
            print("\nAfter adding class-level masked instance")
            print("{} masked instances".format(len(masked_instance_to_values)))
            print("on average {} values for each unique masked instances".format(np.mean(number_of_values)))
            print("on average {} unique values for each unique masked instances".format(np.mean(number_of_unique_values)))

        self.masked_instance_to_values = masked_instance_to_values
        self.masked_instance_to_mask_position = masked_instance_to_mask_position
        self.masked_instances = list(masked_instance_to_values.keys())

    def get_raw_item(self, idx):
        masked_instance_tuple = self.masked_instances[idx]
        truth_values = self.masked_instance_to_values[masked_instance_tuple]
        mask_position = self.masked_instance_to_mask_position[masked_instance_tuple]
        return masked_instance_tuple, truth_values, mask_position

    def __len__(self):
        return len(self.masked_instances)

    def __getitem__(self, idx):

        masked_instance_tuple = self.masked_instances[idx]

        output = np.zeros(len(self.value2idx))
        for value in self.masked_instance_to_values[masked_instance_tuple]:
            output[self.value2idx[value]] = 1
        if self.label_smooth != 0.0:
            output = (1.0 - self.label_smooth) * output + (1.0 / len(output))

        input = []
        for r, v in masked_instance_tuple:
            input.append(self.role2idx[r])
            input.append(self.value2idx[v])

        for _ in range(self.max_arity - len(masked_instance_tuple)):
            input.append(self.role2idx["#PAD_TOKEN"])
            input.append(self.value2idx["#PAD_TOKEN"])

        mask_position = self.masked_instance_to_mask_position[masked_instance_tuple]

        # Convert to torch tensors
        input = torch.LongTensor(input)
        output = torch.FloatTensor(output)
        mask_position = torch.LongTensor([mask_position])

        if self.debug:
            print("input raw", masked_instance_tuple)
            print("output raw", self.masked_instance_to_values[masked_instance_tuple])
            print("input", input)
            print("output", output)
            print("mask_position", mask_position)
            print("input vectorized shape", input.shape)
            print("output shape", output.shape)
            print("mask_position shape", mask_position.shape)

        return input, output, mask_position

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """
        # data: input, output

        input = torch.stack([_[0] for _ in data], dim=0)
        output = torch.stack([_[1] for _ in data], dim=0)
        mask_position = torch.stack([_[2] for _ in data], dim=0)

        return input, output, mask_position


