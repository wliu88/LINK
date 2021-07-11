import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RoleValueDataset(Dataset):
    """
    :ivar label_smooth
    :ivar debug
    :ivar role2idx
    :ivar value2idx
    :ivar inputs
    :ivar outputs
    """

    def __init__(self, positive_data, negative_data, max_arity, role2idx, value2idx, label_smooth, debug=False):
        super().__init__()

        self.debug = debug

        # params
        self.label_smooth = label_smooth
        self.max_arity = max_arity

        # dicts
        self.role2idx = role2idx
        self.value2idx = value2idx
        # add pad token
        # self.role2idx["#PAD_TOKEN"] = len(self.role2idx)
        # self.value2idx["#PAD_TOKEN"] = len(self.value2idx)

        # data
        self.inputs = None
        self.outputs = None
        self.load_and_process_data(positive_data, negative_data)

    def load_and_process_data(self, positive_data, negative_data):

        print("Loading data...")

        # verify that data is list of instances, where each instance is a list of role-value tuples
        assert type(positive_data) == type(negative_data) == list
        assert type(positive_data[0]) == type(negative_data[0]) == list
        assert type(positive_data[0][0]) == type(negative_data[0][0]) == tuple

        max_arity = 0
        inputs_raw = positive_data + negative_data
        for d in inputs_raw:
            if len(d) > max_arity:
                max_arity = len(d)
        print("set max arity {}, max arity in data is {}".format(self.max_arity, max_arity))
        assert max_arity <= self.max_arity, "max arity in data is larger than the value specified"

        # flatten and add paddings
        inputs = []
        for d in inputs_raw:
            instance = []
            arity = len(d)
            for role, value in d:
                instance.append(role)
                instance.append(value)
            # add paddings
            for _ in range(self.max_arity - arity):
                instance.append("#PAD_TOKEN")
                instance.append("#PAD_TOKEN")
            inputs.append(instance)

        outputs = [1] * len(positive_data) + [-1] * len(negative_data)

        self.inputs = inputs
        self.outputs = outputs
        print("Input size is {} and output size is {}".format(len(inputs), len(outputs)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_raw = self.inputs[idx]
        output = self.outputs[idx]

        input = []
        for i in range(int(self.max_arity)):
            input.append(self.role2idx[input_raw[2 * i]])
            input.append(self.value2idx[input_raw[2 * i + 1]])

        # Convert to torch tensors
        input = torch.LongTensor(input)
        output = torch.FloatTensor([output])

        if self.debug:
            print("input raw", input_raw)
            print("input", input)
            print("output", output)
            print("input vectorized shape", input.shape)
            print("output shape", output.shape)

        return input, output

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """
        # data: input, output

        input = torch.stack([_[0] for _ in data], dim=0)
        output = torch.stack([_[1] for _ in data], dim=0)

        return input, output
