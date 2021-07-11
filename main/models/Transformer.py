import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict


class Transformer(torch.nn.Module):

    def __init__(self, role2idx, value2idx, embedding_dim, max_arity, num_encoder_layer, num_attention_heads,
                 encoder_hidden_dim, encoder_dropout, encoder_activation, use_output_layer_norm, use_position_embedding,
                 pooling_method, use_mask_pos_output):
        super(Transformer, self).__init__()

        # params
        self.use_mask_pos_output = use_mask_pos_output
        self.pooling_method = pooling_method
        self.use_position_embedding = use_position_embedding

        self.embedding_dim = embedding_dim

        self.max_arity = max_arity

        self.value2idx = value2idx
        if "#MASK" not in self.value2idx:
            self.value2idx["#MASK"] = len(self.value2idx)

        # model
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_attention_heads, encoder_hidden_dim,
                                                 encoder_dropout, encoder_activation)
        if not use_output_layer_norm:
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layer)
        else:
            raise Exception("use output layer norm not implemented")

        self.role_embeddings = torch.nn.Embedding(len(role2idx), embedding_dim, padding_idx=role2idx["#PAD_TOKEN"])
        self.value_embeddings = torch.nn.Embedding(len(self.value2idx), embedding_dim, padding_idx=value2idx["#PAD_TOKEN"])

        if self.use_position_embedding:
            raise Exception("Use position embedding not implemented")

        if self.use_mask_pos_output:
            # only use embedding from mask position for final prediction
            self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        else:
            # use embedding from all positions for final prediction
            if self.pooling_method == "concat":
                self.fc = torch.nn.Linear(embedding_dim * max_arity, embedding_dim)
            else:
                self.fc = torch.nn.Linear(embedding_dim, embedding_dim)

        self.bceloss = torch.nn.BCELoss()

    def forward(self, input, mask_position):

        batch_size = input.shape[0]
        arity = int(input.shape[1] / 2)
        assert arity == self.max_arity

        # input: [batch_size, sequence_length * 2]
        # input in format [r1, v1, r2, v2, ..., rN, vN]
        # sequence_length := N
        # mask_position: [batch_size, ]

        roles_ids = input[:, 0::2].flatten()
        values_ids = input[:, 1::2].flatten()

        # roles_embedded: [batch_size, arity, embedding_size]
        roles_embedded = self.role_embeddings(roles_ids).view(batch_size, arity, self.embedding_dim)
        values_embedded = self.value_embeddings(values_ids).view(batch_size, arity, self.embedding_dim)
        input_embedded = roles_embedded + values_embedded

        # input to transformer needs to have dimenion [arity, batch_size, embedding_size]
        input_embedded = input_embedded.transpose(1, 0)

        # create a mask for paddings
        padding_mask = input[:, 0::2] == 0

        # encode: [arity, batch_size, embedding_size]
        encode = self.encoder(input_embedded, src_key_padding_mask=padding_mask)

        if self.use_mask_pos_output:
            # select encoding at the mask position
            encode = encode.transpose(1, 0)
            encode = encode[torch.arange(batch_size), mask_position.flatten()]
            # encode: [batch_size, embedding_size]
        else:
            if self.pooling_method == 'concat':
                encode = encode.transpose(1, 0).reshape(-1, self.max_arity * self.embedding_dim)
            elif self.pooling_method == "avg":
                encode = torch.mean(encode, dim=0)
            elif self.pooling_method == "min":
                encode, _ = torch.min(encode, dim=0)

        encode = self.fc(encode)

        # similarity: [batch_size, number of entities]
        similarity = torch.mm(encode, self.value_embeddings.weight.transpose(1, 0))
        score = torch.sigmoid(similarity)

        return score

    def criterion(self, predictions, labels):
        return self.bceloss(predictions, labels)