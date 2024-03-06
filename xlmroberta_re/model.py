import torch
import torch.nn as nn
import copy

from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from dataclasses import dataclass

from infoxlm_re.model import RelationExtractionOutput, RegionExtractionDecoder


class XLMRoberaForRelationExtraction(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder = RegionExtractionDecoder(model.config)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

    def forward(self, input_ids,
                bbox=None,
                labels=None,
                image=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                entities=None,
                relations=None):

        outputs = self.model(
            input_ids=input_ids,
        )

        seq_length = input_ids.size(1)
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.decoder(
            sequence_output, entities, relations)

        return RelationExtractionOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )
