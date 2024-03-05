import torch
import torch.nn as nn
import copy

from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from dataclasses import dataclass

class BiaffineAttention(nn.Module):
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.
    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.
    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.Bilinear(
            in_features, in_features, out_features, bias=False)
        self.linear = nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class RegionExtractionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(
            3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(
                list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set(
                [i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(
                positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]],
                           entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]],
                           entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(
                relations[b]["label"], device=device)
            entities_start_index = torch.tensor(
                entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(
                logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations



@dataclass
class RelationExtractionOutput(ModelOutput):
    """
    Class for outputs of [`LayoutLMv2ForRelationExtraction`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`:
            Classification (or regression if config.num_labels==1) loss.
        entities (`list[dict]`):
            List of dictionaries (one per example in the batch). Each dictionary contains 3 keys: `start`, `end` and
            `label`.
        relations (`list[dict]`):
            List of dictionaries (one per example in the batch). Each dictionary contains 4 keys: `start_index`,
            `end_index`, `head` and `tail`.
        pred_relations (`list[dict]`):
            List of dictionaries (one per example in the batch). Each dictionary contains 7 keys: `head`, `head_id`,
            `head_type`, `tail`, `tail_id`, `tail_type` and `type`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    pred_relations: dict = None
    entities: dict = None
    relations: dict = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class InfoXLMForRelationExtraction(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.relation_extraction_decoder = RegionExtractionDecoder(
            model.config)
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
        print(outputs)

        seq_length = input_ids.size(1)
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.relation_extraction_decoder(
            sequence_output, entities, relations)

        return RelationExtractionOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )
