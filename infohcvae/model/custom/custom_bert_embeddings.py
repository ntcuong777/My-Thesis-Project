import torch
import torch.nn as nn
from transformers import BertModel


class CustomBertEmbedding(nn.Module):
    def __init__(self, bert_model):
        super(CustomBertEmbedding, self).__init__()
        bert_embeddings = BertModel.from_pretrained(bert_model).embeddings
        self.word_embeddings = bert_embeddings.word_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        print(words_embeddings.size())
        print(token_type_embeddings.size())
        print(position_embeddings.size())
        embeddings = words_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
