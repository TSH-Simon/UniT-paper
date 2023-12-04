import torch
import torch.nn as nn

import sys
from transformers.modeling_bert_lipnorm import BertForSequenceClassification
sys.path.append('..')

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.model_name = 'Base_model'
        self.args = args
      
        self.base_model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),num_labels=2)
        self.bert_model = self.base_model.bert
        self.embedding_encoder = self.bert_model.embeddings.word_embeddings
        self.position_encoder = self.bert_model.embeddings.position_embeddings
        self.token_type_encoder = self.bert_model.embeddings.token_type_embeddings
        self.embedding_layer_norm = self.bert_model.embeddings.LayerNorm
        self.embedding_dropout = self.bert_model.embeddings.dropout
        self.args.embedding_dim = 768

    def get_emb(self, tokens):
        if self.args.embedding_training:
            return self.embedding_encoder(tokens)
        else:
            return self.embedding_encoder(tokens).detach()
