from bert_cnn_classifier_2_gauss_linear.bert_base import BaseModel
from bert_cnn_classifier_2_gauss_linear.textcnn_lipdy_bert import EncoderModel
import torch
import numpy as np
import math
import os
import torch.nn.functional as F
import torch.nn as nn

class bert_cnnencoder(BaseModel):
    def __init__(self, args):
        super(bert_cnnencoder, self).__init__(args)
        self.args = args
        self.encoder = EncoderModel()

        self.NLL = torch.nn.CrossEntropyLoss(reduction='none')
        self.marginloss = nn.MultiMarginLoss(margin=0,reduction='none')



        self.celoss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, input_orig_ids, attention_mask, token_type_ids, labels):
        input_emb  = self.get_emb(input_ids)
        bert_emb  = self.classify_by_bert(input_ids, attention_mask, token_type_ids, labels, input_emb )

        input_emb2 = self.get_emb(input_orig_ids)
        bert_emb2 = self.classify_by_bert(input_orig_ids, attention_mask, token_type_ids, labels,input_emb2 )


        clas,lip_emb1 = self.encoder(bert_emb,0.0)
        clas2, lip_emb2 = self.encoder(bert_emb2,self.args.var)

        norm_diff = torch.norm(lip_emb1-lip_emb2,dim=1)

        margin_loss = self.marginloss(-clas2,labels)*math.sqrt(2) # /2*math.sqrt(2)

        hinge_loss = margin_loss - norm_diff
        hinge_loss[hinge_loss> -self.args.margin] = 0
        hinge_loss = hinge_loss.mean()
        ce_loss = self.celoss(clas, labels)


        loss = ce_loss - self.args.norm_weight*hinge_loss

        return loss, clas

    
    def classify_by_bert(self,input_ids, attention_mask, token_type_ids, labels, tem):

        if tem.shape[1]>512:
            tem=tem[:,0:512,:]
            attention_mask=attention_mask[:,0:512]

        pool_emb = self.base_model(inputs_embeds=tem, attention_mask=attention_mask, labels=labels)

        return pool_emb

        