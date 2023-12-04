from model.decoder import Decoder,Decoder_RNN
from model.encoder import Encoder,Encoder_IBP,TextCNN,CNNModel,BOWModel,CNNModel_bypass
from model.base import BaseModel
import torch
from model.ibp import max_diff_norm
import numpy as np
import model.ibp as ibp
import os
from aux_function import correct_rate_func
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from scipy.stats import binom_test
from torch.distributions.normal import Normal
from utils import IntervalBoundedTensor
import torch.nn.functional as F
import math

from model.cnndecoder import EncoderModel

def grl_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1

class Certify_VAE(BaseModel):

    def __init__(self, args):
        super(Certify_VAE, self).__init__(args)
        self.args=args
        self.args.ibp_encode = True

        self.NLL = torch.nn.CrossEntropyLoss(reduction='none')
        self.STD=args.std
        self.celoss = torch.nn.CrossEntropyLoss()


        self.margin = 0.6
        self.marginloss = torch.nn.MultiMarginLoss(margin=0,reduction='none')

        self.cnndecoder = EncoderModel()

        self.var =  0.1



    def coef_function(self,anneal_function, step, k, x0,start=0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return k*min(1, max((step-start),0) / x0)
        elif anneal_function == 'zero':
            return 0.0

    def reparameterize(self,mu,sigma):
        z=torch.randn(mu.shape).to(mu.device)

        return mu + self.STD*z

    def classify_on_hidden(self,input,label):
        mu,log_sigma,mu_ibp = self.encoder(input)
        if self.args.ibp_encode:
            mu_val=mu.val
            log_sigma_val=log_sigma.val
        else:
            mu_val = mu
            log_sigma_val = log_sigma

        z = mu_val
        logit=self.decoder.hidden_to_label(z)
        tem = torch.tensor(0.0).to(input["sent"].device)
        if self.args.radius_on_hidden_classify:
            radius = self.soft_radius(logit, input['label'])
        else:
            radius=tem

        loss_clas_hidden = self.NLL(logit, label).mean()
        if self.args.ibp_loss:
            ibp_loss=self.compute_IBP_loss(mu)
        else:
            ibp_loss=tem
        return loss_clas_hidden,correct_rate_func(logit,label),ibp_loss,radius


    def encode(self,input,sample_num=1):
        mu,log_sigma,_=self.encoder(input)
        if self.args.ibp_encode:
            mu_val=mu.val
            log_sigma_val=log_sigma.val
        else:
            mu_val = mu
            log_sigma_val = log_sigma

        mu_val = mu_val.repeat(sample_num, 1)
        log_sigma_val = log_sigma_val.repeat(sample_num, 1)

        z = mu_val

        return z,mu_val,log_sigma_val,mu

    def decode(self,z,input_decode):
        return self.decoder(z,input_decode)

    def loss_fn(self,logp1,logp2, target, mask, mean, logv,mu_ibp):
        tem = torch.tensor(0.0).to(logp1.device)
        batch_size=logp2.shape[0]


        if self.args.reconstruct_by_RNN:
            logp1 = logp1.view(-1, logp1.size(2))
            target=target.view(-1)
            mask=mask.view(-1)

            NLL_loss1 = (self.NLL(logp1, target)*mask).sum()/batch_size
        else:
            NLL_loss1=tem

        logp2 = logp2.reshape([-1, logp2.size(2)])
        target = target.reshape([-1])
        mask = mask.reshape([-1])

        NLL_loss2 = (self.NLL(logp2, target) * mask).sum() / batch_size

        KL_loss=tem

        return NLL_loss1,NLL_loss2, KL_loss

    def generate(self,latent_variable):
        cnn_out = self.decoder.conv_decoder(latent_variable)
        state=None
        init_input_decoder =self.args.tokens_to_ids['[PAD]'] * torch.ones([latent_variable.shape[0], 1]).to(latent_variable.device).type(torch.long)
        res=[]
        input=init_input_decoder
        for i in range(cnn_out.shape[1]):

            input_emb = self.get_emb(input)
            logits1,state = self.decoder.rnn_decoder(cnn_out[:,i,:].unsqueeze(1), input_emb, initial_state=state,return_final=True)
            next = logits1.argmax(2)
            res.append(next)
            input=next
        res=torch.cat(res,1)
        return res

    def generate_from_text(self,input):
        mask = None
        ibp_input = None
        sent = input["sent"]
        text_like_syn = input["text_like_syn"]
        text_like_syn_valid = input["text_like_syn_valid"]
        if "mask" in input:
            mask = input["mask"]

        if self.args.ibp_encode:


            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)

            input = ibp_input
        else:
            input = self.get_emb(sent)
        z, mu, log_sigma,_ = self.encode(input)
        res=self.generate(z)

        return res

    def input_preprocess(self, input, convex_hull=True):

        sent = input["sent"]

        if self.args.ibp_encode:
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn"]
            text_like_syn_valid = input["text_like_syn_valid"]
            if "mask" in input:
                mask = input["mask"]


            if convex_hull:
                if ibp_input == None:
                    ibp_input, input_syn_l2_max, perturb_val = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask, perturb=False)

            else:
                ibp_input = self.ibp_input_from_convex_hull(sent, None, None, mask)

            input_emb = ibp_input
        else:
            input_emb = self.get_emb(sent)

        input['input_emb'] = input_emb
        input['input_syn_l2_max'] = input_syn_l2_max
        input['perturb_emb'] = perturb_val
 
        

        if 'premises' in input:
            sent = input["premises"]
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn_premises"]
            text_like_syn_valid = input["text_like_syn_valid_premises"]
            if "mask_premises" in input:
                mask = input["mask_premises"]


            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)

            input_emb = ibp_input
            input['input_emb_premises'] = input_emb

        if 'hypothesis' in input:
            sent = input["hypothesis"]
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn_hypothesis"]
            text_like_syn_valid = input["text_like_syn_valid_hypothesis"]
            if "mask_hypothesis" in input:
                mask = input["mask_hypothesis"]


            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)

            input_emb = ibp_input
            input['input_emb_hypothesis'] = input_emb

        return input

    def compute_IBP_loss(self,mu_ibp):
        if self.args.IBP_loss_type == 'mean':
            tem2 = mu_ibp.ub - mu_ibp.lb
            ibp_loss = tem2.mean(1).mean(0)
        elif self.args.IBP_loss_type == 'l2':
            ibp_loss = torch.norm(torch.maximum(
                (mu_ibp.ub - mu_ibp.val).abs(), (mu_ibp.val - mu_ibp.lb).abs()
            ), dim=1)
        return ibp_loss

    def forward_vae(self,input,coeff=1):

        tem = torch.tensor(0.0).to(input["label"].device)

        if self.args.encoder_type != 'cnnibp':
            z,mu,log_sigma,mu_ibp=self.encode(input)
        else:
            z, mu, log_sigma, mu_ibp = self.encode(input)

        input_decoder=None
        recons1,recons2,emb1,emb2,clas=self.decode(z,input_decoder)


        if self.args.info_loss=='reconstruct':
            NLL_loss1, NLL_loss2, KL_loss = self.loss_fn(recons1, recons2, input["sent"],
                                                                               input["mask"], mu, log_sigma, mu_ibp,
                                                                               )
        else:
            NLL_loss1=tem
            NLL_loss2=tem
            KL_loss=tem

        ibp_loss = input['input_syn_l2_max']

        return NLL_loss1, NLL_loss2, KL_loss,ibp_loss,recons1, recons2, z, mu, log_sigma,emb1,emb2,clas



    def data_to_loss(self,input):
        NLL_loss1, NLL_loss2, KL_loss, ibp_loss, recons1, recons2, z, mu, log_sigma=self.forward_vae(input)
        return NLL_loss1.mean(), NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), recons1, recons2, z, mu, log_sigma


    def classify_by_bert(self,input,emb2,coeff,sample_num=1):


        weight = self.bert_model.embeddings.word_embeddings.weight.detach()

        
        new_emb = emb2
        new_emb = torch.nn.functional.normalize(new_emb, p=2, dim=2) * 1.4

        try:
            new_emb.register_hook(grl_hook(coeff))
        except:
            pass

        tem = weight[self.args.tokens_to_ids['[CLS]'], :].unsqueeze(0).repeat(emb2.shape[0], 1).unsqueeze(1)
        tem = torch.cat([tem, new_emb], 1)
        mask = torch.cat([torch.ones([input['label'].shape[0], 1]).to(input['mask'].device), input['mask']], 1)

        if tem.shape[1]>512:
            tem=tem[:,0:512,:]
            mask=mask[:,0:512]

        pooled_output = self.base_model(inputs_embeds=tem, attention_mask=mask.repeat([sample_num,1]), labels=input['label'].repeat([sample_num,1]))


        return pooled_output

    def soft_radius(self,logits,label):
        pred = torch.softmax(logits, 1)
        acc = (pred.argmax(1) == label).float()
        pred_sorted, _ = pred.sort(1, descending=True)
        PA = pred_sorted[:, 0] * acc + pred_sorted[:, 1] * (1 - acc)
        PB = pred_sorted[:, 1] * acc + pred_sorted[:, 0] * (1 - acc)
        PA = torch.minimum(PA, self.args.soft_upper_bound * torch.ones_like(PA))
        PB = torch.maximum(PB, (1-self.args.soft_upper_bound) * torch.ones_like(PB))
        m = Normal(torch.tensor([0.0]).to(pred.device),
                   torch.tensor([1.0]).to(pred.device))

        radius = (m.icdf(PA) - m.icdf(PB))*self.STD  /2
        return radius

    def ortho_certificates(self,output, class_indices):
        batch_size = output.shape[0]
        batch_indices = torch.arange(batch_size)
        
        onehot = torch.zeros_like(output).to(output.device)
        onehot[torch.arange(output.shape[0]), class_indices] = 1.
        output_trunc = output - onehot*1e6

        output_class_indices = output[batch_indices, class_indices]
        output_nextmax = torch.max(output_trunc, dim=1)[0]
        output_diff = output_class_indices - output_nextmax
        
        return output_diff

    def forward(self,input,coeff,input_preprocess=True,idirect_nput_to_bert_by_sent=True):

        input = self.input_preprocess(input)
        


        input['input_orig'] = input['input_emb'].val
        input['input_emb'].val=input['input_emb'].val+torch.normal(mean=torch.zeros_like(input['input_emb'].val),std=self.STD)

        emb_orig = input['input_orig']
        feat_orig = self.classify_by_bert(input,emb_orig ,coeff)
                    
        emb_gauss = input['input_emb'].val 
        feat_gauss=self.classify_by_bert(input,emb_gauss,coeff)


        clas,lip_emb1 = self.cnndecoder(feat_gauss,0.0)
        clas2, lip_emb2 = self.cnndecoder(feat_orig,self.var)


        radius=self.soft_radius(clas,input['label'])
        

        norm_diff = torch.norm(lip_emb1 -lip_emb2,dim=1)
        margin_loss = F.relu(self.ortho_certificates(clas2, input['label'])/math.sqrt(2))
        hinge_loss = margin_loss - norm_diff
        hinge_loss[hinge_loss> -self.margin] = 0
        hinge_loss = hinge_loss.mean()
        ce_loss = self.celoss(clas, input['label'])

        loss_cls=ce_loss-hinge_loss

        
        ibp_loss = input['input_syn_l2_max']

        return loss_cls,correct_rate_func(clas,input['label']),  ibp_loss,radius




    def certify_prediction(self,input,sample_num,alpha=0.05):

        input=self.input_preprocess(input)

        z, mu, log_sigma, mu_ibp = self.encode(input, sample_num)
        input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input['sent'].shape[0], 1])
                                  .to(input['sent'].device), input['sent'][:, 0:-1]], 1).type(torch.long)

        input_decoder = self.get_emb(input_decoder)
        input_decoder = input_decoder.repeat(sample_num, 1, 1)
        recons1, recons2, emb1, emb2, clas_hidden = self.decode(z, input_decoder)

        clas = self.classify_by_bert(input, recons1, recons2, emb1, emb2, 0, sample_num)

        count = torch.nn.functional.one_hot(clas.logits.argmax(dim=1).reshape([input['sent'].shape[0], sample_num]),num_classes=self.args.num_classes).sum(1)
        _,count_sort = count.sort(dim=1,descending=True)
        count_max=count_sort[:,0]
        cont_second=count_sort[:,1]
        count_1=count.gather(1,count_max.unsqueeze(1)).detach().cpu().numpy()
        count_2=count.gather(1,cont_second.unsqueeze(1)).detach().cpu().numpy()
        P=self.binomial_test(count_1,count_2,z.device,0.5)
        res = (P < alpha) * count_max + (P>alpha) * -1
        return res




    def certify(self,input,sample_num,sample_num_2,alpha=0.05):
        input = self.input_preprocess(input)


        emb2 = input['input_emb'].val.repeat(sample_num, 1,1)
        emb2 = emb2+torch.normal(mean=torch.zeros_like(emb2),std=self.STD)
        clas_feat=self.classify_by_bert(input,emb2,0.0,  sample_num)
        clas,lip_emb1 = self.cnndecoder(clas_feat,0.0)
        mu_ibp = input['input_syn_l2_max']


        if self.args.soft_verify:
            pred = torch.softmax(clas, 1).reshape([sample_num,input['sent'].shape[0],self.args.num_classes]).sum(0)

            count_max=pred.argmax(dim=1)
        else:
            count=torch.nn.functional.one_hot(clas.argmax(dim=1).reshape([sample_num,input['sent'].shape[0]]),self.args.num_classes).sum(0)
            pred = torch.softmax(clas, 1).reshape(
                [sample_num, input['sent'].shape[0], self.args.num_classes]).sum(0)
            count_max=count.argmax(dim=1)


        assert sample_num_2>=300 and sample_num_2%300 ==0
        sample_num_each=100
        iter_num=sample_num_2//sample_num_each
        result=[]
        for _ in range(iter_num):

            emb2 = input['input_emb'].val.repeat(sample_num_each, 1,1)
            emb2 = emb2+torch.normal(mean=torch.zeros_like(emb2),std=self.STD)
            clas_feat=self.classify_by_bert(input,emb2,0.0, sample_num_each)
            clas,lip_emb1 = self.cnndecoder(clas_feat,0.0)
            result.append(clas)

        result=torch.cat(result,dim=0)
        if self.args.soft_verify:
            count_soft=torch.softmax(self.args.soft_beta*result, 1).reshape([sample_num_2,input['sent'].shape[0],self.args.num_classes]).sum(0)
            sum_square=torch.softmax(self.args.soft_beta*result, 1).reshape([sample_num_2,input['sent'].shape[0],self.args.num_classes]).square().sum(0)
            P_A, radius = self.lower_confidence_bound_soft(count_soft.gather(1, count_max.unsqueeze(1)).round(), sample_num_2, alpha,sum_square.gather(1, count_max.unsqueeze(1)))
        else:
            count=torch.nn.functional.one_hot(result.argmax(dim=1).reshape([sample_num_2,input['sent'].shape[0]]),self.args.num_classes).sum(0)
            P_A,radius=self.lower_conf_bound(count.gather(1,count_max.unsqueeze(1)),sample_num_2,alpha)

        res=(P_A>0.5)*count_max+(P_A<0.5)*-1

        return res, radius, mu_ibp






    def input_pertubation(self,input,sample_num):

        input['sent']=input['sent'].repeat(sample_num,1)
        input['mask'] = input['mask'].repeat(sample_num,1)
        input['token_type_ids'] = input['token_type_ids'].repeat(sample_num,1)
        input['label'] = input['label'].repeat(sample_num)
        input["text_like_syn"]=input["text_like_syn"].repeat(sample_num,1,1)
        input["text_like_syn_valid"] = input["text_like_syn_valid"].repeat(sample_num, 1, 1)

        seed = torch.rand(input["text_like_syn"].shape).to(input["text_like_syn"].device)*input["text_like_syn_valid"]
        index=torch.argmax(seed,dim=2)
        sent=torch.gather(input["text_like_syn"],2,index.unsqueeze(-1)).squeeze(-1)

        if self.args.ibp_encode:
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn"]
            text_like_syn_valid = input["text_like_syn_valid"]
            if "mask" in input:
                mask = input["mask"]

            if "ibp" in input:
                ibp_input = input["ibp_input"]

            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                input["ibp_input"] = ibp_input
            input_emb = ibp_input
        else:
            input_emb = self.get_emb(sent)

        input['input_emb'] = input_emb

        return input

    def lower_conf_bound(self,count,num,alpha=0.05):
        P_A= proportion_confint(count.detach().cpu().numpy(),num,alpha=2*alpha,method='beta')[0]
        radius = self.STD*norm.ppf(P_A)
        return torch.Tensor(P_A).to(count.device).squeeze(),torch.Tensor(radius).to(count.device).squeeze()

    def binomial_test(self,count_max,count_second,device,P=0.5):
        res=[]
        for i in range(count_max.shape[0]):
            res.append(binom_test(count_max[i],count_max[i]+count_second[i],P))
        return torch.Tensor(res).to(device).squeeze()





    def lower_confidence_bound_soft(self, NA, N, alpha, ss):
        NA=NA.detach().cpu().numpy()
        ss=ss.detach().cpu().numpy()
        sample_variance = (ss - NA * NA / N) / (N - 1)
        if sample_variance < 0:
            sample_variance = 0
        t = np.log(2 / alpha)
        P_A=NA / N - np.sqrt(2 * sample_variance * t / N) - 7 * t / 3 / (N - 1)
        radius = self.STD * norm.ppf(P_A)
        return  torch.Tensor(P_A).to(self.args.device[0]).squeeze(),torch.Tensor(radius).to(self.args.device[0]).squeeze()


    def ascc_certify(self, input, attack_type_dict):
        text_like_syn = input["text_like_syn"]
        text_like_syn_valid = input["text_like_syn_valid"]
        device = text_like_syn.device
        y = input["label"]
        emb_ibp = self.input_preprocess(input, convex_hull=False)['input_emb']

        num_steps = attack_type_dict['num_steps']
        loss_func = attack_type_dict['loss_func']
        w_optm_lr = attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']

        syn, _ = self.build_convex_hull(text_like_syn, text_like_syn_valid)
        batch_size, text_len, embd_dim = emb_ibp.val.shape
        batch_size, text_len, syn_num, embd_dim = syn.shape

        w = torch.empty(batch_size, text_len, syn_num, 1).to(device).float()
        torch.nn.init.kaiming_normal_(w)
        w.requires_grad_()
        params = [w]
        optimizer = torch.optim.Adam(params, lr=w_optm_lr, weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww = w * syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500 * (
                        syn_valid.reshape(batch_size, text_len, syn_num, 1) - 1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww = w * syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500 * (
                        syn_valid.reshape(batch_size, text_len, syn_num, 1) - 1)
            return ww

        def get_comb(p, syn):
            return (p * syn.detach()).sum(-2)

        input_ori = dict([])
        input_ori["input_emb"] = emb_ibp
        input_ori["mask"] = input["mask"].detach()
        input_ori["label"] = input["label"]

        for step in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, text_like_syn_valid)
                embd_adv = get_comb(F.softmax(ww, -2), syn)
                embd_adv_ibp = IntervalBoundedTensor(embd_adv, embd_adv, embd_adv)
                input_ori["input_emb"] = embd_adv_ibp
                loss_cls, accu, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.forward(input_ori, 1)
                loss = -loss_cls
                loss.backward()
                optimizer.step()

        ww_discrete = ww
        embd_adv = get_comb(F.softmax(ww * (1e10), -2), syn).detach()
        embd_adv_ibp = IntervalBoundedTensor(embd_adv, embd_adv, embd_adv)
        input_ori["input_emb"] = embd_adv_ibp

        return input_ori

    def emperical_certify(self,input):
        input = self.input_preprocess(input)
        input_per = self.input_pertubation(input,20)

        z, mu, log_sigma, mu_ibp = self.encode(input)
        z_per, mu_per, log_sigma_per, mu_ibp_per = self.encode(input_per)
        ub = mu_per.max(0)[0]
        lb = mu_per.min(0)[0]
        ub_ibp=mu_ibp.ub
        lb_ibp=mu_ibp.lb
        ind1=ub_ibp[0,:]>=ub
        ind2=lb_ibp[0,:]<=lb
        bounded=ind1.prod()*ind2.prod()
        input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input_per['sent'].shape[0], 1])
                                  .to(input_per['sent'].device), input_per['sent'][:, 0:-1]], 1).type(torch.long)

        input_decoder = self.get_emb(input_decoder)
        recons1, recons2, emb1, emb2, clas_hidden = self.decode(mu_per, input_decoder)

        clas = self.classify_by_bert(input_per, recons1, recons2, emb1, emb2, 0)

        return bounded.float(),correct_rate_func(clas.logits, input_per[
            'label'])
