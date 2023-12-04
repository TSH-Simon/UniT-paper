from __future__ import print_function

import numpy as np

import os
import torch
import torch.nn.parallel

import torch.utils.data

from torch import nn

class DataGather(object):
    def __init__(self, keys,options,save_path=None):
        self.keys = keys
        self.data = self.get_empty_data_dict()
        self.options=options
        assert len(self.keys)==len(self.options)

        self.save_path= save_path
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path=os.path.join(save_path,'log.txt')

    def get_empty_data_dict(self):
        dic={}
        for key in self.keys:
            dic[key]=[]
        return dic

    def insert(self, keys,data):
        assert len(keys)==len(data)
        for i in range(len(keys)):
            if isinstance(data[i],torch.Tensor):
                tem=data[i].item()
            else:

                tem=data[i]
            self.data[keys[i]].append(tem)

    def flush(self):
        self.data = self.get_empty_data_dict()

    def get_mean(self):
        res=[]
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.mean(self.data[key]))
            else:
                res.append(0)
        return res

    def get_min(self):
        res = []
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.min(self.data[key]))
            else:
                res.append(None)
        return res

    def get_max(self):
        res = []
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.max(self.data[key]))
            else:
                res.append(None)
        return res

    def get_sum(self):
        res=[]
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.sum(self.data[key]))
            else:
                res.append(0)
        return res

    def get_report(self,):
        mins=self.get_min()
        means=self.get_mean()
        maxs=self.get_max()
        res=[]
        for i in range(len(mins)):
            res.append([mins[i],maxs[i],means[i]])
        res1=[]
        for i in range(len(mins)):
            res1.append(res[i][self.options[i]])
        return res1

    def report(self,additional=None):
        res=self.get_report()
        string=''
        for ind,key in enumerate(self.keys):
            string=string+str(key)+' '
            string=string+str(res[ind])+' '
        if additional is not None:
            string=str(additional)+' '+string
        print(string)
        if self.save_path is not None:
            with open(self.save_path, 'a+') as f:
                f.write(string+'\n')
    def write(self,string):
        print(string)
        if self.save_path is not None:
            with open(self.save_path, 'a+') as f:
                f.write(str(string)+'\n')

def correct_rate_func(out,label):
    return (torch.argmax(out,1)==label).float().mean()

class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


