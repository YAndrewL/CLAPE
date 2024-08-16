# -*- coding: utf-8 -*-
'''
@File   :  clape.py
@Time   :  2024/08/14 17:55
@Author :  Yufan Liu
@Desc   :  clape main file
'''


import os.path as osp
import re

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from .model import CNNOD


class Clape(object):
    def __init__(self, 
                 model_path,
                 threshold:float=0.5,
                 pretrained_cache:str="protbert",
                 ligand=None):
        
        self._sanity_check(threshold, ligand)
        self.threshold = threshold
        self.ligand = ligand
        self.model_path = model_path
        # pretrained model
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir=pretrained_cache)
        self.pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=pretrained_cache)
    
        # prediction model
        self.model = self._load_model(ligand)
    

    def _seq_embedding(self, seq):
        sequence_Example = ' '.join(seq)
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = self.tokenizer(sequence_Example, return_tensors='pt')
        last_hidden = self.pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1, :]
        return last_hidden.detach()
    
    def _sanity_check(self, thres, lig):
        if thres > 1 or thres < 0:
            raise ValueError("Threshold is out of range.")
        if lig not in ['DNA', 'RNA', 'AB']:
            raise ValueError(f"{lig} is not supported currently.")

    def _load_model(self, lig):
        model = CNNOD()
        model.load_state_dict(torch.load(osp.join(self.model_path, lig + ".pth")))
        return model
                    
    def _process_seq(self, input_file:str):
        assert input_file.split(".")[-1] in ['fasta', 'fa'], "Please assure suffix is .fasta or .fa"
        fasta = open(input_file, 'r').readlines()
        seq_ids = []
        seqs = []
        for line in fasta:
            if line.startswith('>'):
                seq_ids.append(line.strip())
            else:
                seqs.append(line.strip())
        if len(seq_ids) != len(seqs):
            raise ValueError("FASTA file is not valid.") 
        return seqs             
    
    def predict(self, input_file:str, keep_score=False):
        seqs = self._process_seq(input_file)
        features = []
        for s in seqs:
            features.append(self._seq_embedding(s).unsqueeze(0))
        results = []
        
        print(f"=====Predicting {self.ligand}-binding sites=====")
        self.model.eval()
        for f in features:
            score = self.model(f).squeeze(0).detach().numpy()[:, 1]
            out = ''.join([str(1) if x > self.threshold else str(0) for x in score])
            results.append(out) 
        print("=========Finished!=========")
        
        if keep_score:
            return score, results   
        else:
            return results   
    
    def switch_ligand(self, ligand):
        self.ligand = ligand
        self.model = self._load_model(ligand)
        
    def visualization(self, result):
        ...