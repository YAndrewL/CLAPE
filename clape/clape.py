# -*- coding: utf-8 -*-
'''
@File   :  clape.py
@Time   :  2024/08/14 17:55
@Author :  Yufan Liu
@Desc   :  clape main file
'''


import os
import os.path as osp
import re

import torch
import torch.nn as nn
from Bio import SeqIO
from Bio.PDB import PDBList
from transformers import BertModel, BertTokenizer

from .model import CNNOD
from .vis import visualize as vis_tool


class Clape(object):
    def __init__(self, 
                 model_path,
                 ligand,
                 threshold:float=0.5,
                 pretrained_cache:str="protbert",
                 ):
        
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
        assert input_file.split(".")[-1] in ['fasta', 'fa'], "Please ensure suffix is .fasta or .fa"
        sequences = SeqIO.parse(input_file, "fasta")
        seq_ids = []
        seqs = []
        for seq in sequences:
            seq_ids.append(seq.id)
            seqs.append(str(seq.seq))
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
        
        if keep_score:
            return score, results   
        else:
            return results  
        
    def predict_from_pdb(self, 
                         chain, 
                         pdb_file=None,
                         pdb_id=None, 
                         pdb_cache=".",
                         keep_score=False,
                         visualize=False):  
        if pdb_id:
           pdbl = PDBList()
           try:
               pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=pdb_cache, overwrite=True)
               raw_file = osp.join(pdb_cache, "pdb" + pdb_id.lower() + ".ent")
               file = osp.join(pdb_cache, pdb_id.lower() + ".pdb")
               os.rename(raw_file, file)
           except Exception as e:
               print(e)
        else:
            file = pdb_file
        seqs = SeqIO.parse(file, "pdb-atom")
        chain_id = None
        for seq in seqs:
            if seq.id[-1] == chain:
                s = seq
                chain_id = seq.id
        if not chain_id:
            raise KeyError(f"{chain} is not a valid protein chain in {osp.split(file)[-1]}")

        print(f"=====Predicting {self.ligand}-binding sites=====")
        self.model.eval()
        feature = self._seq_embedding(s).unsqueeze(0)
        score = self.model(feature).squeeze(0).detach().numpy()[:, 1]
        out = ''.join([str(1) if x > self.threshold else str(0) for x in score])
        
        if visualize:
            try: 
                vis_tool(pdb_file=pdb_file, chain=chain, result=out, out_file="out.pse")
            except:
                raise RuntimeError("Trying to visualize with an invalid structure input.")
        
        if keep_score:
            return score, out
        else:
            return out
         
    def switch_ligand(self, ligand):
        self.ligand = ligand
        self.model = self._load_model(ligand)