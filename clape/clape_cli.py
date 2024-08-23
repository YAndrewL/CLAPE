# -*- coding: utf-8 -*-
# @Time         : 2023/5/10 15:28
# @Author       : Yufan Liu
# @Description  : predicting process


import argparse
import os
import re
import warnings

import torch
import torch.nn as nn
from Bio import SeqIO
from transformers import BertModel, BertTokenizer

from .model import CNNOD

warnings.filterwarnings('ignore')

def main():
    # load instructions
    parse = argparse.ArgumentParser()
    parse.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA, RNA, and antibody', choices=['DNA', 'RNA', 'AB'], required=True)
    parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
    parse.add_argument('--input', '-i', help='Input protein sequences in FASTA format', required=True)
    parse.add_argument('--output', '-o', help='Output file path, default clape_result.txt',
                    default='clape_result.txt')
    parse.add_argument('--model', '-p', help='Path for downloaded model parameters', required=True)
    parse.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')

    args = parse.parse_args()

    # parameter judge
    if args.threshold > 1 or args.threshold < 0:
        raise ValueError("Threshold is out of range.")

    # input sequences
    seq_ids = []
    seqs = []
    input_file = SeqIO.parse(args.input, 'fasta')
    for seq in input_file:
        seq_ids.append(seq.id)
        seqs.append(str(seq.seq))
    print(seqs)
    if len(seq_ids) != len(seqs):
        raise ValueError("FASTA file is not valid.")

    # feature generation
    print("=====1. Loading pre-trained protein language model=====")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir=args.cache)
    pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=args.cache)


    def get_protein_features(seq):
        sequence_Example = ' '.join(seq)
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, return_tensors='pt')
        last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1, :]
        return last_hidden.detach()
    
    # generate sequence feature
    features = []
    print("=====2. Generating protein sequence feature=====")
    for s in seqs:
        features.append(get_protein_features(s).unsqueeze(0))

    # load CNN model
    print("=====3. Loading classification model=====")
    predictor = CNNOD()
    model_path = os.path.join(args.model, args.ligand + ".pth")
    print(f"Model loaded from {model_path}")
    predictor.load_state_dict(torch.load(model_path))


    # prediction process
    results = []
    print(f"=====4. Predicting {args.ligand}-binding sites=====")
    predictor.eval()
    for f in features:
        out = predictor(f).squeeze(0).detach().numpy()[:, 1]
        score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
        results.append(score)

    print(f"=====5. Writing result files into {args.output}=====")
    with open(args.output, 'w') as f:
        for i in range(len(seq_ids)):
            f.write('>'+seq_ids[i] + '\n')
            f.write(seqs[i] + '\n')
            f.write(results[i] + '\n')
    print(f"Congrats! All process done! Your result file is saved as {args.output}")
    
if __name__ == "__main__":
    main()