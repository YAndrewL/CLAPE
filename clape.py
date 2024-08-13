# -*- coding: utf-8 -*-
# @Time         : 2023/5/10 15:28
# @Author       : Yufan Liu
# @Description  : predicting process


from transformers import BertModel, BertTokenizer
import re
import torch.nn as nn
import torch
import argparse


# 1DCNN definition
class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)
        self.head = nn.Softmax(-1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x)


# load instructions
parse = argparse.ArgumentParser()
parse.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA, RNA, and antibody',
                   default='DNA', choices=['DNA', 'RNA', 'AB'])
parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parse.add_argument('--input', '-i', help='Input protein sequences in FASTA format', required=True)
parse.add_argument('--output', '-o', help='Output file path, default clape_result.txt',
                   default='clape_result.txt')
parse.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')

args = parse.parse_args()

# parameter judge
if args.threshold > 1 or args.threshold < 0:
    raise ValueError("Threshold is out of range.")

# input sequences
input_file = open(args.input, 'r').readlines()
seq_ids = []
seqs = []
for line in input_file:
    if line.startswith('>'):
        seq_ids.append(line.strip())
    else:
        seqs.append(line.strip())
if len(seq_ids) != len(seqs):
    raise ValueError("FASTA file is not valid.")

# feature generation
print("=====Loading pre-trained protein language model=====")
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
print("=====Generating protein sequence feature=====")
for s in seqs:
    features.append(get_protein_features(s).unsqueeze(0))

# load CNN model
print("=====Loading classification model=====")
predictor = CNNOD()
if args.ligand == 'DNA':
    predictor.load_state_dict(torch.load("./weights/DNA.pth"))
elif args.ligand == 'RNA':
    predictor.load_state_dict(torch.load("./weights/RNA.pth"))
elif args.ligand == 'AB':
    predictor.load_state_dict(torch.load("./weights/AB.pth"))
else:
    raise ValueError(args.ligand)

# prediction process
results = []
print(f"=====Predicting {args.ligand}-binding sites=====")
predictor.eval()
for f in features:
    out = predictor(f).squeeze(0).detach().numpy()[:, 1]
    score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
    results.append(score)

print(f"=====Writing result files into {args.output}=====")
with open(args.output, 'w') as f:
    for i in range(len(seq_ids)):
        f.write(seq_ids[i] + '\n')
        f.write(seqs[i] + '\n')
        f.write(results[i] + '\n')
print(f"Congrats! All process done! Your result file is saved as {args.output}")
