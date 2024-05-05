**If you have any questions regarding the code/data, please contact Yufan Liu via andyalbert97@gmail.com. 
**

# Repo for CLAPE framework

This repo holds the code of CLAPE (Contrastive Learning And Pre-trained Encoder) framework for protein-ligands binding sites prediction. We provide 3 ligand-binding tasks including protein-DNA, protein-RNA, and antibody-antigen binding sites prediction. 

CLAPE is primarily dependent on a large-scale pre-trained protein language model [ProtBert](https://huggingface.co/Rostlab/prot_bert)  implemented using [HuggingFace's Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance. 

## Usage:

We provide the Python script for predicting ligand-binding sites of given protein sequences in FASTA format. Here we provide a sample file, and please use CLAPE as following commands:

```
python clape.py --input example.fa --output out.txt
```

This command will first load the pre-trained models, users can specify the downloading directory using the `--cache` parameter.

Some parameters are described as follows:

| Parameters  | Descriptions                                                 |
| ----------- | ------------------------------------------------------------ |
| --help      | Show the help doc.                                           |
| --ligand    | Specify the ligand for prediction, DNA, RNA, and AB (antibody) are supported now, default: DNA. |
| --threshold | Specify the threshold for identifying the binding site, the value needs to be between 0 and 1, default: 0.5. |
| --input     | The path of the input file in FASTA format.                  |
| --output    | The path of the output file, the first and the second line are the same as the input file, and the third line is the prediction result. |
| --cache     | The path for saving the pre-trained parameters, default: protbert. |

Reference: 

[Protein-DNA binding sites prediction based on pre-trained protein language model and contrastive learning](https://academic.oup.com/bib/article/25/1/bbad488/7505238) by Yufan Liu and Boxue Tian. Published in Briefings in Bioinformatics.

Update: The training code is released with CLAPE-SMB, please check [the repo](https://github.com/JueWangTHU/CLAPE-SMB) for reference.


