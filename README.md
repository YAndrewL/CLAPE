**If you have any questions regarding the code/data, please contact Yufan Liu via andyalbert97@gmail.com.**

# Repo for CLAPE framework

This repo holds the code of CLAPE (Contrastive Learning And Pre-trained Encoder) framework for protein-ligands binding sites prediction. We provide 3 ligand-binding tasks including protein-DNA, protein-RNA, and antibody-antigen binding sites prediction. 

CLAPE is primarily dependent on a large-scale pre-trained protein language model [ProtBert](https://huggingface.co/Rostlab/prot_bert)  implemented using [HuggingFace's Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance. 

## Usage

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

## Citation
If you find our work helpful, please kindly cite the BibTex as following:
```
@article{10.1093/bib/bbad488,
    author = {Liu, Yufan and Tian, Boxue},
    title = "{Protein–DNA binding sites prediction based on pre-trained protein language model and contrastive learning}",
    journal = {Briefings in Bioinformatics},
    volume = {25},
    number = {1},
    pages = {bbad488},
    year = {2024},
    month = {01},
    abstract = "{Protein–DNA interaction is critical for life activities such as replication, transcription and splicing. Identifying protein–DNA binding residues is essential for modeling their interaction and downstream studies. However, developing accurate and efficient computational methods for this task remains challenging. Improvements in this area have the potential to drive novel applications in biotechnology and drug design. In this study, we propose a novel approach called Contrastive Learning And Pre-trained Encoder (CLAPE), which combines a pre-trained protein language model and the contrastive learning method to predict DNA binding residues. We trained the CLAPE-DB model on the protein–DNA binding sites dataset and evaluated the model performance and generalization ability through various experiments. The results showed that the area under ROC curve values of the CLAPE-DB model on the two benchmark datasets reached 0.871 and 0.881, respectively, indicating superior performance compared to other existing models. CLAPE-DB showed better generalization ability and was specific to DNA-binding sites. In addition, we trained CLAPE on different protein–ligand binding sites datasets, demonstrating that CLAPE is a general framework for binding sites prediction. To facilitate the scientific community, the benchmark datasets and codes are freely available at https://github.com/YAndrewL/clape.}",
    issn = {1477-4054},
    doi = {10.1093/bib/bbad488},
    url = {https://doi.org/10.1093/bib/bbad488},
    eprint = {https://academic.oup.com/bib/article-pdf/25/1/bbad488/55381199/bbad488.pdf},
}
```

## Update
- [Mar. 2024] The training code is released with CLAPE-SMB, please check [this repo](https://github.com/JueWangTHU/CLAPE-SMB) for reference.

- [Jan. 2024] Our paper is publised in Briefings in Bioinformatics, please check [the online version](https://academic.oup.com/bib/article/25/1/bbad488/7505238).

