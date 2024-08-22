**If you have any questions regarding the code/data, please contact Yufan Liu via andyalbert97@gmail.com.**

# CLAPE framework

This repo holds the code of CLAPE (Contrastive Learning And Pre-trained Encoder) framework for protein-ligands binding sites prediction. We provide 3 ligand-binding tasks including protein-DNA, protein-RNA, and antibody-antigen binding sites prediction, an we will also provide small molecules binding sites weight in the future (check [CLAPE-SMB](https://github.com/JueWangTHU/CLAPE-SMB) for reference).


## Usage

CLAPE is primarily dependent on a large-scale pre-trained protein language model [ProtBert](https://huggingface.co/Rostlab/prot_bert)  implemented using [HuggingFace's Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance, or create a conda/mamba envrionment using provided environment file. If you are using CLAPE-SMB, please install [ESM](https://github.com/facebookresearch/esm).

```shell
wget https://github.com/YAndrewL/CLAPE/blob/main/environment.yaml
conda env create -f environment.yaml
conda activate clape 
```
### 1. Python package from pypi
We provide a python package for predicting ligand-binding sites of given protein sequences in FASTA format. Here we provide a sample file, and please use CLAPE as following steps, taking DNA-binding sites prediction as an example:

```shell 
# download model weights and example file
wget https://github.com/YAndrewL/CLAPE/blob/main/example.fa
wget https://github.com/YAndrewL/CLAPE/blob/main/weights/DNA.pth
pip install clape  # install clape from pypi
```

```python
# package usage example
from clape import Clape

model = Clape(model_path="model_path", ligand="DNA")
results = model.predict(input_file="example.fa")
```
You can set `keep_score` to `True` to keep the predicted score from model, and use `switch_ligand` to change to another binding site prediction task.


### 2. Command line tools
We also provide a command line tool, which will be installed along the python package, you may use as below:

```shell
clape --input example.fa --output out.txt --ligand DNA --model /path/to/downloaded/model
```

This command will first load the pre-trained models, users can specify the downloading directory using the `--cache` parameter.

Some parameters are described as follows:

| Parameters  | Descriptions                                                 |
| ----------- | ------------------------------------------------------------ |
| --help      | Show the help doc.                                           |
| --ligand    | Specify the ligand for prediction, DNA, RNA, and AB (antibody) are supported now. |
| --threshold | Specify the threshold for identifying the binding site, the value needs to be between 0 and 1, default: 0.5. |
| --input     | The path of the input file in FASTA format.                  |
| --output    | The path of the output file, the first and the second line are the same as the input file, and the third line is the prediction result. |
| --cache     | The path for saving the pre-trained parameters, default: protbert. |
| --model     | The path for trained backbone models.|

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
- [Aug. 2024] CLAPE can be used as a python package now, please check [clape in pypi](https://pypi.org/project/clape/).

- [Mar. 2024] The training code is released with CLAPE-SMB, please check [this repo](https://github.com/JueWangTHU/CLAPE-SMB) for reference.

- [Jan. 2024] Our paper is publised in Briefings in Bioinformatics, please check [the online version](https://academic.oup.com/bib/article/25/1/bbad488/7505238).
