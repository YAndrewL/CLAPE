# CLAPE Framework

**Contact:** For any questions regarding the code or data, please contact Yufan Liu at andyalbert97@gmail.com.

This repository contains the code for the CLAPE (Contrastive Learning And Pre-trained Encoder) framework for protein–ligand binding site prediction. We provide three ligand-binding tasks: protein–DNA, protein–RNA, and antibody–antigen binding site prediction. For support of small molecule binding site weights and training scripts for the CLAPE series of models, please refer to  [CLAPE-SMB](https://github.com/JueWangTHU/CLAPE-SMB).

---

## Usage

**Note:** For comprehensive usage instructions, please refer to our [CLAPE protocol](https://link.springer.com/protocol/10.1007/978-1-0716-4623-6_18) in Methods in Molecular Biology.

CLAPE primarily depends on the large-scale pre-trained protein language model [ProtBert](https://huggingface.co/Rostlab/prot_bert), implemented using [HuggingFace Transformers](https://huggingface.co/) and [PyTorch](https://pytorch.org/). Please install the dependencies in advance, or create a conda/mamba environment using the provided environment file. If you are using CLAPE-SMB, please also install [ESM](https://github.com/facebookresearch/esm).

```shell
wget https://github.com/YAndrewL/CLAPE/blob/main/environment.yaml
conda env create -f environment.yaml
conda activate clape 
```

### 1. Python Package from PyPI

We provide a Python package for predicting ligand-binding sites from protein sequences in FASTA format. Below is an example for DNA-binding site prediction:

```shell
# Download model weights and example file
wget https://github.com/YAndrewL/CLAPE/blob/main/example.fa
wget https://github.com/YAndrewL/CLAPE/blob/main/weights/DNA.pth
pip install clape  # Install CLAPE from PyPI
```

```python
# Example usage
from clape import Clape

model = Clape(model_path="model_path", ligand="DNA")
results = model.predict(input_file="example.fa")
```

You can set `keep_score=True` to retain the predicted scores from the model, and use `switch_ligand` to change the binding site prediction task.

### 2. Command Line Tools

A command line tool is also provided and installed with the Python package:

```shell
clape --input example.fa --output out.txt --ligand DNA --model /path/to/downloaded/model
```

This command loads the pre-trained models. You can specify the download directory using the `--cache` parameter.

**Parameter descriptions:**

| Parameter   | Description                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------|
| --help      | Show the help documentation.                                                                     |
| --ligand    | Specify the ligand for prediction: DNA, RNA, or AB (antibody).                                  |
| --threshold | Threshold for identifying binding sites (0–1, default: 0.5).                                     |
| --input     | Path to the input file in FASTA format.                                                          |
| --output    | Path to the output file. The first two lines match the input; the third line is the prediction.  |
| --cache     | Path for saving pre-trained parameters (default: protbert).                                      |
| --model     | Path to trained backbone models.                                                                 |

---

## Citation

If you find our work helpful, please cite as follows:

**CLAPE:**
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
    issn = {1477-4054},
    doi = {10.1093/bib/bbad488},
    url = {https://doi.org/10.1093/bib/bbad488},
    eprint = {https://academic.oup.com/bib/article-pdf/25/1/bbad488/55381199/bbad488.pdf},
}
```

**CLAPE-SMB:**
```
@article{10.1186/s13321-024-00920-2,
    author={Wang, Jue and Liu, Yufan and Tian, Boxue},
    title={Protein-small molecule binding site prediction based on a pre-trained protein language model with contrastive learning},
    journal={Journal of Cheminformatics},
    year={2024},
    month={Nov},
    day={06},
    volume={16},
    number={1},
    pages={125},
    issn={1758-2946},
    doi={10.1186/s13321-024-00920-2},
    url={https://doi.org/10.1186/s13321-024-00920-2}
}
```

**CLAPE protocol:**
```
@inbook{Liu2025,
  author    = {Yufan Liu and Boxue Tian},
  editor    = {Dukka B. KC},
  title     = {CLAPE: Protein--Ligand Binding Site Prediction via Protein Language Models},
  booktitle = {Large Language Models (LLMs) in Protein Bioinformatics},
  year      = {2025},
  publisher = {Springer US},
  address   = {New York, NY},
  pages     = {293--311},
  isbn      = {978-1-0716-4623-6},
  doi       = {10.1007/978-1-0716-4623-6_18},
  url       = {https://doi.org/10.1007/978-1-0716-4623-6_18}
}
```

---

## Updates

- **Jul. 2025:** CLAPE protocol published in Methods in Molecular Biology. See [the online version](https://link.springer.com/protocol/10.1007/978-1-0716-4623-6_18).
- **Nov. 2024:** CLAPE-SMB published in Journal of Cheminformatics. See [the online version](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00920-2).
- **Aug. 2024:** CLAPE is now available as a Python package. See [clape on PyPI](https://pypi.org/project/clape/).
- **Mar. 2024:** Training code released with CLAPE-SMB. See [this repo](https://github.com/JueWangTHU/CLAPE-SMB) for reference.
- **Jan. 2024:** Our paper published in Briefings in Bioinformatics. See [the online version](https://academic.oup.com/bib/article/25/1/bbad488/7505238).
