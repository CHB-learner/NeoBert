# NeoBert
Accurately predicting epitope and HLA binding remains a critical and challenging task in immunology, which prompted the development of many tools and models such as NetMHCpan. Very few current models took into consideration of noncanonical antigens, but our work thus far has revealed that distinct features exist between canonical and noncanonical neoantigens. Towards this end, we adopted the Bidirectional Encoder Representations from Transformers (BERT) model to predict the binding affinity of neoantigens and HLA, and named our new model NeoBERT.

NeoBERT is an advanced model designed specifically for predicting the binding affinity between neoantigens and HLA. It is a variant of the original BERT model, enhanced to integrate biological features with deep learning techniques. Unlike traditional tools, NeoBERT considers both canonical and noncanonical neoantigens, offering unique insights into the binding capabilities of these diverse antigens.


## Installation

### Install NeoBert in the virtual environment by conda.(Recommended)
#### Step 1: install conda: https://docs.anaconda.com/anaconda/install/index.html, and then create a envs named NeoBert with python 3.8

```
conda create -n NeoBert python=3.8 p
conda activate NeoBert
```

#### Step 2: install NeoBert from Github:

```
git clone https://github.com/CHB-learner/NeoBert.git
cd NeoBert
pip install -r requirements.txt
```

#### Step 3: download the BERT base model:

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/google-bert/bert-base-uncased
# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google-bert/bert-base-uncased
```

## Usage
### Step 1. Data Preprocessing
We first clean the combined MHC molecule sequences and peptide sequences by removing duplicates, errors, or missing pairs. Next, we sequentially move several characters from the end of the MHC sequences to the beginning to generate the negative samples required for training. We also ensure that these negative samples do not overlap with the positive samples in the original data.
After obtaining the negative samples, we will split the dataset, retaining a certain proportion of positive and negative samples for evaluation after training. The rest of the training data will be organized in the following format.
```
  {
      "label": "1",
      "text": "YFAMYGEKVAHTHVDTLYLRYHYYTWAVWAYTWY[SEP]QLNVVIHQL"
  }
```
The text composition consists of the MHC sequence + [SEP] + peptide sequence, while the label indicates whether the two sequences bind.<br>
All of the above operations are implemented in `process.py`.
```
python process.py
```

### Step 2. Train
Before starting the training, adjust the parameters in `train.py` to ensure proper initialization of the training process.
```
python train.py
```

### Step 3. Predict
We provide two inference modes: one for inputting multiple data entries via a file, and the other for specifying two particular sequences.
```
python infer.py
```
Parameter Description:<br>
--input_mode: 0 for file input, 1 for specific HLA and peptide sequences (required parameter).<br>
--model_path: Path to load model weights (required parameter).<br>

#### input_mode: 0
--input_file: The input file in CSV or XLSX format, with the output in the same format. See the example provided in example.xlsx.<br>
--save_dir: Directory to save prediction results. If it does not exist, it will be automatically created.<br>
Example:
```bash
python infer.py --input_mode 0 --model_path ./model_weight/checkpoint/ --save_dir ./predictout/ --input_file ./XXXX.csv
```

#### input_mode: 1
--Peptide: A peptide sequence, e.g., EISGGAMAVVAVFLNNKLYVANVGT.<br>
--HLA: An HLA sequence, e.g., YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY.<br>
Example:
```bash
python infer.py --input_mode 1 --model_path ./model_weight/checkpoint/ --Peptide EISGGAMAVVAVFLNNKLYVANVGT --HLA YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY
```


### Step 4. Eval
```
python eval.py
```

## References
* Bert: https://huggingface.co/google-bert/bert-base-uncased
* Transformers: https://github.com/huggingface/transformers


## Citation
NeoAtlas and NeoBert: A Database and A Predictive Model for Canonical and Noncanonical Tumor Neoantigens.2024. (submitted).
