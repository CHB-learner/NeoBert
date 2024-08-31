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

### Step 2. Train

### Step 3. Predict

### Step 4. Eval


## References
* Bert: https://huggingface.co/google-bert/bert-base-uncased
* Transformers: https://github.com/huggingface/transformers


## Citation
NeoAtlas and NeoBert: A Database and A Predictive Model for Canonical and Noncanonical Tumor Neoantigens.2024. (submitted).
