# NeoBert
![提取自14893_0_merged_1724580574_00](https://github.com/user-attachments/assets/9fa18a5d-fc0a-46cd-9d2a-9278622083ce)
Accurately predicting epitope and HLA binding remains a critical and challenging task in immunology, which prompted the development of many tools and models such as NetMHCpan. Very few current models took into consideration of noncanonical antigens, but our work thus far has revealed that distinct features exist between canonical and noncanonical neoantigens. Towards this end, we adopted the Bidirectional Encoder Representations from Transformers (BERT) model to predict the binding affinity of neoantigens and HLA, and named our new model NeoBERT.

NeoBERT is an advanced model designed specifically for predicting the binding affinity between neoantigens and HLA. It is a variant of the original BERT model, enhanced to integrate biological features with deep learning techniques. Unlike traditional tools, NeoBERT considers both canonical and noncanonical neoantigens, offering unique insights into the binding capabilities of these diverse antigens.


## Installation-安装

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



## Usage-用法

*------ 备注：项目的入门使用方法 ，最好包括界面操作截图及视频*

## References

