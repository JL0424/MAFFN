# Multi-branch Attention Feature Fusion Network for Person Re-identification

## Overview
Current multi-branch methodologies for person re-identification primarily concentrate on extracting pedestrian features through a multi-branch architecture, often neglecting the correlation between different features. In this paper, we propose a Multi-Branch Attention Feature Fusion Network (MAFFN), which learns diverse feature representations through a multi-branch structure and employs a feature fusion mechanism to enhance the correlation and  obustness among features.  The MAFFN consists of three components: the attention branch, the global branch, and the local branch. The attention branch incorporates a Parallel Attention Module (PAM) designed to integrate channel attention and spatial attention, thereby extracting discriminative information related to pedestrians from multiple dimensions. The global branch is tasked with capturing overall features, while the local branch concentrates on acquiring fine-grained features. The proposed fusion strategy enhances the correlation between the global branch and the attention branch, significantly improving the discrimination and robustness of pedestrian features. 

The main contribution of this paper can be outlined as follows: 
<li>We propose a novel Multi-branch Attention Feature Fusion Network (MAFFN) is proposed, which extracts features at different scales through a multi-branch structure and improves robustness by fusing different branch features. </li>
<li>We design a Parallel Attention Module (PAM), which extracts the discriminative information of pedestrians through channel and spatial attention, and the parallel structure both improves the model efficiency and enhances the intercorrelation between features through fusion technique. </li>
<li>Experiments on the Market-1501, DukeMTMC-ReID, and CUHK03 datasets show that MAFFN outperforms the comparative methods and achieves the best level of performance in the three datasets tested. </li>

## Methods
### 1.Feature fusion between different branches
![](./image/MAFFN.jpg)
Inter-branch feature fusion makes full use of the complementary information of different branches by designing global branch, local branch and attention branch. The global branch extracts overall features through generalised average pooling, the local branch obtains fine-grained local information through horizontal chunking, and the attention branch extracts multi-dimensional attention features through PAM. Subsequently, the output of the attention branch is spliced with the global features in the channel dimension, and the interaction and correlation of the features are enhanced by the convolutional dimensionality reduction operation. Eventually, the fused features are combined with the fine-grained features extracted by the local branch to form comprehensive discriminative features for the pedestrian re-identification task, which significantly improves the robustness and adaptability of the model in complex scenes.The implementation of the method can be found in the file `modelling/MAFFN.py`.

### 2.Parallel Attention Module
![](./image/PAM.jpg)
PAM extracts multidimensional discriminative information of the feature map through channel attention and spatial attention mechanisms to enhance the robustness of pedestrian features. The channel attention mechanism generates global feature descriptors by average pooling and maximum pooling of the input feature map, which are then multiplied element-by-element with the original features after generating channel weights via the shared fully-connected layer to enhance important channel information. The spatial attention mechanism, on the other hand, generates spatial feature descriptors through average pooling and maximum pooling of channel dimensions, and inputs the two into the convolutional layer to generate spatial weights after splicing, which are then applied to the feature map to highlight key location features. Ultimately, the channel and spatial attention features are summed with the original features to obtain the fused attention features to improve the discriminative power and efficiency of the model.The implementation of the method can be found in the file `modelling/layer/attention.py`.

## Datasets: 
All three datasets are publicly available and download links have been provided to access the datasets by clicking on the corresponding dataset.The dataset used for the experiment is as follows：

[Market1501](https://www.kaggle.com/datasets/sachinsarkar/market1501)

[DukeMTMC-ReID](https://www.kaggle.com/datasets/whurobin/dukemtmcreid)

[CUHK03](https://www.kaggle.com/datasets/priyanagda/cuhk03)

## Test Result
The test results on the three datasets are shown in the table below:
|    Dataset     | Top-1 |  mAP  |
| :------------: | :---: | :---: |
|  Market-1501   | 96.00 | 89.60 |
| DukeMTMC-Re-ID | 90.10 | 80.60 |
|     CUHK03     | 81.20 | 78.30 |

## Quick Star
1.Prepare dataset

Create a directory to store reid datasets under this repo, taking Market1501 for example
```
cd MAFFN
mkdir datasets
```
* Set _C.DATASETS.ROOT_DIR = (`./datasets`) in `config/defaults.py`
* Download dataset to datasets
* Extract dataset and rename to market1501. The data structure would like:
```
datasets
    market1501
        bounding_box_test/
        bounding_box_train/
        ......
```
The rest of the dataset operates as above

2.Install dependencies

Algorithm dependencies are installed as described in the `requirements.txt` file

3.Train and Test
Enter the following commands to train and test the model:
```
python tools/main.py --config_file='configs/experiment.yml' DATASETS.NAMES "('market1501')"
```
```
python tools/main.py --config_file='configs/experiment.yml' DATASETS.NAMES "('market1501')"  MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./pretrained/market1501_AGW.pth')" TEST.EVALUATE_ONLY "('on')"
```

### Please consult the author for code issues.
Contact: Z20231090862@stu.ncwu.edu.cn or HNYCJL@163.com or 1710715380@qq.com
