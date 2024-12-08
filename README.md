# Multi-branch Attention Feature Fusion Network for Person Re-identification

## The innovations of this paper include the following:
<li>We propose a novel Multi-branch Attention Feature Fusion Network (MAFFN) is proposed, which extracts features at different scales through a multi-branch structure and improves robustness by fusing different branch features. </li>
<li>We design a Parallel Attention Module (PAM), which extracts the discriminative information of pedestrians through channel and spatial attention, and the parallel structure both improves the model efficiency and enhances the intercorrelation between features through fusion technique. </li>
<li>Experiments on the Market-1501, DukeMTMC-ReID, and CUHK03 datasets show that MAFFN outperforms the comparative methods and achieves the best level of performance in the three datasets tested. </li>

## Description of the key algorithms
### 1.Feature fusion between different branches
![](./image/network.jpg)
MAFFN adopts a multi-branch structure of global branching, attention branching and local branching to achieve multi-level feature extraction. Among them, the global branch extracts global features through the generalised mean pooling layer (GMP) to provide holistic information; the attention branch uses the PAM to focus on locally salient regions to obtain fine discriminative features; and the local branch captures fine-grained regional features through horizontal chunking cuts. The inter-branch feature fusion mechanism combines the advantages of the global branch and the attention branch, and enhances the information interaction between different features through feature splicing and dimensionality reduction operations during fusion, so that the final extracted pedestrian features are more robust and discriminative in complex scenarios.

implement：

### 2.Parallel Attention Module
![](./image/PAM.jpg)
In MAFFN (Multi-branch Attention Feature Fusion Network), PAM is integrated into the attention branch for enhancing the discriminative power of local features. Specifically, PAM extracts different dimensional information of the feature map through the channel attention module and the spatial attention module, and then fuses these attention features to form the final attention features by combining the original features. These features improve the overall feature discrimination and robustness in the subsequent feature fusion, and at the same time, due to its parallel structure design, it is able to efficiently utilise the parallel computing power of GPUs, further improving the processing speed of the model.

implement：

## Datasets: 
The dataset used for the experiment is as follows：

[Market1501](https://github.com/sybernix/market1501)

[DukeMTMC-ReID](https://gitcode.com/Resource-Bundle-Collection/f7e06/?utm_source=pan_gitcode&index=top&type=card&)

[CUHK03](https://aistudio.baidu.com/datasetdetail/86044/0)


## Quick Star
1.Prepare dataset

Create a directory to store reid datasets under this repo, taking Market1501 for example
```
cd MAFFN
mkdir datasets
```
* Set _C.DATASETS.ROOT_DIR = ('./datasets') in config/defaults.py
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

Algorithm dependencies are installed as described in the requirements.txt file

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
