# Multi-branch Attention Feature Fusion Network for Person Re-identification

## The innovations of this paper include the following:
<li>We propose a novel Multi-branch Attention Feature Fusion Network (MAFFN) is proposed, which extracts features at different scales through a multi-branch structure and improves robustness by fusing different branch features. </li>
<li>We design a Parallel Attention Module (PAM), which extracts the discriminative information of pedestrians through channel and spatial attention, and the parallel structure both improves the model efficiency and enhances the intercorrelation between features through fusion technique. </li>
<li>Experiments on the Market-1501, DukeMTMC-ReID, and CUHK03 datasets show that MAFFN outperforms the comparative methods and achieves the best level of performance in the three datasets tested. </li>

## Description of the key algorithms
### 1.Feature fusion between different branches
![](./image/network.jpg)
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
<div>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=558
 style='width:418.15pt;border-collapse:collapse;border:none'>
 <thead>
  <tr>
   <td width=132 rowspan=2 style='width:99.2pt;border-top:solid windowtext 1.0pt;
   border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>Methods</span></p>
   </td>
   <td width=142 colspan=2 style='width:106.35pt;border-top:solid windowtext 1.0pt;
   border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>Market-1501</span></p>
   </td>
   <td width=142 colspan=2 style='width:106.3pt;border-top:solid windowtext 1.0pt;
   border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>DukeMTMC-ReID</span></p>
   </td>
   <td width=142 colspan=2 style='width:106.3pt;border-top:solid windowtext 1.0pt;
   border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>CUHK03</span></p>
   </td>
  </tr>
  <tr>
   <td width=71 style='width:55.15pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>Rank-1</span></p>
   </td>
   <td width=71 style='width:53.2pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>mAP</span></p>
   </td>
   <td width=71 style='width:55.15pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>Rank-1</span></p>
   </td>
   <td width=71 style='width:55.15pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>mAP</span></p>
   </td>
   <td width=71 style='width:55.15pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>Rank-1</span></p>
   </td>
   <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
   padding:0cm 5.4pt 0cm 5.4pt'>
   <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
   style='font-family:"Times New Roman",serif'>mAP</span></p>
   </td>
  </tr>
 </thead>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>PCB</span></p>
  </td>
  <td width=71 style='width:55.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>92.3</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>77.4</span></p>
  </td>
  <td width=71 style='width:55.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>81.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>66.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>59.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>53.2</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>PCB+RPP</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>93.8</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>81.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>83.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>69.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>62.8</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>52.7</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MGN</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.7</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>86.9</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>88.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>78.4</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>66.8</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>66.0</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>AGW</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.1</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>87.8</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.0</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>79.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>63.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>62.0</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MMHPN</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>94.6</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>83.4</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>87.8</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>75.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>68.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>65.4</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>HBMCN</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>94.4</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>85.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>85.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>74.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>73.8</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>69.0</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>AWPCN</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>94.0</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>82.0</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>85.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>74.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>67.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>62.8</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>CadNet</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>94.6</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>85.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>86.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>72.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>SFL-MFF</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.7</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>88.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>90.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>80.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>79.0</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>75.8</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MSDENet</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.8</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>78.4</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>75.7</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>AAformer</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.4</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>88.0</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>90.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>80.9</span></b></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>78.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>77.2</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>DFFRRID</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.3</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>88.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>78.9</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>80.5</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>73.9</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MCFR</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>93.7</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>82.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>84.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>72.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MBA-Net</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.7</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>79.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>PSF-C-Net</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.2</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>87.3</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>87.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>76.9</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MHDNet</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>94.6</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>87.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>79.7</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>INMM</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>95.7</span></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>88.1</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>89.2</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>79.5</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>&nbsp;</span></p>
  </td>
 </tr>
 <tr>
  <td width=132 style='width:99.2pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>MAFFN (Ours)</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>96.0</span></b></p>
  </td>
  <td width=71 style='width:53.2pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>89.6</span></b></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>90.1</span></b></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>80.6</span></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>81.2</span></b></p>
  </td>
  <td width=71 style='width:53.15pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-family:"Times New Roman",serif'>78.3</span></b></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US>&nbsp;</span></p>

</div>

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
