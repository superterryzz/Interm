解析：无监督特征降维
====================

>先以二分类的数据进行特征选择:
    将样本划分 0标签样本为一类样本 非0标签样本为另一类样本
-----------------------------
> 主成分分析降维 PCA
>   PCA
>>SparsePCA 是PCA的变体（the goal of extracting the set of sparse components that best reconstruct the data.）
>>>Mini-batch sparse PCA (MiniBatchSparsePCA) 是SparsePCA的变体 它速度更快，但是结果精确程度不如SparsePCA
****
--------------------------
> RandomForest随机森林的无监督版本——随机树随机投影（Random Trees Embedding）来估计特征的重要性
    思路：让41个特征分别做一次随机森林分类，得到41个特征的重要性，然后根据重要性排序，得出重要特征的排序
-------------------------------
> 过滤式特征选择  Relief算法（二分类） Relief-F（多分类
    Relief 给出特征值的权重，权重越大，该特征越重要。但现在特征值的权重的绝对值都大于100
------------------------------
> LDA线性判别分析法  有监督学习 不适用


-----------------------------------
> 多因素分析  Multiple factor analysis  
> 1. 通过PCA降维，将41个特征降到2维，然后对这两个维度进行多因素分析
>  