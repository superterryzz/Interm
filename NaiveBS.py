'''
coding:utf-8
@Software:PyCharm
@Time:2023/6/26 10:43
@Author:Super Cao
'''
# 通过特征工程构建联合特征,利用朴素贝叶斯选择1）关联性较强；2）不同类别舞弊手段下的较为显著的关联特征

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 非负离散特征引入伯努利、多项式朴素贝叶斯以及CategoricalNB
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.model_selection import train_test_split

# raw_data_df = pd.read_csv(r"all_samples.txt", sep=',')
raw_data_df = pd.read_excel(r"all_samples_clean.xlsx")
feature_code_map_df = pd.read_excel(r"all_samples_clean.xlsx", sheet_name="编号映射")
feature_code_map = feature_code_map_df.set_index(['特征名称'])["特征编号"].to_dict()


# 当前共有41个离散（假设为独立）特征
# 其中0代表正常,1代表普通触发,2代表高危触发

def visualization(my_feature_prob, my_feature_names):
    """
    根据P(x_i|y)排序并绘制得分柱状图
    :param my_feature_prob:以字典形式存储构造出来的【联合特征】发生的似然概率,键作为可视化label
    :param my_feature_names:独立特征的名称,为可视化更加清晰,暂用对应编号表示
    :return:
    """
    fig = plt.figure(dpi=200)
    # 不同y【即不同的联合特征】类别当作一条折线,观察不同类别的趋势,【即重要性排列前位】
    for i in range(len(my_feature_prob)):
        label = list(my_feature_prob.keys())[i]
        this_feature_prob = dict(zip(my_feature_names, list(my_feature_prob.values())[i]))
        sort_this_feature_prob = sorted(this_feature_prob.items(), key=lambda x: x[1], reverse=True)  # 降序排列联合特征的似然概率
        plt.plot([x[1] for x in sort_this_feature_prob], label=str(label))
        plt.xticks(np.arange(len(my_feature_names)), [x[0] for x in sort_this_feature_prob], rotation=45,
                   fontsize=5,
                   position=(-7.5, 0))
    plt.legend()
    plt.show()


def bi_gram_joint_feature(my_raw_data_df, my_y_name):
    """
    采用bi_gram的思想进行2*C^2_(41)次比较
    该函数进行单次比较,旨在选出成对的联合特征
    计算P(y|x)
    :param my_raw_data_df:存储所有特征取值的df
    :param my_y_name:相当于target在raw_data_df中的colname
    :return:
    """
    y = my_raw_data_df[my_y_name].to_numpy()
    # x = my_raw_data_df.drop(['BASIC_entity_name', 'BASIC_year', my_y_name], axis=1).to_numpy()
    x = my_raw_data_df.drop(['BASIC_entity_name', 'BASIC_year', my_y_name], axis=1)
    # feature_names = my_raw_data_df.drop(['BASIC_entity_name', 'BASIC_year', my_y_name], axis=1).columns
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2023)
    cnb = CategoricalNB(alpha=1.0, fit_prior=True, min_categories=2)
    cnb.fit(x_train, y_train)
    # 观察不同类别的样本是否平衡
    class_count = cnb.class_count_
    class_prob = np.exp(cnb.class_log_prior_)
    print(f"全部样本的特征{my_y_name}频次分布为{class_count},概率分布为[" + ",".join(
        f"{cprob:.2f}" for cprob in class_prob) + "]")
    # 展示不同特征和目标特征的共同出现的概率,并据此选择组成联合特征
    feature_log_prob = cnb.feature_log_prob_
    feature_prob = [np.exp(x).tolist() for x in
                    feature_log_prob]  # shape:[feature_num*[class_of_y,category_num_of_feature_i]]
    my_feature_names = cnb.feature_names_in_
    feature_codes = [feature_code_map[name] for name in my_feature_names]
    # 选择class_of_y且category_of_feature_i均非零的似然概率进行,各存储为列表后进行排序
    abnormal_class_y = {}
    # 某些【特征】的某些维度可能缺失某些【属性】的样本
    num_class_y = max([np.shape(x)[0] for x in feature_prob])
    num_category_feature = max([np.shape(x)[1] for x in feature_prob])
    # 暂时考虑num_class_y*num_category_feature种情况下的似然概率
    this_key1 = None
    this_key2 = None
    for i in range(1, max(num_class_y, num_category_feature)):
        for j in range(1, max(num_class_y, num_category_feature)):
            if i == 1:  # 普通触发
                this_key1 = "Key1:Abnormal"
            elif i == 2:  # 高危触发
                this_key1 = "Key1:Severe_Abnormal"
            if j == 1:  # 自变量普通触发
                this_key2 = "Key2:Abnormal"
            elif j == 2:  # 自变量高危触发
                this_key2 = "Key2:Severe_Abnormal"
            this_key = (this_key1, this_key2)  # 作为【bi-gram联合特征】的名称【暂时】
            this_value_lst = []
            for m in range(len(my_feature_names)):
                this_shape = np.shape(feature_prob[m])
                # 判断是否某【特征】的某属性样本缺失,若发生维度溢出,则需要跳过并赋值为【0】
                if i + 1 > this_shape[0] or j + 1 > this_shape[1]:
                    this_value_lst.append(0)
                else:
                    temp_value = feature_prob[m][i][j]
                    this_value_lst.append(temp_value)
            abnormal_class_y[this_key] = this_value_lst
    # 观察朴素贝叶斯得分
    visualization(abnormal_class_y, feature_codes)
    # 观察得分
    x_predict = cnb.predict(x_test)
    score = cnb.score(x_test, y_test)
    print(f"The Score of Predict {col} is {score:.2f}")
    return abnormal_class_y, my_feature_names


if __name__ == "__main__":
    # 首先进行bi-gram在单个特征中进行成对评价
    bi_gram_feature_pair = {}  # 用于记录【似然概率】显著的【成对组合特征】
    for col in raw_data_df.columns[2:]:
        feature_prob_dict, feature_names = bi_gram_joint_feature(raw_data_df, col)
        # 对返回的特征按照似然概率进行显著性排序,选择总排序最大的
        sort_feature_prob_lst = []
        for i in range(len(feature_prob_dict)):
            this_feature_prob = dict(zip(feature_names, list(feature_prob_dict.values())[i]))
            sort_feature_prob_lst.append(
                sorted(this_feature_prob.items(), key=lambda x: x[1], reverse=True))
        this_bi_gram_feature_pair = sorted(
            [bi_gram_feature[0] for bi_gram_feature in sort_feature_prob_lst]
            , key=lambda x: x[1], reverse=True)[0]
        this_bi_gram_name = (col, this_bi_gram_feature_pair[0])
        bi_gram_feature_pair[this_bi_gram_name] = this_bi_gram_feature_pair[1]
    # 将筛选出的【成对新特征】记录为DataFrame
    feature1 = [x[0][0] for x in list(bi_gram_feature_pair.items())]
    feature2 = [x[0][1] for x in list(bi_gram_feature_pair.items())]
    value = [x[1] for x in list(bi_gram_feature_pair.items())]
    # 写入xlsx
    bi_gram_feature_df = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2, 'Value': value})
    bi_gram_feature_df.to_excel(r"bi_gram_feature.xlsx", index=False, header=True)
