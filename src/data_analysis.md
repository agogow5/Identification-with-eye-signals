# 柱状图分析
![](D:\jpt_path\Biometric\data_analysis_md\attribute_histogram_plots.png)


# 相关性分析
![](D:\jpt_path\Biometric\data_analysis_md\corr_matrix.png)

![](D:\jpt_path\Biometric\data_analysis_md\correlation_plots.png)

# 测试集与训练集制作
根据Label的比例进行分层抽取
在测试集中设置 k-fold-cv 这里用 7

# confusion_matrix
## 训练集

![](D:\jpt_path\Biometric\data_analysis_md\train_confusion_matrix.png)

## 测试集

![](D:\jpt_path\Biometric\data_analysis_md\test_confusion_matrix.png)

# precision 与 recall
## 训练集

![](D:\jpt_path\Biometric\data_analysis_md\train_pre_rec.png)

## 测试集

![](D:\jpt_path\Biometric\data_analysis_md\test_pre_rec.png)

# MLP Learning Curve

![](D:\jpt_path\Biometric\data_analysis_md\mlp_learning_curve.png)

在加入 l2 正则化之后，过拟合情况有所缓解。