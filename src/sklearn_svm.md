# svm 函数接口
```python
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, hrinking=True, 				probability=False, tol=0.001, cache_size=200, class_weight=None, 				verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
```

# svm 参数解释
|                            参数名                            |                             解释                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|               **C** （float参数 默认值为1.0）                | 表示错误项的惩罚系数C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低；相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。 |
|              **kernel** （str参数 默认为‘rbf’）              | 该参数用于选择模型所使用的核函数，算法中常用的核函数有：  -- linear：线性核函数  --  poly：多项式核函数  --rbf：径像核函数/高斯核  --sigmod：sigmod核函数  --precomputed：核矩阵，该矩阵表示自己事先计算好的，输入后算法内部将使用你提供的矩阵进行计算 |
|               **degree** （int型参数 默认为3）               | 该参数只对'kernel=poly'(多项式核函数)有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。 |
|              **gamma** （float参数 默认为auto）              | 该参数为核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。如果gamma设置为auto，代表其值为样本特征数的倒数，即1/n_features，也有其他值可设定。 |
|              **coef0**:（float参数 默认为0.0）               | 该参数表示核函数中的独立项，只有对‘poly’和‘sigmod’核函数有用，是指其中的参数c。 |
|           **probability**（ bool参数 默认为False）           | 该参数表示是否启用概率估计。 这必须在调用fit()之前启用，并且会使fit()方法速度变慢。 |
| **shrinkintol: float参数 默认为1e^-3g**（bool参数 默认为True） |              该参数表示是否选用启发式收缩方式。              |
|              **tol**（ float参数 默认为1e^-3）               |              svm停止训练的误差精度，也即阈值。               |
|            **cache_size**（float参数 默认为200）             |  该参数表示指定训练所需要的内存，以MB为单位，默认为200MB。   |
| **class_weight**（字典类型或者‘balance’字符串。默认为None）  | 该参数表示给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。如果给定参数‘balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。 |
|            **verbose** （ bool参数 默认为False）             | 该参数表示是否启用详细输出。此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。 |
|              **max_iter** （int参数 默认为-1）               |     该参数表示最大迭代次数，如果设置为-1则表示不受限制。     |
| **random_state**（int，RandomState instance ，None 默认为None） | 该参数表示在混洗数据时所使用的伪随机数发生器的种子，如果选int，则为随机数生成器种子；如果选RandomState instance，则为随机数生成器；如果选None,则随机数生成器使用的是np.random。 |


### --
|   type    | Linear Kernel | Poly Kernel | RBF Kernel | Sigmoid Kernel |
| :-------: | ------------- | ----------- | ---------- | :------------: |
| precision |               |             |            |                |
|  recall   |               |             |            |                |
| f1-score  |               |             |            |                |


