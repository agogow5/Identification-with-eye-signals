# Hough  transform

|论文名称|页码 +位置|注释|
|-|-|-|
|虹膜定位及识别算法研究|P29|公式插入|



# 边缘检测

## [三种方法][https://www.jianshu.com/p/2334bee37de5]

### Canny
[canny算法原理](https://blog.csdn.net/jialeheyeshu/article/details/49129741)

根据二维灰度矩阵梯度向量来寻找图像灰度矩阵的灰度跃变位置，然后在图像中将这些位置的点连起来就构成了所谓的图像边缘

### Sobel

## 注意事项
在Hough Transform 之前得先手工进行 Canny 检测
如果跳过这一步，单纯依赖Hough Transform中的Canny会产生不一样的结果。

![](D:\jpt_path\Biometric\data_analysis_md\Canny+HT_01.png)

上图为原始的图像



![](D:\jpt_path\Biometric\data_analysis_md\Canny+HT_02.png)

跳过Canny直接送入HT，由于没有高斯平滑处理，噪点过多，两个虹膜的半径差距比较大。

# 眼距计算
。。毫无思路

![](D:\jpt_path\Biometric\data_analysis_md\eye_distance.png)



corner 检测

https://blog.csdn.net/jwh_bupt/article/details/7628665