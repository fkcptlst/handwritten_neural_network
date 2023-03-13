## 1. 如何评测

（在有mnist_test.csv的前提下）在eval.py目录下运行python eval.py即可。输出结果即为最终的评测结果（测试集准确率）。

评测函数在common.py中实现，与提供的框架代码完全一致。

本地评测结果：performance = 0.9883

## 2. 介绍

新的框架参考了Pytorch的Api设计思想，将不同层模块化解耦合，使各个模块能灵活排列组合。

底层完全由numpy实现。主要模块有：

- Linear：全连接层
- ReLU：ReLU激活函数
- Sigmoid：Sigmoid激活函数
- LeakyReLU：LeakyReLU激活函数
- MSE: 均方误差损失函数
- SGD：随机梯度下降优化器（带动量）

## 3.如何训练
1. 下载mnist数据集，解压在当前目录下
2. 运行`preprocess.py`，得到csv格式数据集
3. 运行`train_homework.py`开始训练。训练超参数在代码中进行定义。
