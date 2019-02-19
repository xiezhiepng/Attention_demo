# Attention_demo
## Introduction

Encoder-Decoder框架一个小demo

参考：https://github.com/Choco31415/Attention_Network_With_Keras



### Contents

以["six hours and fifty five am","06:55"]实例为例进行模型分析
问题定义：
将人类语言描述的时间，记为X；将标准数字描述的时间，
记为Y。即<X,Y>类型，符合Encoder-Decoder框架。 
X=["six hours and fifty five am"]，Y=["06:55"]    
任务：将X通过模型转换成Y



### prerequisites

- Python 3.5
- keras>=2.1.6
- numpy
- jupyter
- matplotlib
