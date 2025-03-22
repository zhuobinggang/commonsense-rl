## 2025.3.18 说明

为了对齐Ledeepchef对prompt和walkthrough进行更改，因此需要重新对模型进行训练

1) 更改prompt常年显示inventory
2) 训练时候对于每个walkthrough，我们筛掉inventory命令


## 2025.3.19 说明

两个epoch训练发现模型之间的性能差异太大（0.77和0.65），将epoch数提高到3，并且将step_interval（相当于批次）提高到8，并且重新训练。

1) 将epoch数提高到3
2) 将step_interval（相当于批次）提高到8