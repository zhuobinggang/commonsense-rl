# 该文件夹用于记录2025.1.20的实验结果

模型说明：使用BERT进行行为模仿。训练调用bert_training_data.exp_and_shutdown()。结果获取调用bert_rl.final_test()。基本特征是不使用desc，仅使用可用行动列表。

## 实验： 首先确定了最佳epoch数 = 2 (一个epoch大概2个小时)

模型在epoch3的时候达到最佳检验分数0.8596491228070176，epoch2达到0.847457627118644，epoch1达到0.813953488372093。

可以发现epoch2的提高是相当明显的大概有0.03左右，但是2到3并不明显，所以epoch2应该被视为训练终点。

因此实验应该继续进行。

## epoch 1 的结果

test temp: 0.6868568839829065
test real: 0.4466036887089519

## epoch 2 (2025.1.21)

test temp: 0.814
test real: 0.566



