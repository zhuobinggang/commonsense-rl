## taku 2025.3.27

数据集路径： `/home/taku/Downloads/cog2019_ftwp/cogni_dataset/README.md`

最新生成的walkthrough_train.csv和walkthrough_valid.csv的command行有以下特征：

1. command行的指令是walkthrough首先经过我的过滤，过滤逻辑在common.py
2. 然后经过cogni agent的walkthrough逻辑修正： 在检查到cookbook之前，首先将房间内目力所及的门打开，应该是为了方便导航（可能需要检查该项操作对qa模型性能的影响）

`虽然经过这些过滤，但是train和valid数据集中所有游戏都能完成所有任务。`

生成数据集的代码在`bert_bc_dataset.py`。