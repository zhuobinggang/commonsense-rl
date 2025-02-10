2025.2.10 写入

虽然结果不错（分数为0.576左右），但是最近的先行研究调查结果显示，目前FTWP上最好的性能是ledeepchef，这个模型的分数为（0.693）。ledeepchef是基于强化学习的模型，针对FTWP做了大量优化，因此分数比不过它也是正常的，但是我们的模型有两个优点：第一是更加简单，使得它很容易适用于其他数据集，比如TWC，而LeDeepChef只在FTWP上进行了实验；第二是比起强化学习模型训练效率更高，因为我们只使用了基于监督学习的行为模仿，以及事先学习的transformer模型，不需要探索环境，因此训练的效率更高。

注意：和LeDeepChef不同，我们并不执着于不使用指令列表，这大幅降低了任务的难度。

具体内容参照： [谷歌文档](https://docs.google.com/document/d/1gu0MZIQdj__4vCKdOZATsq9HsnvOSsJx9yvb9Cgzcp4/edit?tab=t.0)