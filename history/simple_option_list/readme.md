## 2024.8.12 实验汇总

重复5次得到的实验结果： baseline: 0.62

简化action list的组: 0.72

但是t检定结果只有0.07。因此第二步改进也要安排一下：简化环境描述。


## 详情

将action list替换成以下：

```
Action you can take:
1. take [object]
2. put [object in inventory] on [supporter]
3. insert [object in inventory] into [container]
4. go west
```

其中第四点根据当前情况进行更改，其他保持不变。

该变化能够减少噪音输入。对比下面原来的action list可以看出这一点：

```
Action you can take:
* close cutlery drawer
* close dishwasher
* close fridge
* close kitchen cupboard
* close oven
* close sliding patio door
* close trash can
* drop wet azure skirt
* examine counter
* examine cutlery drawer
* examine dining chair
* examine dining table
* examine dirty cake slice
* examine dirty pot
* examine dishwasher
* examine fridge
* examine kitchen cupboard
* examine ladderback chair
* examine milk
* examine oven
* examine stove
* examine trash can
* go east
* insert wet azure skirt into cutlery drawer
* insert wet azure skirt into dishwasher
* insert wet azure skirt into fridge
* insert wet azure skirt into kitchen cupboard
* insert wet azure skirt into oven
* insert wet azure skirt into trash can
* look
* put wet azure skirt on counter
* put wet azure skirt on dining chair
* put wet azure skirt on dining table
* put wet azure skirt on ladderback chair
* put wet azure skirt on stove
* take dirty cake slice from dishwasher
* take dirty pot from dishwasher
* take milk from fridge
```
