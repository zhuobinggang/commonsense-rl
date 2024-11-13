# A clone from TextWorld Commonsense (TWC)

# 2024.8.9

对项目进行了重构，以降低每次重新操业的学习成本。
* 第一，将关于原生游戏环境的所有代码放入`env.py`，直接初始化Env对象，就可以调用act(command)方法，该方法相当于直接在命令行中输入指令command。
* 第二，将LLM调用的所有代码放入`llm_caller.py`。直接初始化GPT_Caller对象，就可以调用act_and_call(command)方法，该方法会自动调用env的act方法跟游戏环境交互，得到反馈之后，调用在初始化时候设定好的LLM模型。

# 2024.11.9 自动化agent的运行错误修正：

一般来说用以下命令可以启动自动化agent的进程

```py
env = Env(2, hard_test_index, 1, no_augment = False) # 2是hard level，1是test set
caller = Claude_Caller(env, zero_shot = True, gpt_type = 'claude-3-5-sonnet-20241022', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder1(), filename_prefix='B0')
caller.act_until_error(None)
```

但是运行可能会终止，终止的情况有两种，今天我们先解决无法从LLM的response text中提取command的问题。这种情况的话，一般会提示"BE CAREFULL!"。解决方法很简单，因为在询问LLM之前，必要信息已经存储在caller里边了，所以只要再调用`command = c.recall_and_get_command()`即可，然后我们使用`caller.act_until_error(command)`就可以重新启动自动化进程。

```py
command = c.recall_and_get_command()
caller.act_until_error(command)
```

# 打印记录

`c.log()`

# 2024.11.13 对于论文总体结构的重新思考以及实验

## 主要的目的

使用LLM来解决TWC问题。在其上提高模型的性能。

## Baseline Prompt

使用的信息和不基于llm的先行研究基本一致。

* action history
* inventory
* environment
* action list

在信息的简单罗列之上，我们的prompt还包含了两个额外的部分。

1. 添加于开头的对于任务的描述，这个符合LLM的使用实践。
2. 为了便于分析，我们要求模型输出思考过程。

该prompt作为我们手法的baseline。

## 我们的手法

1. 对于环境描述 + 选项列表的精简化。
2. 反馈增强，作为先行研究的成果被沿用。