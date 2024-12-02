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

## 实验记录以及代码寻路

实验结果在`TWC新论文构思中文.doc`。

* gpt4o base -> `exp3.test_gpt4o_baseline`
* gpt4o desc + actions simplify -> `exp3.test_gpt4o_simplified`
* gpt4o desc simplify -> `exp5.test_gpt4o_desc_only_simplify`
* claude base -> `exp3.test_claude_baseline`
* claude desc + actions simplify -> `exp3.test_claude_simplified`
* 4omini desc simple smooth -> `exp6.test_4omini_desc_only_simplify_smooth`

# 2024.11.18 对于desc only simple的反思

因为gpt系列的模型对于desc only都会降低性能，必须反思其原因。大概原因在于过于简单化的描述导致一些判断所需的信息欠缺，比如说容器开启与否。为了保留足够的信息需要考虑更加详细一点的房间信息。

# 2024.11.19 将smooth desc作为新的基线，以后的所有模型基于此基线

同标题。

* Desc simple smooth (新的baseline) -> `exp6.test_gpt4o_desc_only_simplify_smooth`
* Desc simple smooth + another room info -> `exp6.test_gpt4o_desc_only_simplify_smooth_another_room`
* Claude -> `exp6.test_claude_desc_smooth` + `exp6.test_claude_desc_smooth_another_room`

# 2024.11.20 引入reflexion尝试提高4omini的性能

失败了疯狂的失败。4omini真的不行。

# 2024.11.26 微调4omini

分数为0.54。

为了微调4omini创建了`human_play.py`和`finetuned_play.py`。文件就是文件名所传达的意思。实验通过`finetuned_play.py`的`llm_auto_play`函数启动。

微调用的数据是我在5个train game上面游玩的结果，总共包含70条数据。文件存储在`exp/finetune_4omini/out.jsonl`。文件夹内还包含3个辅助文件。具体自己探索。

zemi发表的内容如下：

> 我认为新的论文中必须包含FTWP上的实验。但是FTWP有514个游戏，而且游戏的流程比TWC更长，因此全部使用GPT4o来完成是不切实际的（消耗会非常大）。

> 全部使用4omini的话，成本可以控制在10美元以下（估计）。因此最近的实验主要探索了如何通过提示工程提高4omini的性能。但是4omini的性能很差，在TWC数据集上，无论怎么改进prompt性能都会落在0.25~0.35的范围内。

> 最近一个实验尝试在TWC数据集上对4omini进行微调，性能改进到了0.54。微调用的数据是我自己在5个困难等级的训练集游戏上游玩的结果。微调以及使用4omini的成本都很低，我认为可以在这个方向上继续进行研究。因此现在的研究方向主要是寻找一个最大化微调4omini性能的策略。

# 2024.11.28 使用自己的反思来微调自己4omini

结果并不理想。虽然只实验了一次，分数为0.24.

可以想象的原因：
1. 训练导致注意力放在生成思考上而不是去选择最好的答案。-> 为了解决这个问题，应该让思考抱持简单明了，可以让4o来生成思考。
2. 4omini自己的思考比较挫。 -> 可以用4o来思考。或者也可以先强调要抱持简洁，让4omini先输出着看看。
3. 训练的次数不够。 -> 可以再多训练训练，不过这个不太确定。

## 2024.12.2 使用更好的cot来微调4omini

尝试把结论放在最后，并再次微调，结果确实比上次好，但是不比没有cot输出的模型更好。因此放弃这个想法，转向简化description & action list微调。既然大模型上的实验证明可行，微调实验应该也不错。