# A clone from TextWorld Commonsense (TWC)

# 2024.8.9

对项目进行了重构，以降低每次重新操业的学习成本。
* 第一，将关于原生游戏环境的所有代码放入`env.py`，直接初始化Env对象，就可以调用act(command)方法，该方法相当于直接在命令行中输入指令command。
* 第二，将LLM调用的所有代码放入`llm_caller.py`。直接初始化GPT_Caller对象，就可以调用act_and_call(command)方法，该方法会自动调用env的act方法跟游戏环境交互，得到反馈之后，调用在初始化时候设定好的LLM模型。

