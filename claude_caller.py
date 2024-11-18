from functools import lru_cache
import global_variable as G
import pyperclip
import re

@lru_cache(maxsize=None)
def get_client():
    import anthropic
    client = anthropic.Anthropic()
    return client

def fake_message():
    system = """Task: You are a experienced text game player, your goal is put things in there proper locations and improve your score.
Example walkthrough: Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point.Right position. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point.Right position. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point.Right position. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point.Right position. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point.Right position. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point.Right position. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.Right position.
Action history: No action was taken now.
Inventory: You are carrying:  some milk
Current enviroment: -= Backyard =-You've entered a backyard.You see a BBQ. The BBQ is recent. On the BBQ you make out a wooden spoon. You see a clothesline. The clothesline is typical. But the thing is empty. Hm. Oh well What's that over there? It looks like it's a patio chair. Now why would someone leave that there? On the patio chair you can see a wet white jumper. You see a patio table. The patio table is stylish. The patio table appears to be empty. Hey, want to see a workbench? Look over there, a workbench. On the workbench you see a clean pot. Something scurries by right in the corner of your eye. Probably nothing.There is an open sliding patio door leading west.
"""
    user = """Action you can take:
* close sliding patio door
* drop milk
* examine BBQ
* examine clean pot
* examine clothesline
* examine patio chair
* examine patio table
* examine wet white jumper
* examine wooden spoon
* examine workbench
* go west
* look
* put milk on BBQ
* put milk on clothesline
* put milk on patio chair
* put milk on patio table
* put milk on workbench
* take clean pot from workbench
* take wet white jumper from patio chair
* take wooden spoon from BBQ

Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""
    return system, user


def text_from_raw_response(claude_response):
    return claude_response.content[0].text

def usage_from_raw_response(claude_response):
    return claude_response.usage

def command_from_text(text):
    copied = False
    the_command = None
    for line in text.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            #text_to_paste = re.sub('^\d\.\s*', '', text_to_paste)
            the_command = text_to_paste
            pyperclip.copy(f"c.act_until_error('{text_to_paste}')")
            print(f'COMMAND GOT: {text_to_paste}')
            copied = True
    if not copied:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX command from text failed XXXXXXXXXXXXXX')
        print(text)
        pyperclip.copy('')
        print(f'BE CAREFULL!')
    return the_command

def quest_claude(client, system, user, claude_type = "claude-3-5-sonnet-20241022", max_tokens = 500, need_print = False):
    message = client.messages.create(
        model=claude_type,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user
                    }
                ]
            }
        ]
    )
    text = text_from_raw_response(message)
    if need_print:
        print(text)
    dic = {'response': text, 'usage': usage_from_raw_response(message)}
    the_command = command_from_text(text) # Might be None
    return message, dic, the_command


class Prompt_builder:

    def __init__(self,
                 task=G.TASK,
                 action_history=None,
                 inventory=None,
                 current_enviroment=None,
                 example=None,
                 action_list=None,
                 question=G.QUESTION,
                 consideration=G.FILL_IN_TEMPLATE,
                 next_action=G.FILL_IN_TEMPLATE,
                 another_room_info=None):
        self.task = task
        self.action_history = action_history
        self.inventory = inventory
        self.current_enviroment = current_enviroment
        self.example = example
        self.action_history = action_history
        self.action_list = action_list
        self.question = question
        self.consideration = consideration
        self.next_action = next_action
        self.another_room_info = another_room_info
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''

    def build(self):
        system_msg = ''
        system_msg += f'Task: {self.task}\n' if self.task else ''
        system_msg += f'Example walkthrough: {self.example}\n' if self.example else ''
        system_msg += f'Action history: {self.action_history}\n' if self.action_history else ''
        system_msg += f'Inventory: {self.inventory}\n' if self.inventory else ''
        system_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        system_msg += f'Current enviroment: {self.current_enviroment}\n' if self.current_enviroment else ''
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action you can take:\n{self.action_list}\n' if self.action_list else ''
        user_msg += f'Question: {self.question}\n' if self.question else ''
        user_msg += f'Consideration: {self.consideration}\n' if self.consideration else ''
        user_msg += f'Next action: {self.next_action}\n' if self.next_action else ''
        user_msg = user_msg.strip()
        self.user_msg = user_msg
        self.prompt = f'{system_msg}\n{user_msg}'


class Builder1:  # 2024.8.9之前的

    def __init__(self):
        self.builder = None
        self.name = 'normal_builder'

    def build(self,
              current_enviroment,
              inventory,
              available_actions,
              action_obs_pairs=[],
              zero_shot=True,
              cot=True,
              one_shot_easy=False,
              no_augment=False):
        builder = Prompt_builder()
        builder.inventory = inventory
        builder.current_enviroment = current_enviroment
        available_action_text = ''
        for act in available_actions:
            available_action_text += f'* {act}\n'
        builder.action_list = available_action_text
        action_history = ''
        if len(action_obs_pairs) > 0:
            for idx, (act, obs) in enumerate(action_obs_pairs):
                action_history += f'Action {idx}: {act} -> {obs} '
        else:
            action_history = 'No action was taken now.'
        builder.action_history = action_history
        if zero_shot:
            builder.example = None
        else:  # one shot
            if one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_SIMPLE_NO_AUG
            elif one_shot_easy and not no_augment:
                builder.example = G.ONE_SHOT_EXP_AUGMENT_SIMPLE
            elif not one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_NO_AUG
            elif not one_shot_easy and not no_augment:  # 提案手法
                builder.example = G.ONE_SHOT_EXP_AUGMENT
        if cot:
            builder.question = G.QUESTION
        else:
            builder.question = G.QUESTION_NO_COT
            builder.consideration = None
        self.builder = builder
        self.builder.build()

    def sys_usr_msg(self):
        return self.builder.system_msg, self.builder.user_msg
    




class Claude_Caller:

    def __init__(self,
                 env,
                 zero_shot=True,
                 gpt_type='claude-3-5-sonnet-20241022',
                 cot=True,
                 one_shot_easy=False,
                 no_augment=False,
                 step_limit=20,
                 builder=None,
                 filename_prefix=''):
        self.zero_shot = zero_shot
        self.gpt_type = gpt_type
        self.env = env
        self.cot = cot
        self.one_shot_easy = one_shot_easy
        self.no_augment = no_augment
        self.step_limit = step_limit
        self.filename_prefix = filename_prefix
        self.file_name_generate()
        # print(f'ZERO SHOT: {zero_shot}')
        # print(f'COT: {cot}')
        # print(f'GPT VERSION: {gpt_type}')
        # print(f'ONE_SHOT_EASY: {one_shot_easy}')
        # print(f'NO_AUGMENT: {no_augment}')
        # print(f'STEP_LIMIT: {step_limit}')
        self.step_counter = 0
        # Add 2024.8.9
        self.builder = builder or Builder1()
        # 2024.11.16 用于存储行动之前的状态信息，desc之类的，用于展示相邻房间的信息
        self.info_backup = {}

    def file_name_generate(self):
        shot = 'ZERO_SHOT' if self.zero_shot else 'ONE_SHOT'
        if not self.zero_shot:
            shot += '_EASY' if self.one_shot_easy else '_NORMAL'
        cot = 'COT_ON' if self.cot else 'COT_OFF'
        augment = 'AUGMENT_OFF' if self.no_augment else 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename = f'{filename_prefix}{shot}_{cot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}.pkl'
        self.filename_raw = f'{filename_prefix}{shot}_{cot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        print(self.filename_raw)

    def quest_my_llm(self, system_msg, user_msg, llm_type, need_print = False):
        complete, dic, the_command = quest_claude(get_client(), system_msg,
                                        user_msg,
                                        claude_type=llm_type, need_print=need_print) # 获得the_command，可能为空
        return complete, dic, the_command

    def __call__(self, description, inventory, available_actions,
                 action_obs_pairs, need_print = False):
        self.builder.build(description,
                           inventory,
                           available_actions,
                           action_obs_pairs,
                           zero_shot=self.zero_shot,
                           cot=self.cot,
                           one_shot_easy=self.one_shot_easy,
                           no_augment=self.no_augment)
        system_msg, user_msg = self.builder.sys_usr_msg()
        complete, dic, the_command = self.quest_my_llm(system_msg,
                                             user_msg,
                                             self.gpt_type, need_print=need_print) # 获得the_command，可能为空
        if the_command is None:
            print('__call__(): QUEST LLM GET NONE COMMAND, THE RESPONSE DIC IS BELOW, I WILL TRY IT AGAIN!')
            print(dic)
        if self.env is not None:
            self.env.env.system_user_msgs.append(system_msg + user_msg) # 2024.11.9: 不管command是否为空，都存储sys, user信息，这个可以用于再次请求才对。
            self.env.env.gpt_responses.append(complete)
            self.env.env.readable_log += (system_msg + user_msg + '\n\n\n' +
                                          dic['response'] + '\n\n\n\n')
        return the_command

    def is_command_available(self, the_command, available_actions):
        available_actions = [act.lower() for act in available_actions]
        if the_command in available_actions:
            return True
        commands = []
        for act in available_actions:
            if act.startswith('take'):
                commands.append(re.sub('\sfrom\s.*', '', act))
            if act.startswith('go '):
                commands.append(act)
            if act.startswith('put '):
                commands.append(act)
            if act.startswith('insert '):
                commands.append(
                    act.replace('insert', 'put').replace('into', 'in'))
        if the_command not in commands:
            print('XXXXXXXXXXXX')
            print(the_command)
            print(commands)
            return False
        else:
            return True

    def updated_description(self, description):
        return description

    def set_act_result_to_body(self, desc, inventory, actions, action_obs_pairs):
        self.desc = desc
        self.inventory = inventory
        self.available_actions = actions
        self.action_obs_pairs = action_obs_pairs

    def recall_and_get_command(self, need_print= False):
        return self.__call__(self.desc, self.inventory,
                                    self.available_actions,
                                    self.action_obs_pairs, need_print = need_print)
        
    def command_remove_number_prefix(self, cmd):
        # NOTE: Added 2024.8.11 去除 1. take xxx 类似的数字前缀
        cmd = re.sub('^\d\.\s*', '', cmd)
        return cmd
    
    def object_cut(self, obj, idx = -2):
        obj = obj.strip()
        ss = obj.split()
        if len(ss) > 1:
            obj = ' '.join(ss[idx:]) # 只保留两个
        return obj
        
    def command_cut_object_name(self, cmd):
        if cmd.startswith('put'):
            obj = None
            place = None
            preposition = None
            objs = re.findall('^put\s(.*?)\son\s(.*)', cmd)
            if len(objs) > 0:
                preposition = 'on'
            else:
                objs = re.findall('^put\s(.*?)\sin\s(.*)', cmd)
                if len(objs) > 0:
                    preposition = 'in'
                else:
                    objs = re.findall('^put\s(.*?)\sinto\s(.*)', cmd)
                    if len(objs) > 0:
                        preposition = 'in'
                    else:
                        print('SOME THING WRONG! PUT WITHOUT PREPOSITION!')
                        return cmd # 提取失败的情况
            # 成功提取的情况
            obj, place = objs[0]
            obj = self.object_cut(obj) # 削减obj
            # 重新组装cmd
            cmd = ' '.join(['put', obj, preposition, place])
            return cmd
        if cmd.startswith('take'):
            objs = re.findall('^take\s(.*?)\sfrom\s(.*)', cmd)
            if len(objs) > 0:
                obj, place = objs[0]
                obj = self.object_cut(obj) # 削减obj
                cmd = ' '.join(['take',obj])
                return cmd
            obj = cmd.strip().replace('take ', '')
            obj = self.object_cut(obj) # 削减obj
            cmd = ' '.join(['take', obj])
            return cmd
        print('CMD NOT START WITH [put, take]!')
        return cmd # 2024.11.11 BUG
        
    def command_switch_preposition(self, cmd):
        if cmd.startswith('put'):
            if ' in ' in cmd:
                return cmd.replace(' in ', ' on ')
            elif ' on ' in cmd:
                return cmd.replace(' on ', ' in ')
            else:
                print('Something Wrong about command_switch_preposition()!')
                return cmd
        else:
            print('No need switch preposition.')
            return cmd

    def try_adjust_and_execute(self, command): # 2024.8.17 
        command_backup = command
        description, inventory, available_actions, action_obs_pairs = self.env.act(command, no_augment=self.no_augment)
        if description is None and not self.env.env.end:
            command = self.command_remove_number_prefix(command)
            print('Trying command_remove_number_prefix')
            description, inventory, available_actions, action_obs_pairs = self.env.act(command, no_augment=self.no_augment)
        if description is None and not self.env.env.end:
            command = self.command_cut_object_name(command)
            print('Trying command_cut_object_name')
            description, inventory, available_actions, action_obs_pairs = self.env.act(command, no_augment=self.no_augment)
        if description is None and not self.env.env.end:
            command = self.command_switch_preposition(command)
            print('Trying command_switch_preposition')
            description, inventory, available_actions, action_obs_pairs = self.env.act(command, no_augment=self.no_augment)
        if description is None and not self.env.env.end:
            print(f'TAKU: llm_caller try_ajust_and_execute failed, command_backup : {command_backup}, command_adjust : {command}')
            # NOTE: 2024.11.14 指令无法执行，应该记录到历史记录中
            if command_backup.startswith('put milk in clean pot'):
                obs = 'COMMAND NOT EXECUTABLE. The clean pot is not a container!'
            else:
                obs = 'COMMAND NOT EXECUTABLE. You should try other commands!'
            print(f'指令无法执行，添加到历史记录: {command_backup} -> {obs}')
            self.env.append_command_obs_pair(command_backup, obs)
        if command_backup != command:
            self.env.env.readable_log += f'\n\nCommand adjusted: {command_backup} -> {command}\n\n'
        return description, inventory, available_actions, action_obs_pairs
    
    def env_act_succeed_callback(self):
        pass

    def act_and_call(
            self,
            command=None):  # @RETURN: None means 2 path, first means the command non-executable, second means response from LLM irregular.
        if self.step_counter <= self.step_limit:
            self.current_command = command
            description, inventory, available_actions, action_obs_pairs = self.try_adjust_and_execute(command)
            if description is None and not self.env.env.end: # NOTE: 2024.11.14 指令无法执行，需要将这次失败放到历史记录中，但是不应该计算步数
                # 给予一次重新请求的机会
                print('Please Try Recalling.')
                recommand = self.recall_and_get_command()
                print(f'\n\nTrying recall LLM!!! {command} -> {recommand}\n\n')
                self.env.env.readable_log += f'\n\nTrying recall LLM!!! {command} -> {recommand}\n\n'
                description, inventory, available_actions, action_obs_pairs = self.try_adjust_and_execute(recommand)
                if description is None and not self.env.env.end:
                    print(f'Recall but Failed!')
                    self.env.env.readable_log += f'\n\nRecall but Failed\n\n'
                    return None
                else:
                    print(f'Recall success!')
                    self.env.env.readable_log += f'\n\nRecall success!\n\n'
            # 执行成功之后
            self.step_counter += 1
            self.env_act_succeed_callback()
            if self.env.env.end: # 如果赢了就不再需要调用LLM
                print('YOU WIN! NO API CALL NEED.')
                self.save()
                return None
            else:
                # 更新房间描述的钩子函数
                self.desc_before_update = description
                description = self.updated_description(description)
                self.desc_after_update = description
                self.set_act_result_to_body(self.desc_after_update, inventory, available_actions, action_obs_pairs)
                self.env.env.score_by_step.append(self.env.env.last_reward)
                # NOTE: 2024.11.11 如果返回的文本中提取指令失败，给予一次重试机会
                command_may_none = self.recall_and_get_command()
                if command_may_none is None:
                    print('XXXXXXX')
                    print('act_and_call() request LLM and get command failed, try again (only once)!')
                    command_may_none = self.recall_and_get_command()
                return  command_may_none# 得到的是提取成功的，如果提取不成功，需要手动操作
        else:
            print(
                f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}'
            )
            self.save()
            return None

    def act_until_error(self, command=None):
        next_command = self.act_and_call(command)
        while next_command is not None:
            next_command = self.act_and_call(next_command)

    def save(self):
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}.txt'
        f = open(filename, 'w')
        f.write(env.readable_log)
        f.close()
        # 处理错位问题
        # dic = {'env_meta': env.meta_info, 'system_user_msgs': env.system_user_msgs, 'gpt_responses': env.gpt_responses}
        # import pickle
        # with open(filename, 'wb') as handle:
        #     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.save_hook()
        
    def save_hook(self):
        pass

    def log(self, index=-4000):
        print(self.env.env.readable_log[index:])