import global_variable as G
from openai import OpenAI
import re
import pyperclip

client = OpenAI()


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


def quest_gpt_raw(system_msg, user_msg, gpt_type):
    completion = client.chat.completions.create(
        model=gpt_type,  # 
        messages=[{
            "role": "system",
            "content": system_msg
        }, {
            "role": "user",
            "content": user_msg
        }],
        temperature=0)
    # To clipboard
    content = completion.choices[0].message.content
    usage = str(completion.usage)
    # To clipboard
    # NOTE: 自动提取command
    copied = False
    the_command = None
    for line in content.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            #text_to_paste = re.sub('^\d\.\s*', '', text_to_paste)
            the_command = text_to_paste
            pyperclip.copy(f"c.act_until_error('{text_to_paste}')")
            print(f'COMMAND GOT: {text_to_paste}')
            copied = True
    if not copied:
        pyperclip.copy('')
        print(f'BE CAREFULL!')
    dic = {'response': content, 'usage': usage}
    return completion, dic, the_command


class GPT_Caller:

    def __init__(self,
                 env,
                 zero_shot=True,
                 gpt_type='gpt-3.5-turbo-0613',
                 cot=True,
                 one_shot_easy=False,
                 no_augment=False,
                 step_limit=20,
                 builder=None):
        self.zero_shot = zero_shot
        self.gpt_type = gpt_type
        self.env = env
        self.cot = cot
        self.one_shot_easy = one_shot_easy
        self.no_augment = no_augment
        self.step_limit = step_limit
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

    def file_name_generate(self):
        shot = 'ZERO_SHOT' if self.zero_shot else 'ONE_SHOT'
        if not self.zero_shot:
            shot += '_EASY' if self.one_shot_easy else '_NORMAL'
        cot = 'COT_ON' if self.cot else 'COT_OFF'
        augment = 'AUGMENT_OFF' if self.no_augment else 'AUGMENT_ON'
        self.filename = f'{shot}_{cot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}.pkl'
        self.filename_raw = f'{shot}_{cot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        print(self.filename_raw)

    def __call__(self, description, inventory, available_actions,
                 action_obs_pairs):
        self.builder.build(description,
                           inventory,
                           available_actions,
                           action_obs_pairs,
                           zero_shot=self.zero_shot,
                           cot=self.cot,
                           one_shot_easy=self.one_shot_easy,
                           no_augment=self.no_augment)
        system_msg, user_msg = self.builder.sys_usr_msg()
        dd, dic, the_command = quest_gpt_raw(system_msg,
                                             user_msg,
                                             gpt_type=self.gpt_type)
        if self.env is not None:
            self.env.env.system_user_msgs.append(system_msg + user_msg)
            self.env.env.gpt_responses.append(dd)
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

    def recall_and_get_command(self):
        return self.__call__(self.desc, self.inventory,
                                    self.available_actions,
                                    self.action_obs_pairs)
        
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
                preposition = 'in'
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
        if command_backup != command:
            self.env.env.readable_log += f'\n\nCommand adjusted: {command_backup} -> {command}\n\n'
        return description, inventory, available_actions, action_obs_pairs

    def act_and_call(
            self,
            command=None):  # @RETURN: None means 2 path, first means the command non-executable, second means response from LLM irregular.
        if self.step_counter <= self.step_limit:
            description, inventory, available_actions, action_obs_pairs = self.try_adjust_and_execute(command)
            if description is None and not self.env.env.end:
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
            if self.env.env.end: # 如果赢了就不再需要调用LLM
                print('YOU WIN! NO API CALL NEED.')
                self.save()
                return None
            else:
                # 更新房间描述的钩子函数
                description = self.updated_description(description)
                self.set_act_result_to_body(description, inventory, available_actions, action_obs_pairs)
                self.env.env.score_by_step.append(self.env.env.last_reward)
                return self.recall_and_get_command() # 得到的是提取成功的，如果提取不成功，需要手动操作
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

    def log(self, index=-4000):
        print(self.env.env.readable_log[index:])
