# 2024.11.13 将精简化进行到底

from llm_caller import get_client, GPT_Caller, text_from_raw_response, Claude_Caller
import global_variable as G

GPT4O = 'gpt-4o-2024-08-06'
GPT4OMINI = 'gpt-4o-mini-2024-07-18'

class Summarization_Prompt_builder:

    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.desc = ''

    def build(self):
        system_msg = ''
        system_msg += 'Reform the environment description using format: Room[Furniture[Object]]\n'
        system_msg += 'Example: Bedroom[wardrobe, chest of drawers[black sock], desk[pen, eraser]]\n'
        system_msg += 'Response with the new environment description directly.\n'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += 'The environment description:\n'
        user_msg += self.desc.strip().replace('\n', '')
        user_msg = user_msg.strip() + '\n'
        user_msg += 'Response:\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg
    


class Prompt_builder:

    def __init__(self,
                 task=G.TASK_NEW,
                 action_history=None,
                 inventory=None,
                 current_enviroment=None,
                 example=None,
                 action_list=None,
                 question=None,
                 consideration=None,
                 next_action=None,
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
        system_msg += f'Environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += f'Question: {self.question}\n' if self.question else ''
        user_msg += f'Consideration: {self.consideration}\n' if self.consideration else ''
        user_msg += f'Next action: {self.next_action}\n' if self.next_action else ''
        # NOTE: 2024.11.13
        user_msg += 'Answer (in json form: {"consideration": string, "action": string}):'
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'


class Builder_old_style:  # 2024.8.9之前的

    def __init__(self):
        self.builder = None
        self.name = 'normal_builder'

    def build(self,
              current_enviroment,
              inventory,
              available_actions,
              action_obs_pairs=[],
              another_room_info = ''):
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
        # another room info
        builder.another_room_info = another_room_info
        self.builder = builder
        self.builder.build()

    def sys_usr_msg(self):
        return self.builder.system_msg, self.builder.user_msg
    


def quest_summarization(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI,
                        verbose = True):
    client = get_client()
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
    content = completion.choices[0].message.content
    if verbose:
        print(content)
    usage = str(completion.usage)
    return content


class GPT_Caller_Simple_Desc(GPT_Caller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summarize_prompt_builder = Summarization_Prompt_builder()
        self.summarize_log = ''

    def updated_description(self, description):
        self.summarize_prompt_builder.desc = description
        self.summarize_prompt_builder.build()
        sys_msg, usr_msg = self.summarize_prompt_builder.sys_usr_msg()
        new_desc = quest_summarization(sys_msg, usr_msg)
        self.summarize_log += f'{self.summarize_prompt_builder.prompt}\n\n{new_desc}\n\n'
        return new_desc
        
    def save_hook(self):
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}_summarization_log.txt'
        f = open(filename, 'w')
        f.write(self.summarize_log)
        f.close()



class Builder_Simple_Option(Builder_old_style):
    def __init__(self):
        super().__init__()
    def build(self, current_enviroment, inventory, available_actions, action_obs_pairs = []):
        super().build(current_enviroment, inventory, available_actions, action_obs_pairs)
        move_actions = [act for act in available_actions if act.startswith('go ')]
        if len(move_actions) > 0:
            move_action = [act for act in available_actions if act.startswith('go ')][0]
            self.builder.action_list = f'1. take [object]\n2. put [object in inventory] on [supporter]\n3. put [object in inventory] in [container]\n4. {move_action}' # NOTE: NEW
        else:
            self.builder.action_list = f'1. take [object]\n2. put [object in inventory] on [supporter]\n3. put [object in inventory] in [container]' # NOTE: NEW
        self.builder.build()


def quest_gpt_simple(client, system_msg, user_msg, gpt_type, verbose = False):
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
    usage = str(completion.usage)
    # To clipboard
    # NOTE: 自动提取command
    text = text_from_raw_response(completion)
    obj = json_obj_from_text(text)
    if verbose:
        print(text)
        print('------------------>')
        print(obj)
    dic = {'response': obj, 'usage': usage}
    the_command = obj['action'] # Might be None
    return completion, dic, the_command

class GPT_Caller_Simplify(Claude_Caller):
    def __init__(self,
                 env,
                 step_limit=20,
                 gpt_type=GPT4O,
                 builder=None,
                 filename_prefix=''):
        self.gpt_type = gpt_type
        self.env = env
        self.step_limit = step_limit
        self.filename_prefix = filename_prefix
        self.file_name_generate()
        self.step_counter = 0
        # Add 2024.8.9
        self.no_augment = False
        self.init_prompt_builder(builder)
        self.init_summarize_prompt_builder()
        self.summarize_log = ''

    def init_prompt_builder(self, builder):
        self.builder = builder or Builder_Simple_Option()

    def init_summarize_prompt_builder(self):
        self.summarize_prompt_builder = Summarization_Prompt_builder()

    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}SIMPLIFIED_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)

    def description_updated_callback(self, old_desc, new_desc):
        pass

    def updated_description(self, description):
        self.summarize_prompt_builder.desc = description
        self.summarize_prompt_builder.build()
        sys_msg, usr_msg = self.summarize_prompt_builder.sys_usr_msg()
        new_desc = quest_summarization(sys_msg, usr_msg)
        self.summarize_log += f'{self.summarize_prompt_builder.prompt}\n\n{new_desc}\n\n'
        self.description_updated_callback(description, new_desc)
        return new_desc
        
    def save_hook(self):
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}_summarization_log.txt'
        f = open(filename, 'w')
        f.write(self.summarize_log)
        f.close()

    def quest_my_llm(self, system_msg, user_msg, llm_type, verbose = False):
        complete, dic, the_command = quest_gpt_simple(get_client(), system_msg,
                                        user_msg,
                                        llm_type, verbose=verbose) # 获得the_command，可能为空
        return complete, dic, the_command
    
    def build_prompt(self, desc, inventory, available_actions, act_obs_pairs):
        self.builder.build(desc,
                    inventory,
                    available_actions,
                    act_obs_pairs)
    
    def __call__(self, description, inventory, available_actions,
                 action_obs_pairs, need_print = False):
        self.build_prompt(description, inventory, available_actions, action_obs_pairs)
        system_msg, user_msg = self.builder.sys_usr_msg()
        complete, dic, the_command = self.quest_my_llm(system_msg,
                                             user_msg,
                                             self.gpt_type) # 获得the_command，可能为空
        if the_command is None:
            print('__call__(): QUEST LLM GET NONE COMMAND, THE RESPONSE DIC IS BELOW, I WILL TRY IT AGAIN!')
            print(dic)
        if self.env is not None:
            self.env.env.system_user_msgs.append(system_msg + user_msg) # 2024.11.9: 不管command是否为空，都存储sys, user信息，这个可以用于再次请求才对。
            self.env.env.gpt_responses.append(complete)
            self.env.env.readable_log += (system_msg + user_msg + '\n\n\n' +
                                          str(dic['response']) + '\n\n\n\n')
        return the_command


def json_obj_from_text(text):
    import re
    import json
    text = text.replace('\n','')
    pattern = r'\{.+\}'
    result = re.search(pattern, text)
    try:
        json_string = result.group(0)
        json_data = json.loads(json_string)
        print(json_data)
        return json_data
    except IndexError:
        print("No json found!")
        return None