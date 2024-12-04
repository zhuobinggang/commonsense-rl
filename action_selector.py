from llm_simplify import quest_simple_get_text

class Action_select_promptor:
    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.target_action = ''
        self.action_list = []

    def action_list_text(self, action_list):
        return '[' + ', '.join(action_list) + ']'

    def build(self):
        system_msg = ''
        system_msg += 'Task: Select the action from the list that is semantically closest to the target action.\n'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Target Action: {self.target_action}\n'
        user_msg += f'Action List: {self.action_list_text(self.action_list)}\n'
        user_msg += 'Answer (action closest to the target action):\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg

def sys_usr_from_actions_and_action(action_list, action):
    promptor = Action_select_promptor()
    promptor.target_action = action
    promptor.action_list = action_list
    promptor.build()
    return promptor.sys_usr_msg()

def test_promptor():
    action = 'examine table'
    action_list = ['eat book', 'drink food']
    sys, usr = sys_usr_from_actions_and_action(action_list, action)
    print(sys + usr)

def quest_closet_action(action_list, action):
    if len(action_list) < 1:
        return None
    if not action:
        return None
    sys, usr = sys_usr_from_actions_and_action(action_list, action)
    answer = quest_simple_get_text(sys, usr)
    answer = answer.strip()
    if answer not in action_list:
        print('好像没有找到最相近的命令，应该考虑在action history加上类似的提醒。')
        print(answer)
    return answer
    