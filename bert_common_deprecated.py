import common

class Prompt_builder_for_bert:
    def __init__(self):
        self.room_name = ''
        self.action_history = ''
        self.action_list = ''
        # others
        self.recipe = ''
        self.prompt = ''
        self.prompt_without_final_hint = ''

    def build(self):
        user_msg = ''
        user_msg += f"Room: {self.room_name if self.room_name else 'Unknown now'}\n"
        user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
        user_msg += f"Action history: {self.action_history if self.action_history else ''}\n" 
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        self.prompt_without_final_hint = user_msg
        user_msg += 'Next action: [MASK]'
        self.prompt = user_msg

def prompt_from_env_feedback_action_template(description, action_obs_pairs, available_actions, recipe, with_final_hint = True, need_action_history = True):
    promptor = Prompt_builder_for_bert()
    if need_action_history:
        promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
    promptor.room_name = common.extract_room_name(description)
    promptor.action_list = common.actions_to_list_number(available_actions)
    promptor.recipe = recipe
    promptor.build()
    if with_final_hint:
        return promptor.prompt
    else:
        return promptor.prompt_without_final_hint

def bert_prompt_from_game(game, with_final_hint = True, need_action_history = True):
    return prompt_from_env_feedback_action_template(game.description, 
                                                    game.action_obs_pairs, 
                                                    game.available_actions, 
                                                    game.recipe, 
                                                    with_final_hint = with_final_hint, 
                                                    need_action_history=need_action_history)