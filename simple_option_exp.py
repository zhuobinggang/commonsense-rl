from llm_caller import Builder1, Prompt_builder
import global_variable as G

class Builder_Simple_Option(Builder1):
    def __init__(self):
        super().__init__()
    def build(self, current_enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False):
        builder = Prompt_builder()
        builder.inventory = inventory
        builder.current_enviroment = current_enviroment
        move_actions = [act for act in available_actions if act.startswith('go ')]
        if len(move_actions) > 0:
            move_action = [act for act in available_actions if act.startswith('go ')][0]
            builder.action_list = f'1. take [object]\n2. put [object in inventory] on [supporter]\n3. put [object in inventory] in [container]\n4. {move_action}' # NOTE: NEW
        else:
            builder.action_list = f'1. take [object]\n2. put [object in inventory] on [supporter]\n3. put [object in inventory] in [container]' # NOTE: NEW
        action_history = ''
        if len(action_obs_pairs) > 0:
            for idx, (act, obs) in enumerate(action_obs_pairs):
                action_history += f'Action {idx}: {act} -> {obs} '
        else:
            action_history = 'No action was taken now.'
        builder.action_history = action_history
        if zero_shot:
            builder.example = None
        else: # one shot
            if one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_SIMPLE_NO_AUG
            elif one_shot_easy and not no_augment:
                builder.example = G.ONE_SHOT_EXP_AUGMENT_SIMPLE
            elif not one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_NO_AUG
            elif not one_shot_easy and not no_augment: # 提案手法
                builder.example = G.ONE_SHOT_EXP_AUGMENT
        if cot:
            builder.question = G.QUESTION
        else:
            builder.question = G.QUESTION_NO_COT
            builder.consideration = None
        self.builder = builder 
        self.builder.build()


