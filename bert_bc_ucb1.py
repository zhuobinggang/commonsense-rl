# 使用ucb1来增强behavior cloning
# TODO: 将Game继承过来，增加get_state方法，获取游戏的inventory，recipe还有location信息组成一个state文本返回
# TODO: 将Model继承过来，增加get_command_logits(game_state)和next_action_ucb1(self, logits, state_action_count)方法
from bert_behavior_clone import Model_behavior_clone, Game_no_history
from bert_common import Game_for_rl, get_command_logits_simple, Game_state, game_for_train, SOME_GAMES
import numpy as np
import torch
import common

# Example:
# Room: Kitchen
# Inventory: {'an orange bell pepper', 'a knife'}
# Recipe: True
def game_state_to_ucb1_key(game_state: Game_state):
    recipe_got = True if game_state.recipe else False
    return f'Room: {game_state.location}\nInventory: {game_state.inventory}\nRecipe: {recipe_got}'


def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt) # 如果total_cnt=10, action_cnt=1，值大概为2.15，总之都不会比2.5大的感觉。

def maxmin_norm(p): # 返回0-1之间的值
    return (p - p.min())/(p.max() - p.min())

def choose_action_ubc1(logits, action_visited_count, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    total_visits = sum(action_visited_count) # 该状态下所有行动的执行次数
    uscore = [alpha*ucb1(v, total_visits) for v in action_visited_count] # 加权分
    ssc = maxmin_norm(logits) + torch.tensor(uscore).cuda() # 如果所有指令都没有访问过，那么所有指令的分数都是5，很公平。如果指令被访问过，那么logits几乎不影响它的分数 -> 意味着模型会优先选择没有访问过的指令
    return ssc.argmax(), ssc.softmax(dim=0)

class Model_ucb1(Model_behavior_clone):
    def __init__(self, bert=None):
        super().__init__(bert)
        self.state_action_count = {}
        self.verbose = False

    def get_command_logits(self, game_state):
        logits = get_command_logits_simple(self.bert, game_state)
        return logits
    
    def incresase_state_action_count(self, key, action):
        self.state_action_count[key][action] += 1

    def get_state_action_count(self, key, action):
        if key not in self.state_action_count:
            self.state_action_count[key] = {}
        if action not in self.state_action_count[key]:
            self.state_action_count[key][action] = 0
        return self.state_action_count[key][action]

    def reset_state_action_count(self):
        self.state_action_count = {}

    def update_move_action_visited_count(self, location, actions, action_visited_count, world_map):
        """
        读取所有移动指令，如果一个方向的dict存在就意味着它已经被访问过，
        我们“将它视为已经执行过的指令”，但是如果所有方向的dict都存在，
        我们把它们都“视为未执行过的指令”。
        2025.3.17更新！
        我们可以先判断move_action是否确实执行过，是的话，保留值，
        不是的话，观察地图，如果知道房间的存在，我们把这个值临时赋予为1。
        如果true_visited_count + kown_exist_count == 移动指令的总和，
        我们清空known_move_action的值。
        """
        move_directions = ["go north", "go south", "go east", "go west"] 
        move_actions = [action for action in actions if action in move_directions]
        still_new_place_to_go = False
        passed_actions = []
        for action in move_actions:
            direction = action.replace('go ', '')
            action_idx = actions.index(action)
            if action_visited_count[action_idx] > 0: # 实际走过的路
                pass
            else:
                if direction in world_map.get(location, {}): # 来时经过的路
                    action_visited_count[action_idx] = 1
                    passed_actions.append(action)
                else: # 既不在world_map中（不是来时的路），也不在action_visited_count中（不是实际访问过的路），说明有访问的余地，不需要清空来时的路的标记
                    still_new_place_to_go = True
        if not still_new_place_to_go: # 如果所有的路都走过了，那么清空来时的路的标记
            for action in passed_actions:
                action_visited_count[actions.index(action)] = 0
        return action_visited_count

    @torch.no_grad()
    def next_action(self, game_state: Game_state):
        if len(game_state.action_obs_pairs) == 0: # 新游戏
            self.reset_state_action_count()
        logits = self.get_command_logits(game_state) # 注意，logits并没有得到softmax处理
        commands = game_state.action_list
        key = game_state_to_ucb1_key(game_state)
        action_visited_count = [self.get_state_action_count(key, command) for command in commands]
        action_visited_count = self.update_move_action_visited_count(game_state.location, commands, action_visited_count, game_state.world_map)
        best_action_idx, action_prob = choose_action_ubc1(logits, action_visited_count)
        best_action = commands[best_action_idx]
        self.incresase_state_action_count(key, best_action)
        if self.verbose:
            common.beutiful_print_command_and_probs(commands, action_prob)
        return best_action
    
    def test_full(self, game_init_func=Game_for_rl):
        from bert_common import run_test_full
        run_test_full(self, file_prefix=self.prefix, game_init_func=game_init_func)
    
def convinient_script():
    game =game_for_train(SOME_GAMES['game_need_navigation'], verbose=True, reset=True)
    model = Model_ucb1()
    path = '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0311/bert_behavior_clone0.tch'
    model.load_checkpoint(path)
    model.verbose = True
    return game, model

def batch_test_with_history_and_ucb1():
    for i in range(3):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_with_history_and_ucb1'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0311/bert_behavior_clone{i}.tch'
        model.load_checkpoint(path)
        model.test_full()

def batch_test_with_history_and_ucb1():
    for i in range(3):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_with_history_and_ucb1'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0311/bert_behavior_clone{i}.tch'
        model.load_checkpoint(path)
        model.test_full()

def batch_test_wo_history_and_ucb1():
    for i in range(3):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_wo_history_and_ucb1'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_no_history_0318/bert_behavior_clone{i}_no_history.tch'
        model.load_checkpoint(path)
        model.test_full(game_init_func=Game_no_history)

def night_run():
    batch_test_with_history_and_ucb1()
    from bert_behavior_clone import train
    train(repeat = 3, epoch = 3, need_history=False) # 3.17 不需要历史记录的版本
    common.shutdown()

def day_run():
    batch_test_wo_history_and_ucb1()