from bert_cogni_agent import *
from bert_bc_ucb1 import choose_action_ubc1, game_state_to_ucb1_key
import common

logger = logging.getLogger('cogni_agent_ucb1')
dbg = logger.debug

class QAUcb1(QAModel):
    def __init__(self, num_labels=2):
        super().__init__(num_labels)
        self.state_action_count = {}
        self.checkpoint_name = 'qa_ucb1'
    
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
    
    def get_commands_from_game_state(self, game_state):
        return game_state.action_list

    def reset_state_action_count(self):
        self.state_action_count = {}

    def get_state_action_count(self, key, action):
        if key not in self.state_action_count:
            self.state_action_count[key] = {}
        if action not in self.state_action_count[key]:
            self.state_action_count[key][action] = 0
        return self.state_action_count[key][action]

    def incresase_state_action_count(self, key, action):
        self.state_action_count[key][action] += 1

    @torch.no_grad()
    def next_action(self, game_state: Game_state):
        if len(game_state.action_obs_pairs) == 0: # 新游戏
            self.reset_state_action_count()
        # logits = self.get_command_logits(game_state) # 注意，logits并没有得到softmax处理
        commands, logits = self.get_command_probs(game_state)
        logits = torch.from_numpy(logits).cuda()
        # commands = self.get_commands_from_game_state(game_state)
        key = game_state_to_ucb1_key(game_state)
        action_visited_count = [self.get_state_action_count(key, command) for command in commands]
        action_visited_count = self.update_move_action_visited_count(game_state.location, commands, action_visited_count, game_state.world_map)
        best_action_idx, action_prob = choose_action_ubc1(logits, action_visited_count)
        best_action = commands[best_action_idx]
        self.incresase_state_action_count(key, best_action)
        dbg(key)
        common.beutiful_print_command_and_probs(commands, action_prob, log_func = dbg)
        return best_action