# Copy CogniTextAgent strategy
from bert_cogni_agent_ucb1 import *
from scipy.special import softmax
from collections import Counter, namedtuple

State = namedtuple('State','description inventory recipe')

# Adapted from NAIL Agent
class Connection:
    def __init__(self, from_location, action, to_location):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location
        return False

    def __str__(self):
        return "{} --({})--> {}".format(self.from_location,
                                        self.action,
                                        self.to_location)

    def __repr__(self):
        return str(self)


# Adapted from NAIL Agent
class ConnectionGraph:
    def __init__(self):
        self._out_graph = {}
        self._in_graph  = {}

    def add_single(self, connection):
        """ Adds a new connection to the graph if it doesn't already exist. """
        from_location = connection.from_location
        to_location = connection.to_location
        if from_location in self._out_graph:
            if connection in self._out_graph[from_location]:
                return
            self._out_graph[from_location].append(connection)
        else:
            self._out_graph[from_location] = [connection]
        if to_location is not None:
            if to_location in self._in_graph:
                self._in_graph[to_location].append(connection)
            else:
                self._in_graph[to_location] = [connection]

    def add(self, connection):
        self.add_single(connection)
        action = self.revert_action(connection.action)
        rev = Connection(connection.to_location, action, connection.from_location)
        self.add_single(rev)

    def revert_action(self, action):
        revmap = {
            'go north': 'go south',
            'go south': 'go north',
            'go east': 'go west',
            'go west': 'go east'
        }
        return revmap[action]

    def incoming(self, location):
        """ Returns a list of incoming connections to the given location. """
        if location in self._in_graph:
            return self._in_graph[location]
        else:
            return []

    def outgoing(self, location):
        """ Returns a list of outgoing connections from the given location. """
        if location in self._out_graph:
            return self._out_graph[location]
        else:
            return []

    def known_exits(self, location):
        return [connection.action for connection in self.outgoing(location)]

    def reset_except_last(self, location):
        """ reset the list of know outgoing locations except the last one added """
        if location in self._out_graph:
            last = self._out_graph[location][-1]
            self._out_graph[location] = [last]


def maxmin_norm(p): # 返回0-1之间的值
    return (p - p.min())/(p.max() - p.min())


def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt) # 如果total_cnt=10, action_cnt=1，值大概为2.15，总之都不会比2.5大的感觉。


def choose_action(logits, sacnt, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    total_visits = sum(sacnt) # 该状态下所有行动的执行次数
    uscore = [alpha*ucb1(v, total_visits) for v in sacnt] # 加权分
    ssc = maxmin_norm(logits) + uscore # 如果所有指令都没有访问过，那么所有指令的分数都是5，很公平。如果指令被访问过，那么原本的概率值几乎不影响它的分数。
    return np.argmax(ssc), softmax(ssc)


class ClassicUcb1(QAUcb1):
    def __init__(self, num_labels=2):
        super().__init__(num_labels)
        self.checkpoint_name = 'classic_ucb1'
        self.reset_params()
        self.state_action_danger = {} # 不需要重置

    def reset_params(self): # 在每次新游戏开始时调用
        self.state_action_cnt = Counter()
        # keep the previous state-action
        self.prev_game_state = Game_state()
        self.prev_action = ''
        # self.state_action_danger = {}
        # capture the recipe when agent examines the cookbook
        # map of places with list of action_spaces
        self.worldmap = ConnectionGraph()
        # regex for hifen words workaround

    def get_action_cnt(self, commands, state_key, location, recipe):
        state_action_cnt = self.state_action_cnt
        res = []
        # check how many go commands exist, if only one it should not be penalized
        # the return to the same location
        n_go_commands = len([c for c in commands if c.startswith('go')])
        # location = state.location # 房间名
        known_exits = self.worldmap.known_exits(location) # 已知的出口
        # all exits were explored restart the penalties to allow agent to return
        if n_go_commands == len(known_exits):
            self.worldmap.reset_except_last(location)
            known_exits = self.worldmap.known_exits(location)
        
        # dbg('[Known Exits] {} -> {}'.format(location, known_exits))
        for action in commands:
            sac = state_action_cnt.get((state_key,action), 0) # 获得state,action对的执行次数。
            
            # reset counter for go commands
            if action.startswith('go'):
                sac = 0
            # put a +1 penalty to avoid returning to the origin location
            if action in known_exits and n_go_commands > len(known_exits):
                sac = 1
                
            # set a large count (penalty) if the state action shows
            # as a critical danger
            key = action + recipe
            if key in self.state_action_danger:
                # print('[危险行为] '+ key)
                dbg('[危险行为]' + key)
                sac = 1000
            
            res.append(sac) # sac看起来是惩罚系数
        return res
    
    def update_map(self, current_loc, prev_loc = '', prev_action = ''):
        if not prev_loc or not prev_action:
            return
        if current_loc != prev_loc:
            if prev_action.startswith('go'):
                connection = Connection(prev_loc, prev_action, current_loc)
                self.worldmap.add(connection)
            else:
                dbg('[ERROR add connection] {} {} {}'.format(prev_loc, prev_action, current_loc))

    # NOTE: @return: state, commands, pred
    def predict_command(self, game_state: Game_state):
        commands, logits = self.get_command_probs(game_state)
        return commands, logits
    
    def act_single(self, game_state: Game_state):
        self.update_map(game_state.location, self.prev_game_state.location, self.prev_action) # NOTE: 记得更新prev_game_state
        commands, pred = self.predict_command(game_state)
        state_as_key = self.state_key_from_game_state(game_state)
        action_cnt = self.get_action_cnt(commands, state_as_key, game_state.location, game_state.recipe) # 该状态下该行动的执行次数，对于go指令，只有0和1两种情况。
        dbg('[Action Count] {}'.format(action_cnt))
        action_idx, action_proba = choose_action(pred, action_cnt)
        dbg(f'[Proba ucb1]: {action_proba}')
        action = commands[action_idx]
        # needs to increase the action_list counter
        self.state_action_cnt[(state_as_key, action)] += 1
        # NOTE: 需要更新prev_game_state和prev_action
        self.prev_game_state = game_state # NOTE: 更新 prev_game_state
        self.prev_action = action # NOTE: game_state中原本没有action，这里我们手动添加
        return state_as_key, action, zip(commands, action_proba)
    
    @torch.no_grad()
    def next_action(self, game_state: Game_state):
        if len(game_state.action_obs_pairs) == 0: # 新游戏
            self.reset_params()
        state_as_key, action, action_probs = self.act_single(game_state)
        return action
    
    def state_key_from_game_state(self, game_state: Game_state):
        s = State(description=game_state.description, inventory=game_state.inventory_raw, recipe=game_state.recipe_raw)
        return s
    
    def update_danger_action(self, danger_action):
        if not self.prev_game_state.location:
            dbg(f'刚开始就失败？{danger_action}')
            return
        # return # 2025.4.3 暂时不更新这个，看看性能
        # state_key = self.state_key_from_game_state(self.prev_game_state)
        action = danger_action
        key = action + self.prev_game_state.recipe
        # print('[更新危险行为] '+ key)
        dbg('[更新危险行为]' + key)
        self.state_action_danger[key] = True

    def test(self, game_count = 30, learnable = False):
        from bert_common import run_test_full
        run_test_full(self, file_prefix=self.checkpoint_name, game_count=game_count, learnable=learnable)

    def validate(self, game_count = 30, learnable = False):
        from bert_common import run_valid_full
        run_valid_full(self, file_prefix=self.checkpoint_name, game_count=game_count, learnable=learnable)
        

def run_test():
    model = ClassicUcb1()
    model.load_checkpoint('/home/taku/research/zhuobinggang/commonsense-rl/exp/auto_filename/qa_model_checkpoint_epoch_3_repeat_0.tch' )
    # model.validate_partial(game_count = 15)
    model.validate(game_count = -1, learnable=True)
