# 使用平均G来更新policy

from bert_for_ftwp import Model_policy_gradient, Policy_gradient_tester
from bert_common import first_train_game, trained_model_autoplay, initiate_bert
from policy_algorithms import train_one_episode_simplest_baseline
import common

class Model(Model_policy_gradient):
    def __init__(self, bert=None):
        super().__init__(bert)
        self.average_G = 0
    def update_average_G(self, G, alpha = 0.025):
        self.average_G = (1 - alpha) * self.average_G + alpha * G
    def get_average_G(self):
        return self.average_G
    
class Model_speedup(Model):
    def update_average_G(self, G):
        alpha = 0.05 if G  > self.average_G else 0.025
        self.average_G = (1 - alpha) * self.average_G + alpha * G

    
class Tester(Policy_gradient_tester):
    def __init__(self, trained_bert = None, save_prefix = 'policy_gradient') -> None:
        self.init_actor(trained_bert)
        self.game = first_train_game()
        from common import Logger_simple
        self.txtLogger= Logger_simple(save_prefix)
        from bert_common import Logger_loss_and_score
        self.imageLogger = Logger_loss_and_score(save_prefix)
        self.counter = 99
        self.last_valid_score = 0
        self.valid_scores = []
        self.max_valid_score = 0
        self.init_params_for_log() # 2025.2.24
        self.save_prefix = save_prefix # 0303
        self.prefix = save_prefix
        ####
        self.episode_rewards = []
        self.actor_losses = []
        self.average_Gs = []
        self.train_steps_limit = self.init_train_steps_limit()
    def init_train_steps_limit(self):
        return 30
    def init_actor(self, trained_bert):
        if trained_bert:
            self.actor = Model(trained_bert)
            self.trained_bert_provided = True
        else:
            # self.actor = Model_policy_gradient(initiate_lora_bert())
            self.actor = Model(initiate_bert())
            self.trained_bert_provided = False
    def draw_line_chart_no_valid(self):
        common.draw_line_chart(list(range(len(self.episode_rewards))), [self.episode_rewards, self.average_Gs], ['episode_scores', 'G'], path = f'exp/auto_filename/{self.prefix}.png')
    def test_self_improve_on_one_game(self, steps = 2000, use_walkthrough = False):
        self.actor.init_optimizer()
        self.log_steps = 10
        def get_G_denominator():
            game = first_train_game()
            game.reset()
            GAME_MAX_SCORE = game.get_max_score() # I checked it
            G_denominator = GAME_MAX_SCORE / 2 # 用来归一化G
            return G_denominator
        G_denominator = get_G_denominator()
        for i in range(steps):
            game = first_train_game()
            game.verbose = False
            print(f'{i}/{steps}')
            if use_walkthrough:
                _, mean_actor_loss = train_one_episode_simplest_baseline(self.actor, game, walkthrough=game.filter_walkthroughs_with_input_test(), txtLogger = self.txtLogger)
                dic = trained_model_autoplay(game, self.actor.bert)
                norm_score = dic['score'] / dic['max_score']
            else:
                norm_score, mean_actor_loss = train_one_episode_simplest_baseline(self.actor, game, txtLogger = self.txtLogger, steps_limit = self.train_steps_limit)
            print(norm_score)
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(min(mean_actor_loss * 0.1, 1.5)) # NOTE: scale for image ouput
            self.average_Gs.append(self.actor.get_average_G() / G_denominator)
            if i % self.log_steps == 0:
                self.draw_line_chart_no_valid()
                self.txtLogger.write_txt_log()
        self.draw_line_chart_no_valid()
        self.txtLogger.add('训练结束。')
        self.txtLogger.write_txt_log()

class Tester_speedup(Tester):
    def init_actor(self, trained_bert):
        if trained_bert:
            self.actor = Model_speedup(trained_bert)
            self.trained_bert_provided = True
        else:
            # self.actor = Model_policy_gradient(initiate_lora_bert())
            self.actor = Model_speedup(initiate_bert())
            self.trained_bert_provided = False
    def init_train_steps_limit(self):
        return 30
    
def night_run():
    tt = Tester_speedup(save_prefix='one_game_speedup')
    tt.test_self_improve_on_one_game(steps = 10000)
    tt = Tester(save_prefix='one_game_normal')
    tt.test_self_improve_on_one_game(steps = 10000)
    common.shutdown()