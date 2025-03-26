from bert_bc_ucb1 import Model_ucb1
from bert_ucb1_ner_dataset_create import Game_ner
import common

def batch_test_with_history_and_ucb1():
    for i in range(3):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_with_history_and_ucb1_ner'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_full_hist/bert_behavior_clone{i}.tch'
        model.load_checkpoint(path)
        model.test_full(game_init_func=Game_ner)

def batch_test_wo_history_and_ucb1():
    for i in range(2):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_wo_history_and_ucb1_ner'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_no_history/bert_behavior_clone{i}_no_history.tch'
        model.load_checkpoint(path)
        model.test_full(game_init_func=Game_ner)


def batch_test_history_20_and_ucb1():
    for i in range(3):
        model = Model_ucb1()
        model.prefix = f'bert_behavior_clone{i}_history_window20_ucb1_ner'
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_window_20/bert_behavior_clone{i}_history_window20.tch'
        model.load_checkpoint(path)
        model.test_full(game_init_func=Game_ner)

def night_run_0():
    batch_test_history_20_and_ucb1()
    batch_test_wo_history_and_ucb1()
    batch_test_with_history_and_ucb1()
    common.shutdown()
    