# python twc_make_game.py --objects 7 --rooms 2
from twc_make_game import *

config = twc_config()
twc_game_maker = TWCGameMaker(config)

from textworld import GameMaker
from textworld.generator.game import Event, Quest
M = GameMaker()
room = M.new_room('my room')
shit = M.new('o', 'shit')
shit_hole = M.new('c', 'shit hole')
new_shit_hole = M.new('c', 'new shit hole')
shit_hole.add(shit)
room.add(shit_hole)
room.add(new_shit_hole)
M.set_player(room)
# quest1 = ['open shit hole', 'take shit', 'open new shit hole', 'insert shit into new shit hole']
# insert_shit = M.new_event_using_commands(quest1)
win_event = Event(conditions={M.new_fact("in", shit, new_shit_hole)})
quest1 = Quest(win_events=[win_event],
               reward=2)
M.quests = [quest1]




