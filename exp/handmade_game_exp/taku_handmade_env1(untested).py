print('UPDATED ENVIRONMENT, DID NOT BE TESTED!')
raise Exception('Copy this file to other folder, or game will be bugged!!!')

# # python twc_make_game.py --objects 7 --rooms 2
# from twc_make_game import *
# 
# config = twc_config()
# twc_game_maker = TWCGameMaker(config)

from textworld import GameMaker
from textworld.generator.game import Event, Quest
M = GameMaker()
R1 = M.new_room('small bedroom')
R2 = M.new_room('small kitchen')
M.set_player(R1)
path = M.connect(R1.east, R2.west)
path.door = M.new(type='d', name='wooden door')
path.door.add_property("open")
# shit = M.new('o', 'shit')
# shit_hole = M.new('c', 'shit hole')
# new_shit_hole = M.new('c', 'new shit hole')
# shit_hole.add(shit)
# room.add(shit_hole)
# room.add(new_shit_hole)
# # quest1 = ['open shit hole', 'take shit', 'open new shit hole', 'insert shit into new shit hole']
# # insert_shit = M.new_event_using_commands(quest1)
# win_event = Event(conditions={M.new_fact("in", shit, new_shit_hole)})
# quest1 = Quest(win_events=[win_event],
#                reward=2)
# M.quests = [quest1]



# Kitchen Funitures
## Important ones
garbage_can = M.new(type='c', name='garbage can')
garbage_can.add_property("open")

kitchen_pantry = M.new(type='c', name='kitchen pantry')
kitchen_pantry.add_property("open")

## Default ones
dining_table = M.new(type='s', name='dining table')
#
fridge = M.new(type='c', name='fridge')
# fridge.add_property("closed")

sink = M.new(type='c', name='sink')
sink.add_property("open")


# Bedroom
## Important ones
writing_desk = M.new(type='s', name='writing desk')
bedroom_cabinet = M.new(type='c', name='bedroom cabinet')
bedroom_cabinet.add_property("open")

## Default ones
# end_table = M.new(type='s', name='end table')
sofa = M.new(type='s', name='sofa')
bed = M.new(type='s', name='bed')
wardrobe = M.new(type='c', name='wardrobe')
wardrobe.add_property("open")

# Objects
used_noodle_cup = M.new('o', 'used noodle cup', 'an useless dirty noodle cup')
chili_oil = M.new('o', 'chili oil', 'a bottle of normal chili oil')
computer_monitor = M.new('o', 'computer monitor', 'a nice computer monitor')
bag_of_cookies = M.new('f', 'bag of cookies', 'it seems delicious')
family_photo = M.new('o', 'family photo', 'a photo of the family')
game_software = M.new('o', 'game software', 'a box of game software')
apple_core = M.new('o', 'apple core', 'a leftover apple core')
## ## Random place objects
# random_location = ['bedroom ground', 'kitchen ground', 'garbage can', 'kitchen pantry', 'dining table', 'fridge', 'sink', 'writing desk', 'bedroom cabinet', 'end table', 'sofa', 'bed', 'wardrobe']
# import numpy as np
# np.random.seed(2024)
# np.random.choice(random_location, 7)
# -> ['bedroom cabinet', 'sofa', 'bedroom ground', 'bedroom ground', 'bed', 'dining table', 'writing desk']

bedroom_cabinet.add(used_noodle_cup)
sofa.add(chili_oil)
R1.add(computer_monitor)
R1.add(bag_of_cookies)
bed.add(family_photo)
dining_table.add(game_software)
writing_desk.add(apple_core)

# Add container to room
R1.add(wardrobe)
R1.add(bed)
R1.add(sofa)
# R1.add(end_table)
R1.add(writing_desk)
R1.add(bedroom_cabinet)
R2.add(garbage_can)
R2.add(kitchen_pantry)
R2.add(dining_table)
R2.add(fridge)
R2.add(sink)

# Quests
quests = []
used_noodle_cup_in_garbage_can = Event(conditions={M.new_fact("in", used_noodle_cup, garbage_can)})
quests.append(Quest(win_events=[used_noodle_cup_in_garbage_can], reward=1))
chili_oil_in_kitchen_pantry = Event(conditions={M.new_fact("in", chili_oil, kitchen_pantry)})
chili_oil_on_dining_table = Event(conditions={M.new_fact("on", chili_oil, dining_table)})
quests.append(Quest(win_events=[chili_oil_in_kitchen_pantry, chili_oil_on_dining_table], reward=1))
computer_monitor_on_writing_desk = Event(conditions={M.new_fact("on", computer_monitor, writing_desk)})
quests.append(Quest(win_events=[computer_monitor_on_writing_desk], reward=1))
a_bag_of_cookies_in_kitchen_pantry = Event(conditions={M.new_fact("in", bag_of_cookies, kitchen_pantry)})
quests.append(Quest(win_events=[a_bag_of_cookies_in_kitchen_pantry], reward=1))
a_family_photo_in_bedroom_cabinet = Event(conditions={M.new_fact("in", family_photo, bedroom_cabinet)})
a_family_photo_on_writing_desk = Event(conditions={M.new_fact("on", family_photo, writing_desk)})
quests.append(Quest(win_events=[a_family_photo_in_bedroom_cabinet, a_family_photo_on_writing_desk], reward=1))
game_software_on_writing_desk = Event(conditions={M.new_fact("on", game_software, writing_desk)})
game_software_in_cabinet = Event(conditions={M.new_fact("in", game_software, bedroom_cabinet)})
quests.append(Quest(win_events=[game_software_on_writing_desk, game_software_in_cabinet], reward=1))
apple_core_in_garbage_can = Event(conditions={M.new_fact("in", apple_core, garbage_can)})
quests.append(Quest(win_events=[apple_core_in_garbage_can], reward=1))
## Set quests
M.quests = quests

# M.compile('/home/taku/research/zhuobinggang/customized_twc_games')
