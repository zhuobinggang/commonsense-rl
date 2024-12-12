
################### taku added 2024.5.22 START ###################

ONE_SHOT_EXP_AUGMENT = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point.Right position. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point.Right position. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point.Right position. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point.Right position. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point.Right position. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point.Right position. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_NO_AUG = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point."""

ONE_SHOT_EXP_AUGMENT_SIMPLE = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_SIMPLE_NO_AUG = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point."""

QUESTION_TEMPLATE_NO_COT = """Question: To put things in there proper locations and improve your score, what should you do? Choose 'one' action from above list.
Next action: <fill in>"""

QUESTION_TEMPLATE = """Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""

TASK = 'You are a experienced text game player, your goal is put things in there proper locations and improve your score.'
QUESTION = "To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list."
QUESTION_NO_COT = "To put things in there proper locations and improve your score, what should you do? Choose 'one' action from above list."
FILL_IN_TEMPLATE = '<fill in>'

########################
GPT4 = 'gpt-4-1106-preview'
GPT35 = 'gpt-3.5-turbo-0613'
GPT4o = 'gpt-4o-2024-05-13'


##################### 2024.11.13

TASK_NEW = 'You are a text game player, your goal is to put things in their proper locations and improve your score. You can only perform one action at a time. Answer in json form: {"consideration": string, "action": string}.'
TASK_FINETUNE = 'You are a text game player, your goal is to put things in their proper locations and improve your score. You can only perform one action at a time.'
TASK_COT_DISTLL = 'You are a text game player, your goal is to put things in their proper locations and improve your score. You can only perform one action at a time. You have already taken the best action based on the current situation. Now, you need to explain the reasons behind your choice.'

TASK_FTWP = 'Your task is to find and examine the recipe, then prepare the ingredients as required, prepare the meal and eat it.'