from functools import lru_cache

@lru_cache(maxsize=None)
def get_client():
    import anthropic
    client = anthropic.Anthropic()
    return client

def fake_message():
    system = """Task: You are a experienced text game player, your goal is put things in there proper locations and improve your score.
Example walkthrough: Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point.Right position. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point.Right position. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point.Right position. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point.Right position. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point.Right position. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point.Right position. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.Right position.
Action history: No action was taken now.
Inventory: You are carrying:  some milk
Current enviroment: -= Backyard =-You've entered a backyard.You see a BBQ. The BBQ is recent. On the BBQ you make out a wooden spoon. You see a clothesline. The clothesline is typical. But the thing is empty. Hm. Oh well What's that over there? It looks like it's a patio chair. Now why would someone leave that there? On the patio chair you can see a wet white jumper. You see a patio table. The patio table is stylish. The patio table appears to be empty. Hey, want to see a workbench? Look over there, a workbench. On the workbench you see a clean pot. Something scurries by right in the corner of your eye. Probably nothing.There is an open sliding patio door leading west.
"""
    user = """Action you can take:
* close sliding patio door
* drop milk
* examine BBQ
* examine clean pot
* examine clothesline
* examine patio chair
* examine patio table
* examine wet white jumper
* examine wooden spoon
* examine workbench
* go west
* look
* put milk on BBQ
* put milk on clothesline
* put milk on patio chair
* put milk on patio table
* put milk on workbench
* take clean pot from workbench
* take wet white jumper from patio chair
* take wooden spoon from BBQ

Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>
"""
    return system, user

def quest_claude(client, system, user, max_tokens = 1000):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user
                    }
                ]
            }
        ]
    )
    return message
