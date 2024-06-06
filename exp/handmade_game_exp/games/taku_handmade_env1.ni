Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_1 and the r_0 are rooms.

Understand "small kitchen" as r_1.
The internal name of r_1 is "small kitchen".
The printed name of r_1 is "-= Small Kitchen =-".
The small kitchen part 0 is some text that varies. The small kitchen part 0 is "You find yourself in a small kitchen. An ordinary one.

 You see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The small kitchen part 1 is some text that varies. The small kitchen part 1 is " garbage can.[if c_0 is open and there is something in the c_0] The garbage can contains [a list of things in the c_0].[end if]".
The small kitchen part 2 is some text that varies. The small kitchen part 2 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The small kitchen part 3 is some text that varies. The small kitchen part 3 is " You see a kitchen pantry.[if c_1 is open and there is something in the c_1] The kitchen pantry contains [a list of things in the c_1].[end if]".
The small kitchen part 4 is some text that varies. The small kitchen part 4 is "[if c_1 is open and the c_1 contains nothing] The kitchen pantry is empty! This is the worst thing that could possibly happen, ever![end if]".
The small kitchen part 5 is some text that varies. The small kitchen part 5 is " You can make out a fridge.[if c_2 is open and there is something in the c_2] The fridge contains [a list of things in the c_2]![end if]".
The small kitchen part 6 is some text that varies. The small kitchen part 6 is "[if c_2 is open and the c_2 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The small kitchen part 7 is some text that varies. The small kitchen part 7 is " You can see a sink.[if c_3 is open and there is something in the c_3] The sink contains [a list of things in the c_3].[end if]".
The small kitchen part 8 is some text that varies. The small kitchen part 8 is "[if c_3 is open and the c_3 contains nothing] The sink is empty, what a horrible day![end if]".
The small kitchen part 9 is some text that varies. The small kitchen part 9 is " You can make out a dining table. [if there is something on the s_0]You see [a list of things on the s_0] on the dining table.[end if]".
The small kitchen part 10 is some text that varies. The small kitchen part 10 is "[if there is nothing on the s_0]The dining table appears to be empty. Hm. Oh well[end if]".
The small kitchen part 11 is some text that varies. The small kitchen part 11 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The small kitchen part 12 is some text that varies. The small kitchen part 12 is " wooden door leading west.".
The description of r_1 is "[small kitchen part 0][small kitchen part 1][small kitchen part 2][small kitchen part 3][small kitchen part 4][small kitchen part 5][small kitchen part 6][small kitchen part 7][small kitchen part 8][small kitchen part 9][small kitchen part 10][small kitchen part 11][small kitchen part 12]".

west of r_1 and east of r_0 is a door called d_0.
Understand "small bedroom" as r_0.
The internal name of r_0 is "small bedroom".
The printed name of r_0 is "-= Small Bedroom =-".
The small bedroom part 0 is some text that varies. The small bedroom part 0 is "You've entered a small bedroom. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.

 You can make out a bedroom cabinet.[if c_4 is open and there is something in the c_4] The bedroom cabinet contains [a list of things in the c_4].[end if]".
The small bedroom part 1 is some text that varies. The small bedroom part 1 is "[if c_4 is open and the c_4 contains nothing] The bedroom cabinet is empty, what a horrible day![end if]".
The small bedroom part 2 is some text that varies. The small bedroom part 2 is " You can see a wardrobe.[if c_5 is open and there is something in the c_5] The wardrobe contains [a list of things in the c_5]![end if]".
The small bedroom part 3 is some text that varies. The small bedroom part 3 is "[if c_5 is open and the c_5 contains nothing] The wardrobe is empty, what a horrible day![end if]".
The small bedroom part 4 is some text that varies. The small bedroom part 4 is " You see a writing desk. The writing desk is usual.[if there is something on the s_1] On the writing desk you make out [a list of things on the s_1].[end if]".
The small bedroom part 5 is some text that varies. The small bedroom part 5 is "[if there is nothing on the s_1] But the thing hasn't got anything on it.[end if]".
The small bedroom part 6 is some text that varies. The small bedroom part 6 is " You can see a sofa. I guess it's true what they say, if you're looking for a sofa, go to TextWorld. The sofa is normal.[if there is something on the s_2] On the sofa you can make out [a list of things on the s_2]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The small bedroom part 7 is some text that varies. The small bedroom part 7 is "[if there is nothing on the s_2] But the thing hasn't got anything on it. It would have been so cool if there was stuff on the sofa! oh well.[end if]".
The small bedroom part 8 is some text that varies. The small bedroom part 8 is " You see a bed. The bed is standard.[if there is something on the s_3] On the bed you can see [a list of things on the s_3].[end if]".
The small bedroom part 9 is some text that varies. The small bedroom part 9 is "[if there is nothing on the s_3] The bed appears to be empty. You move on, clearly infuriated by your TextWorld experience.[end if]".
The small bedroom part 10 is some text that varies. The small bedroom part 10 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The small bedroom part 11 is some text that varies. The small bedroom part 11 is " wooden door leading east.".
The description of r_0 is "[small bedroom part 0][small bedroom part 1][small bedroom part 2][small bedroom part 3][small bedroom part 4][small bedroom part 5][small bedroom part 6][small bedroom part 7][small bedroom part 8][small bedroom part 9][small bedroom part 10][small bedroom part 11]".

east of r_0 and west of r_1 is a door called d_0.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are privately-named.
The d_0 are doors.
The d_0 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The o_2 and the o_0 and the o_1 and the o_3 and the o_4 and the o_5 are object-likes.
The o_2 and the o_0 and the o_1 and the o_3 and the o_4 and the o_5 are privately-named.
The r_1 and the r_0 are rooms.
The r_1 and the r_0 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 are privately-named.

The description of d_0 is "The wooden door looks hefty. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "wooden door".
Understand "wooden door" as d_0.
Understand "wooden" as d_0.
Understand "door" as d_0.
The d_0 is open.
The description of c_0 is "The garbage can looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "garbage can".
Understand "garbage can" as c_0.
Understand "garbage" as c_0.
Understand "can" as c_0.
The c_0 is in r_1.
The c_0 is open.
The description of c_1 is "The kitchen pantry looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "kitchen pantry".
Understand "kitchen pantry" as c_1.
Understand "kitchen" as c_1.
Understand "pantry" as c_1.
The c_1 is in r_1.
The c_1 is open.
The description of c_2 is "The fridge looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "fridge".
Understand "fridge" as c_2.
The c_2 is in r_1.
The description of c_3 is "The sink looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_3 is "sink".
Understand "sink" as c_3.
The c_3 is in r_1.
The c_3 is open.
The description of c_4 is "The bedroom cabinet looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_4 is "bedroom cabinet".
Understand "bedroom cabinet" as c_4.
Understand "bedroom" as c_4.
Understand "cabinet" as c_4.
The c_4 is in r_0.
The c_4 is open.
The description of c_5 is "The wardrobe looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_5 is "wardrobe".
Understand "wardrobe" as c_5.
The c_5 is in r_0.
The c_5 is open.
The description of f_0 is "You couldn't pay me to eat that standard thing.".
The printed name of f_0 is "bag of cookies".
Understand "bag of cookies" as f_0.
Understand "bag" as f_0.
Understand "cookies" as f_0.
The f_0 is in r_0.
The description of o_2 is "The computer monitor is antiquated.".
The printed name of o_2 is "computer monitor".
Understand "computer monitor" as o_2.
Understand "computer" as o_2.
Understand "monitor" as o_2.
The o_2 is in r_0.
The description of s_0 is "The dining table is reliable.".
The printed name of s_0 is "dining table".
Understand "dining table" as s_0.
Understand "dining" as s_0.
Understand "table" as s_0.
The s_0 is in r_1.
The description of s_1 is "The writing desk is solidly built.".
The printed name of s_1 is "writing desk".
Understand "writing desk" as s_1.
Understand "writing" as s_1.
Understand "desk" as s_1.
The s_1 is in r_0.
The description of s_2 is "The sofa is durable.".
The printed name of s_2 is "sofa".
Understand "sofa" as s_2.
The s_2 is in r_0.
The description of s_3 is "The bed is shaky.".
The printed name of s_3 is "bed".
Understand "bed" as s_3.
The s_3 is in r_0.
The description of o_0 is "The used noodle cup is modern.".
The printed name of o_0 is "used noodle cup".
Understand "used noodle cup" as o_0.
Understand "used" as o_0.
Understand "noodle" as o_0.
Understand "cup" as o_0.
The o_0 is in the c_4.
The description of o_1 is "The chili oil is antiquated.".
The printed name of o_1 is "chili oil".
Understand "chili oil" as o_1.
Understand "chili" as o_1.
Understand "oil" as o_1.
The o_1 is on the s_2.
The description of o_3 is "The family photo would seem to be well matched to everything else here".
The printed name of o_3 is "family photo".
Understand "family photo" as o_3.
Understand "family" as o_3.
Understand "photo" as o_3.
The o_3 is on the s_3.
The description of o_4 is "The game software appears to be out of place here".
The printed name of o_4 is "game software".
Understand "game software" as o_4.
Understand "game" as o_4.
Understand "software" as o_4.
The o_4 is on the s_0.
The description of o_5 is "The apple core is well-used.".
The printed name of o_5 is "apple core".
Understand "apple core" as o_5.
Understand "apple" as o_5.
Understand "core" as o_5.
The o_5 is on the s_1.


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with ""

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The o_0 is in the c_0:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with ""

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The o_1 is in the c_1:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with ""

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The o_2 is on the s_1:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest2 completed is true;

The quest3 completed is a truth state that varies.
The quest3 completed is usually false.

Test quest3_0 with ""

Every turn:
	if quest3 completed is true:
		do nothing;
	else if The f_0 is in the c_1:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest3 completed is true;

The quest4 completed is a truth state that varies.
The quest4 completed is usually false.

Test quest4_0 with ""

Every turn:
	if quest4 completed is true:
		do nothing;
	else if The o_3 is in the c_4:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest4 completed is true;

The quest5 completed is a truth state that varies.
The quest5 completed is usually false.

Test quest5_0 with ""

Every turn:
	if quest5 completed is true:
		do nothing;
	else if The o_4 is on the s_1:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest5 completed is true;

The quest6 completed is a truth state that varies.
The quest6 completed is usually false.

Test quest6_0 with ""

Every turn:
	if quest6 completed is true:
		do nothing;
	else if The o_5 is in the c_0:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest6 completed is true;

Use scoring. The maximum score is 7.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		if V > 0:
			say "Your score has just gone up by [V in words] ";
		else:
			say "Your score changed by [V in words] ";
		if V >= -1 and V <= 1:
			say "point.";
		else:
			say "points.";
		Now the last notified score is the score;
	if quest0 completed is true and quest1 completed is true and quest2 completed is true and quest3 completed is true and quest4 completed is true and quest5 completed is true and quest6 completed is true:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	if maximum score is -32768:
		say "You scored a total of [score] point[s], in [turn count] turn[s].";
	else:
		say "You scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Carry out requesting the score:
	if maximum score is -32768:
		say "You have so far scored [score] point[s], in [turn count] turn[s].";
	else:
		say "You have so far scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;


An objective is some text that varies. The objective is "".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	if maximum score is -32768:
		say "infinity";
	else:
		say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

