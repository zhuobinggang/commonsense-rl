简化action list。

## 2024.12.4：问题

简化action list理所当然地会遇到输出的action无法被环境处理

> Action history: Action 0: hang wet green T-shirt on clothesline -> ("That's not a verb I recognise.\n",) Action 1: put wet green T-shirt on clothesline -> You put the wet green T-shirt on the clothesline.Your score has just gone up by one point. Action 2: take wet black shirt from patio table -> You take the wet black shirt from the patio table. Action 3: put wet black shirt on clothesline -> You put the wet black shirt on the clothesline.Your score has just gone up by one point. Action 4: take wet gray dress from patio chair -> You take the wet gray dress from the patio chair. Action 5: put wet gray dress on clothesline -> You put the wet gray dress on the clothesline.Your score has just gone up by one point. Action 6: take wet plaid blazer from floor -> ("You can't see any such thing.\n",) Action 7: take wet striped blazer from floor -> ("You can't see any such thing.\n",) Action 8: take wet plaid blazer from floor -> ("You can't see any such thing.\n",) 

-> 解决方案：实现一个action selector，从环境列表中获取最接近的action