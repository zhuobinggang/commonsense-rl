## 4omini_ours(lack datas)

因为4omini性能不强，没办法很好地根据template来组装正确的command，所以没有办法，这个只能放弃。但是可以尝试只精简description但是保留所有action list。

## 4omini_simple_desc_only

性能反而下降了，gpt4o mini简直是天才。观察了一下，是因为经常调用close container，导致put things in container从选项列表中消失。

## 