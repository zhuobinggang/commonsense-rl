因为one-shot prompt太长所以对example进行了剪枝。而遇到go east & go west的情况，因为observation和description一致，没有必要重复，所以将obs在第一个句号之前截断，以这样的方式保持prompt的长度适合。
