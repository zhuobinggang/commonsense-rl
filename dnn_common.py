import torch
import torch.nn as nn

# 3層のNNを定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(4, 5)  # 第一層
        self.layer2 = nn.Linear(5, 3)  # 第二層
        self.layer3 = nn.Linear(3, 1)  # 第三層

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x)
        return x

def test():
    # モデルの作成
    model = SimpleNN()

    # 第二層の requires_grad を False に設定
    for param in model.layer2.parameters():
        param.requires_grad = False

    # ダミーデータ
    x = torch.randn(2, 4)
    y = torch.randn(2, 1)

    # 前方計算
    output = model(x)
    loss = (output - y).sum()

    # 逆伝播
    loss.backward()

    # 各層の勾配を確認
    print(model.layer1.weight.grad)  # 勾配あり
    print(model.layer2.weight.grad)  # None (計算されない)
    print(model.layer3.weight.grad)  # 勾配あり