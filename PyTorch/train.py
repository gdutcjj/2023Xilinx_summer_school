import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 训练数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()  # 对样本数据进行处理，转换为张量数据
)
# 测试数据集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()  # 对样本数据进行处理，转换为张量数据
)
# 标签字典，一个key键对应一个label
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# # 设置画布大小
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     # 随机生成一个索引
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     # 获取样本及其对应的标签
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     # 设置标题
#     plt.title(labels_map[label])
#     # 不显示坐标轴
#     plt.axis("off")
#     # 显示灰度图
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# 训练数据加载器
train_dataloader = DataLoader(
    dataset=training_data,
    # 设置批量大小
    batch_size=256,
    # 打乱样本的顺序
    shuffle=True)
# 测试数据加载器
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=256,
    shuffle=True)


# 展示图片和标签
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class LeNet(nn.Module):  # 定义网络 pytorch定义网络有很多方式，推荐以下方式，结构清晰
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2，图片大小变为 28+2*2 = 32 (两边各加2列0)，保证输入输出尺寸相同
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)  # input_size=(6*28*28)，output_size=(6*14*14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # input_size=(6*14*14)，output_size=16*10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)  ##input_size=(16*10*10)，output_size=(16*5*5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5),  # input_size=(16*5*5)，output_size=120*1*1
            nn.ReLU(),
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(16*5*5,120),
        #     nn.ReLU()
        # )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)

    # 网络前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        # x = self.fc1(x)

        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # 前向传播，计算预测值
        pred = model(X)
        # 计算损失
        loss = loss_fn(pred, y)
        # 反向传播，优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 测试模型性能
def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            # 前向传播，计算预测值
            pred = model(X)
            # 计算损失
            test_loss += loss_fn(pred, y).item()
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 定义模型
    # model = NeuralNetwork().to(device)
    model = LeNet().to(device)
    # 设置超参数
    learning_rate = 1e-2
    batch_size = 64
    epochs = 80
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    # 训练模型
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")
    # 保存模型
    torch.save(model.state_dict(), 'mylenet.pth')