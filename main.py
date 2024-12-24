# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import struct
#
#
# # 定义解析 idx 文件的函数
# def load_idx(file_path):
#     with open(file_path, 'rb') as f:
#         magic, num = struct.unpack(">II", f.read(8))
#         if magic == 2051:  # 图像文件
#             rows, cols = struct.unpack(">II", f.read(8))
#             data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
#         elif magic == 2049:  # 标签文件
#             data = np.fromfile(f, dtype=np.uint8)
#         else:
#             raise ValueError(f"Unknown magic number: {magic}")
#     return data
#
#
# # 自定义 Dataset 类
# class MNISTDataset(Dataset):
#     def __init__(self, images_path, labels_path):
#         self.images = load_idx(images_path).astype(np.float32) / 255.0  # 归一化
#         self.labels = load_idx(labels_path).astype(np.int64)
#         self.images = self.images.reshape(-1, 28 * 28)  # 展平
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]
#
#
# # 加载数据
# train_dataset = MNISTDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
# test_dataset = MNISTDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#
# # 定义模型
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.hidden = nn.Linear(28 * 28, 128)  # 隐藏层
#         self.output = nn.Linear(128, 10)  # 输出层
#
#     def forward(self, x):
#         x = torch.relu(self.hidden(x))  # ReLU 激活
#         x = self.output(x)  # 输出层
#         return x
#
#
# # 初始化模型、损失函数和优化器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleNN().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 训练模型
# def train_model():
#     model.train()
#     for epoch in range(5):  # 训练 5 个 epoch
#         total_loss = 0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # 计算损失
#             total_loss += loss.item()
#
#             # 计算准确率
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         # 打印损失和准确率
#         accuracy = 100 * correct / total
#         print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
#
# # 测试模型
# def test_model():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f"Accuracy on test set: {100 * correct / total:.2f}%")
#
#
# # 运行训练和测试
# train_model()
# test_model()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import struct
import matplotlib.pyplot as plt  # 导入绘图库

# 定义解析 idx 文件的函数
def load_idx(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic == 2051:  # 图像文件
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        elif magic == 2049:  # 标签文件
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number: {magic}")
    return data


# 自定义 Dataset 类
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = load_idx(images_path).astype(np.float32) / 255.0  # 归一化
        self.labels = load_idx(labels_path).astype(np.int64)
        self.images = self.images.reshape(-1, 28 * 28)  # 展平

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# 加载数据
train_dataset = MNISTDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
test_dataset = MNISTDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28 * 28, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # ReLU 激活
        x = self.output(x)  # 输出层
        return x


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model():
    accuracy_list = []  # 用于记录每个epoch的准确率
    model.train()
    for epoch in range(5):  # 训练 5 个 epoch
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算损失
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算当前epoch的准确率
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)  # 将每个epoch的准确率加入列表
        print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return accuracy_list  # 返回准确率列表


# 测试模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")


# 运行训练和测试
accuracy_list = train_model()  # 获取训练过程中每个epoch的准确率
test_model()

# 绘制准确率随轮次变化的图
plt.plot(range(1, 6), accuracy_list, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Traditional Linear Layer')
plt.xticks(range(1, 6))
plt.grid(True)
plt.show()
