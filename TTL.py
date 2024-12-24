# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import struct
#
#
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
# class MNISTDataset(Dataset):
#     def __init__(self, images_path, labels_path):
#         self.images = load_idx(images_path).astype(np.float32) / 255.0
#         self.labels = load_idx(labels_path).astype(np.int64)
#         self.images = self.images.reshape(-1, 28 * 28)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]
#
#
# class TuckerTensorLayer(nn.Module):
#     def __init__(self, input_shape, hidden_dim, ranks):
#         super(TuckerTensorLayer, self).__init__()
#         """
#         input_shape: [28, 28]
#         hidden_dim: 隐藏层神经元数量 (128)
#         ranks: [10, 5, 5] (rank1, rank2, rank3)
#         """
#         self.input_shape = input_shape
#         self.hidden_dim = hidden_dim
#         self.ranks = ranks
#
#         # U3: [hidden_dim, rank1] (128×10)
#         self.U3 = nn.Parameter(torch.randn(hidden_dim, ranks[0]) / np.sqrt(hidden_dim))
#
#         # Core: [rank1, rank2*rank3] (10×25)
#         self.core = nn.Parameter(torch.randn(ranks[0], ranks[1] * ranks[2]) / np.sqrt(ranks[0] * ranks[1] * ranks[2]))
#
#         # U2: [input_shape[1], rank2] (28×5)
#         self.U2 = nn.Parameter(torch.randn(input_shape[1], ranks[1]) / np.sqrt(input_shape[1]))
#
#         # U1: [input_shape[0], rank3] (28×5)
#         self.U1 = nn.Parameter(torch.randn(input_shape[0], ranks[2]) / np.sqrt(input_shape[0]))
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         # x shape: [batch_size, 784]
#         x = x.reshape(batch_size, self.input_shape[0], self.input_shape[1])
#
#         # 计算 U2⊗U1
#         # 首先计算kronecker product
#         kron_prod = torch.kron(self.U2, self.U1)  # shape: [784, 25]
#
#         # 重塑输入为 [batch_size, 784]
#         x = x.reshape(batch_size, -1)
#
#         # 完整计算: y = U3 * Core * (U2⊗U1)^T * x
#         # 1. (U2⊗U1)^T * x
#         temp = torch.matmul(x, kron_prod)  # [batch_size, 25]
#
#         # 2. Core * ((U2⊗U1)^T * x)
#         temp = torch.matmul(temp, self.core.t())  # [batch_size, 10]
#
#         # 3. U3 * (Core * ((U2⊗U1)^T * x))
#         output = torch.matmul(temp, self.U3.t())  # [batch_size, 128]
#
#         return output
#
#
# class TuckerNN(nn.Module):
#     def __init__(self, input_shape, hidden_dim, output_dim, ranks):
#         super(TuckerNN, self).__init__()
#         self.ttl = TuckerTensorLayer(input_shape, hidden_dim, ranks)
#         self.fc = nn.Linear(hidden_dim, output_dim)  # 从隐藏层到输出层的普通线性映射
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.ttl(x)
#         x = self.relu(x)
#         x = self.fc(x)
#         return x
#
#
# # 初始化模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ranks = [10, 5, 5]  # [rank1, rank2, rank3]
# model = TuckerNN(
#     input_shape=[28, 28],
#     hidden_dim=128,
#     output_dim=10,
#     ranks=ranks
# ).to(device)
#
# # 数据加载
# train_dataset = MNISTDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
# test_dataset = MNISTDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# def train_model():
#     model.train()
#     for epoch in range(5):
#         total_loss = 0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         accuracy = 100 * correct / total
#         print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
#
# def test_model():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f"Accuracy on test set: {100 * correct / total:.2f}%")
#
#
# # 运行训练和测试
# train_model()
# test_model()

#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import struct
# import matplotlib.pyplot as plt  # 导入绘图库
#
#
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
# class MNISTDataset(Dataset):
#     def __init__(self, images_path, labels_path):
#         self.images = load_idx(images_path).astype(np.float32) / 255.0
#         self.labels = load_idx(labels_path).astype(np.int64)
#         self.images = self.images.reshape(-1, 28 * 28)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]
#
#
# class TuckerTensorLayer(nn.Module):
#     def __init__(self, input_shape, hidden_dim, ranks):
#         super(TuckerTensorLayer, self).__init__()
#         self.input_shape = input_shape
#         self.hidden_dim = hidden_dim
#         self.ranks = ranks
#
#         self.U3 = nn.Parameter(torch.randn(hidden_dim, ranks[0]) / np.sqrt(hidden_dim))
#         self.core = nn.Parameter(torch.randn(ranks[0], ranks[1] * ranks[2]) / np.sqrt(ranks[0] * ranks[1] * ranks[2]))
#         self.U2 = nn.Parameter(torch.randn(input_shape[1], ranks[1]) / np.sqrt(input_shape[1]))
#         self.U1 = nn.Parameter(torch.randn(input_shape[0], ranks[2]) / np.sqrt(input_shape[0]))
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.reshape(batch_size, self.input_shape[0], self.input_shape[1])
#
#         kron_prod = torch.kron(self.U2, self.U1)
#         x = x.reshape(batch_size, -1)
#         temp = torch.matmul(x, kron_prod)
#         temp = torch.matmul(temp, self.core.t())
#         output = torch.matmul(temp, self.U3.t())
#
#         return output
#
#
# class TuckerNN(nn.Module):
#     def __init__(self, input_shape, hidden_dim, output_dim, ranks):
#         super(TuckerNN, self).__init__()
#         self.ttl = TuckerTensorLayer(input_shape, hidden_dim, ranks)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.ttl(x)
#         x = self.relu(x)
#         x = self.fc(x)
#         return x
#
#
# # 初始化模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ranks = [10, 5, 5]
# model = TuckerNN(input_shape=[28, 28], hidden_dim=128, output_dim=10, ranks=ranks).to(device)
#
# # 数据加载
# train_dataset = MNISTDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
# test_dataset = MNISTDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# def train_model():
#     accuracy_list = []  # 用于记录每个epoch的准确率
#     model.train()
#     for epoch in range(5):
#         total_loss = 0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         accuracy = 100 * correct / total
#         accuracy_list.append(accuracy)  # 将每个epoch的准确率加入列表
#         print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
#     return accuracy_list  # 返回准确率列表
#
#
# def test_model():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f"Accuracy on test set: {100 * correct / total:.2f}%")
#
#
# # 运行训练和测试
# accuracy_list = train_model()  # 获取训练过程中每个epoch的准确率
# test_model()
#
# # 绘制准确率随轮次变化的图
# plt.plot(range(1, 6), accuracy_list, marker='o', linestyle='-', color='b')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Tucker Tensor Layer')
# plt.xticks(range(1, 6))
# plt.grid(True)
# plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import struct
import matplotlib.pyplot as plt  # 导入绘图库


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


class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = load_idx(images_path).astype(np.float32) / 255.0
        self.labels = load_idx(labels_path).astype(np.int64)
        self.images = self.images.reshape(-1, 28 * 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class TuckerTensorLayer(nn.Module):
    def __init__(self, input_shape, hidden_dim, ranks):
        super(TuckerTensorLayer, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.ranks = ranks

        self.U3 = nn.Parameter(torch.randn(hidden_dim, ranks[0]) / np.sqrt(hidden_dim))
        self.core = nn.Parameter(torch.randn(ranks[0], ranks[1] * ranks[2]) / np.sqrt(ranks[0] * ranks[1] * ranks[2]))
        self.U2 = nn.Parameter(torch.randn(input_shape[1], ranks[1]) / np.sqrt(input_shape[1]))
        self.U1 = nn.Parameter(torch.randn(input_shape[0], ranks[2]) / np.sqrt(input_shape[0]))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.input_shape[0], self.input_shape[1])

        kron_prod = torch.kron(self.U2, self.U1)
        x = x.reshape(batch_size, -1)
        temp = torch.matmul(x, kron_prod)
        temp = torch.matmul(temp, self.core.t())
        output = torch.matmul(temp, self.U3.t())

        return output


class TuckerNN(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim, ranks):
        super(TuckerNN, self).__init__()
        self.ttl = TuckerTensorLayer(input_shape, hidden_dim, ranks)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ttl(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ranks = [10, 5, 5]
model = TuckerNN(input_shape=[28, 28], hidden_dim=128, output_dim=10, ranks=ranks).to(device)

# 数据加载
train_dataset = MNISTDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
test_dataset = MNISTDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()  # 用于混合精度训练


def train_model():
    accuracy_list = []  # 用于记录每个epoch的准确率
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 开启自动混合精度
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)  # 将每个epoch的准确率加入列表
        print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return accuracy_list  # 返回准确率列表


def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():  # 测试时也使用混合精度
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
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
plt.title('Tucker Tensor Layer')
plt.xticks(range(1, 6))
plt.grid(True)
plt.show()
