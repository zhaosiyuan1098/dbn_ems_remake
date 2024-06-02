import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from attention.ACmix import ACmix
from loader import Loader
from option import Option
from sklearn.metrics import confusion_matrix
def attention():
    # 检查并设置GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 加载数据
    option = Option()
    loader = Loader(option)
    x_train, x_valid, y_train, y_valid = loader.load_for_dbn()

    # 数据预处理：将数据转换为张量并进行适当的形状调整
    x_train_tensor = torch.Tensor(x_train).unsqueeze(1).unsqueeze(3)  # 在第二个维度上增加一个维度
    x_valid_tensor = torch.Tensor(x_valid).unsqueeze(1).unsqueeze(3)
    print(x_train_tensor.size(), x_valid_tensor.size())

    # 创建标签张量
    y_train_tensor = torch.LongTensor(y_train)
    y_valid_tensor = torch.LongTensor(y_valid)

    # 创建训练集和测试集的 TensorDataset 对象
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)

    # 创建批次加载器
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


    # 创建卷积神经网络模型
    class CNN(nn.Module):
        def __init__(self, input_shape, num_classes=12):
            super(CNN, self).__init__()

            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1),
                nn.Tanh(),
                nn.Conv2d(32, 64, kernel_size=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=1, stride=1),
                nn.Conv2d(64, 128, kernel_size=1),
                nn.Tanh(),
                nn.Dropout(p=0.5)
            )

            # 计算卷积层输出的形状
            self._to_linear = self._get_conv_output(input_shape)

            # 添加注意力机制模块
            self.ACmix = ACmix(in_planes=128, out_planes=128)

            self.fc = nn.Sequential(
                nn.Linear(self._to_linear, 84),  # 使用计算得到的形状
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(84, num_classes)
            )

        def _get_conv_output(self, shape):
            # 创建一个假的输入张量以获取卷积层输出的形状
            x = torch.rand(1, *shape)
            x = self.net(x)
            return int(np.prod(x.size()))

        def forward(self, x):
            x = self.net(x)
            x = self.ACmix(x)  # 应用注意力机制模块对特征图进行加权处理
            x = x.view(x.size(0), -1)  # 展平
            y = self.fc(x)
            return y


    # 获取输入形状
    input_shape = (1, 24, 1)

    # 查看网络结构
    X = torch.rand(size=(256, *input_shape))
    model = CNN(input_shape=input_shape)
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 实例化模型、损失函数和优化器
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # 使用 AdamW 优化器

    # 模型训练
    num_epochs = 100
    train_losses = []  # 记录训练损失变化的列表
    train_accuracies = []  # 记录训练准确率变化的列表
    val_losses = []  # 记录验证损失变化的列表
    val_accuracies = []  # 记录验证准确率变化的列表

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        for batch_data, batch_labels in train_dataloader:
            # 把数据和标签小批量读取到GPU上
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            train_losses.append(loss.item())  # 记录损失函数的变化

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)

        # 在验证集上评估模型
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_loss /= len(test_dataloader)
        val_losses.append(val_loss)
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        # 输出当前轮次的训练和验证损失与准确率
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")

    # 绘制损失函数和准确率图像
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

    # 模型测试（使用测试集）
    model.eval()
    test_correct = 0
    test_total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

            true_labels.extend(batch_labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # 计算准确率
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy}%")

    # 计算混淆矩阵

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 可视化混淆矩阵
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.0f')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    attention()




# 下面两个注释掉的程序都是进行数据增强防止过拟合的，但是效果不明显（基本没什么变化）

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import TensorDataset, DataLoader
# from torchvision import transforms
# from attention.ACmix import ACmix
# from loader import Loader
# from option import Option
#
# # 检查并设置GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)
#
# # 加载数据
# option = Option()
# loader = Loader(option)
# x_train, x_valid, y_train, y_valid = loader.load_for_dbn()
#
# # 数据预处理：将数据转换为张量并进行适当的形状调整
# x_train_tensor = torch.Tensor(x_train).unsqueeze(1).unsqueeze(3)  # 在第二个维度上增加一个维度
# x_valid_tensor = torch.Tensor(x_valid).unsqueeze(1).unsqueeze(3)
# print(x_train_tensor.size(), x_valid_tensor.size())
#
# # 创建标签张量
# y_train_tensor = torch.LongTensor(y_train)
# y_valid_tensor = torch.LongTensor(y_valid)
#
# # 数据增强变换
# train_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(10),
# ])
#
# # 创建训练集和测试集的 TensorDataset 对象并应用数据增强
# class AugmentedDataset(torch.utils.data.Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, label
#
# train_dataset = AugmentedDataset(x_train_tensor, y_train_tensor, transform=train_transforms)
# test_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
#
# # 创建批次加载器
# batch_size = 256
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
#
# # 创建卷积神经网络模型
# class CNN(nn.Module):
#     def __init__(self, input_shape, num_classes=12):
#         super(CNN, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.Tanh(),
#             nn.Dropout(p=0.5),
#             nn.AvgPool2d(kernel_size=1, stride=1),
#             nn.Conv2d(64, 128, kernel_size=1),
#             nn.Tanh(),
#             nn.Dropout(p=0.5)
#         )
#
#         # 计算卷积层输出的形状
#         self._to_linear = self._get_conv_output(input_shape)
#
#         # 添加注意力机制模块
#         self.ACmix = ACmix(in_planes=128, out_planes=128)
#
#         self.fc = nn.Sequential(
#             nn.Linear(self._to_linear, 84),  # 使用计算得到的形状
#             nn.Tanh(),
#             nn.Dropout(p=0.5),
#             nn.Linear(84, num_classes)
#         )
#
#     def _get_conv_output(self, shape):
#         # 创建一个假的输入张量以获取卷积层输出的形状
#         x = torch.rand(1, *shape)
#         x = self.net(x)
#         return int(np.prod(x.size()))
#
#     def forward(self, x):
#         x = self.net(x)
#         x = self.ACmix(x)  # 应用注意力机制模块对特征图进行加权处理
#         x = x.view(x.size(0), -1)  # 展平
#         y = self.fc(x)
#         return y
#
# # 获取输入形状
# input_shape = (1, 24, 1)
#
# # 查看网络结构
# X = torch.rand(size=(256, *input_shape))
# model = CNN(input_shape=input_shape)
# for layer in model.net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)
#
# # 实例化模型、损失函数和优化器
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.0001) # 使用 Adam 优化器
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 使用 AdamW 优化器
#
# # 模型训练
# num_epochs = 1000
# train_losses = []  # 记录训练损失变化的列表
# train_accuracies = []  # 记录训练准确率变化的列表
# val_losses = []  # 记录验证损失变化的列表
# val_accuracies = []  # 记录验证准确率变化的列表
#
# for epoch in range(num_epochs):
#     model.train()
#     train_correct = 0
#     train_total = 0
#     for batch_data, batch_labels in train_dataloader:
#         # 把数据和标签小批量读取到GPU上
#         batch_data = batch_data.to(device)
#         batch_labels = batch_labels.to(device)
#
#         # 前向传播
#         outputs = model(batch_data)
#         loss = criterion(outputs, batch_labels)
#         train_losses.append(loss.item())  # 记录损失函数的变化
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 计算训练准确率
#         _, predicted = torch.max(outputs.data, 1)
#         train_total += batch_labels.size(0)
#         train_correct += (predicted == batch_labels).sum().item()
#
#     train_accuracy = 100 * train_correct / train_total
#     train_accuracies.append(train_accuracy)
#
#     # 在验证集上评估模型
#     model.eval()
#     val_loss = 0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for batch_data, batch_labels in test_dataloader:
#             batch_data = batch_data.to(device)
#             batch_labels = batch_labels.to(device)
#
#             outputs = model(batch_data)
#             loss = criterion(outputs, batch_labels)
#             val_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             val_total += batch_labels.size(0)
#             val_correct += (predicted == batch_labels).sum().item()
#
#     val_loss /= len(test_dataloader)
#     val_losses.append(val_loss)
#     val_accuracy = 100 * val_correct / val_total
#     val_accuracies.append(val_accuracy)
#
#     # 输出当前轮次的训练和验证损失与准确率
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")
#
# # 绘制损失函数和准确率图像
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
# plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
# plt.plot(range(num_epochs), val_accuracies, label='Val Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy')
# plt.legend()
#
# plt.show()
#
# # 模型测试（使用测试集）
# model.eval()
# test_correct = 0
# test_total = 0
# true_labels = []
# predicted_labels = []
#
# with torch.no_grad():
#     for batch_data, batch_labels in test_dataloader:
#         batch_data = batch_data.to(device)
#         batch_labels = batch_labels.to(device)
#
#         outputs = model(batch_data)
#         _, predicted = torch.max(outputs.data, 1)
#         test_total += batch_labels.size(0)
#         test_correct += (predicted == batch_labels).sum().item()
#
#         true_labels.extend(batch_labels.cpu().numpy())
#         predicted_labels.extend(predicted.cpu().numpy())
#
# # 计算准确率
# test_accuracy = 100 * test_correct / test_total
# print(f"Test Accuracy: {test_accuracy}%")
#
# # 计算混淆矩阵
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# print("Confusion Matrix:")
# print(conf_matrix)
#
# # 可视化混淆矩阵
# plt.figure(figsize=(12, 7))
# ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.0f')
# ax.set_xlabel('Predicted Labels')
# ax.set_ylabel('True Labels')
# ax.set_title('Confusion Matrix')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.show()
#
# # 保存模型
# torch.save(model, 'CNN1.pth')



# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# import matplotlib.pyplot as plt
# import seaborn as sns
# from loader import Loader
# from option import Option
# from attention.ACmix import ACmix
#
# # 检查并设置GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)
#
# # 加载数据
# option = Option()
# loader = Loader(option)
# x_train, x_valid, y_train, y_valid = loader.load_for_dbn()
#
# # 数据预处理：将数据转换为张量并进行适当的形状调整
# x_train_tensor = torch.Tensor(x_train).unsqueeze(1).unsqueeze(3)  # 在第二个维度上增加一个维度
# x_valid_tensor = torch.Tensor(x_valid).unsqueeze(1).unsqueeze(3)
# print(x_train_tensor.size(), x_valid_tensor.size())
#
# # 创建标签张量
# y_train_tensor = torch.LongTensor(y_train)
# y_valid_tensor = torch.LongTensor(y_valid)
#
# # 创建时间序列数据增强函数
# def time_masking(x, mask_ratio=0.1):
#     """时间遮蔽：随机选择一段时间并将其置零"""
#     length = x.shape[-1]
#     mask_length = int(length * mask_ratio)
#     mask_start = np.random.randint(0, length - mask_length)
#     x[..., mask_start:mask_start+mask_length] = 0
#     return x
#
# def amplitude_perturbation(x, noise_level=0.05):
#     """幅度扰动：在时间序列中添加随机噪声"""
#     noise = torch.randn_like(x) * noise_level
#     return x + noise
#
# def time_shift(x, shift_max=2):
#     """时间偏移：将时间序列整体前移或后移一定的时间步长"""
#     shift = np.random.randint(-shift_max, shift_max)
#     return torch.roll(x, shifts=shift, dims=-1)
#
# def signal_flipping(x):
#     """信号翻转：将时间序列数据翻转"""
#     return torch.flip(x, dims=[-1])
#
# # 创建训练集和测试集的 Dataset 对象并应用数据增强
# class AugmentedDataset(Dataset):
#     def __init__(self, data, labels, augmentations=None):
#         self.data = data
#         self.labels = labels
#         self.augmentations = augmentations
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         if self.augmentations:
#             for aug in self.augmentations:
#                 sample = aug(sample)
#         return sample, label
#
# augmentations = [time_masking, amplitude_perturbation, time_shift, signal_flipping]
# train_dataset = AugmentedDataset(x_train_tensor, y_train_tensor, augmentations=augmentations)
# test_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
#
# # 创建批次加载器
# batch_size = 256
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
#
# # 创建卷积神经网络模型
# class CNN(nn.Module):
#     def __init__(self, input_shape, num_classes=12):
#         super(CNN, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.Tanh(),
#             nn.Dropout(p=0.5),
#             nn.AvgPool2d(kernel_size=1, stride=1),
#             nn.Conv2d(64, 128, kernel_size=1),
#             nn.Tanh(),
#             nn.Dropout(p=0.5)
#         )
#
#         # 计算卷积层输出的形状
#         self._to_linear = self._get_conv_output(input_shape)
#
#         # 添加注意力机制模块
#         self.ACmix = ACmix(in_planes=128, out_planes=128)
#
#         self.fc = nn.Sequential(
#             nn.Linear(self._to_linear, 84),  # 使用计算得到的形状
#             nn.Tanh(),
#             nn.Dropout(p=0.5),
#             nn.Linear(84, num_classes)
#         )
#
#     def _get_conv_output(self, shape):
#         # 创建一个假的输入张量以获取卷积层输出的形状
#         x = torch.rand(1, *shape)
#         x = self.net(x)
#         return int(np.prod(x.size()))
#
#     def forward(self, x):
#         x = self.net(x)
#         x = self.ACmix(x)  # 应用注意力机制模块对特征图进行加权处理
#         x = x.view(x.size(0), -1)  # 展平
#         y = self.fc(x)
#         return y
#
# # 获取输入形状
# input_shape = (1, 24, 1)
#
# # 查看网络结构
# X = torch.rand(size=(256, *input_shape))
# model = CNN(input_shape=input_shape)
# for layer in model.net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)
#
# # 实例化模型、损失函数和优化器
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 使用 AdamW 优化器
#
# # 模型训练
# num_epochs = 100
# train_losses = []  # 记录训练损失变化的列表
# train_accuracies = []  # 记录训练准确率变化的列表
# val_losses = []  # 记录验证损失变化的列表
# val_accuracies = []  # 记录验证准确率变化的列表
#
# for epoch in range(num_epochs):
#     model.train()
#     train_correct = 0
#     train_total = 0
#     for batch_data, batch_labels in train_dataloader:
#         # 把数据和标签小批量读取到GPU上
#         batch_data = batch_data.to(device)
#         batch_labels = batch_labels.to(device)
#
#         # 前向传播
#         outputs = model(batch_data)
#         loss = criterion(outputs, batch_labels)
#         train_losses.append(loss.item())  # 记录损失函数的变化
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 计算训练准确率
#         _, predicted = torch.max(outputs.data, 1)
#         train_total += batch_labels.size(0)
#         train_correct += (predicted == batch_labels).sum().item()
#
#     train_accuracy = 100 * train_correct / train_total
#     train_accuracies.append(train_accuracy)
#
#     # 在验证集上评估模型
#     model.eval()
#     val_loss = 0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for batch_data, batch_labels in test_dataloader:
#             batch_data = batch_data.to(device)
#             batch_labels = batch_labels.to(device)
#
#             outputs = model(batch_data)
#             loss = criterion(outputs, batch_labels)
#             val_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             val_total += batch_labels.size(0)
#             val_correct += (predicted == batch_labels).sum().item()
#
#     val_loss /= len(test_dataloader)
#     val_losses.append(val_loss)
#     val_accuracy = 100 * val_correct / val_total
#     val_accuracies.append(val_accuracy)
#
#     # 输出当前轮次的训练和验证损失与准确率
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")
#
# # 绘制损失函数和准确率图像
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
# plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
# plt.plot(range(num_epochs), val_accuracies, label='Val Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy')
# plt.legend()
#
# plt.show()
#
# # 模型测试（使用测试集）
# model.eval()
# test_correct = 0
# test_total = 0
# true_labels = []
# predicted_labels = []
#
# with torch.no_grad():
#     for batch_data, batch_labels in test_dataloader:
#         batch_data = batch_data.to(device)
#         batch_labels = batch_labels.to(device)
#
#         outputs = model(batch_data)
#         _, predicted = torch.max(outputs.data, 1)
#         test_total += batch_labels.size(0)
#         test_correct += (predicted == batch_labels).sum().item()
#
#         true_labels.extend(batch_labels.cpu().numpy())
#         predicted_labels.extend(predicted.cpu().numpy())
#
# # 计算准确率
# test_accuracy = 100 * test_correct / test_total
# print(f"Test Accuracy: {test_accuracy}%")
#
# # 计算混淆矩阵
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# print("Confusion Matrix:")
# print(conf_matrix)
#
# # 可视化混淆矩阵
# plt.figure(figsize=(12, 7))
# ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.0f')
# ax.set_xlabel('Predicted Labels')
# ax.set_ylabel('True Labels')
# ax.set_title('Confusion Matrix')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.show()
#

