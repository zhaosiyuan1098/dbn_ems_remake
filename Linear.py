import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loader import Loader
from option import Option

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def linear():
    # 定义神经网络模型
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(24, 64)  # 输入层到第一个隐藏层
            self.fc2 = nn.Linear(64, 32)  # 第一个隐藏层到第二个隐藏层
            self.fc3 = nn.Linear(32, 12)  # 第二个隐藏层到输出层

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 加载数据
    option = Option()
    loader = Loader(option)
    x_train, x_valid, y_train, y_valid = loader.load_for_dbn()

    # 将数据转换为TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = NeuralNet().cuda()  # 如果有GPU，使用.cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # 验证模型
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.5f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.5f}')

    # # 保存模型
    # torch.save(model.state_dict(), 'model.pth')

    # 生成混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(conf_matrix).plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    linear()
