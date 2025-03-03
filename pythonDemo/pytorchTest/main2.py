# 启用Python 3的print函数语法，兼容Python 2
from __future__ import print_function

# 导入所需的库
import argparse  # 解析命令行参数

import torch
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 包含常用激活函数和损失函数
import torch.optim as optim  # 优化算法模块
from torch.optim.lr_scheduler import StepLR  # 学习率调整策略
from torchvision import datasets, transforms  # 数据集和预处理方法


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # LSTM层：输入特征28维，隐藏层64维，batch_first=True表示输入形状为(batch, seq, feature)
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)  # 一维批量归一化，处理64维特征
        self.dropout1 = nn.Dropout2d(0.25)  # 2D Dropout，丢弃概率25%
        self.dropout2 = nn.Dropout2d(0.5)  # 2D Dropout，丢弃概率50%
        self.fc1 = nn.Linear(64, 32)  # 全连接层，64输入 -> 32输出
        self.fc2 = nn.Linear(32, 10)  # 全连接层，32输入 -> 10输出（对应10个数字类别）

    def forward(self, input):
        # 输入形状转换：(batch, 1, 28, 28) -> (batch, 28, 28)
        # 将每张图像的28x28像素视为28个时间步，每个时间步输入28维特征
        input = input.reshape(-1, 28, 28)
        # LSTM处理序列数据，output包含每个时间步的输出
        output, hidden = self.rnn(input)

        # 只取最后一个时间步的输出作为特征，形状(batch, 64)
        output = output[:, -1, :]
        output = self.batchnorm(output)  # 批量归一化
        output = self.dropout1(output)  # 随机丢弃部分神经元防止过拟合
        output = self.fc1(output)  # 全连接层
        output = F.relu(output)  # ReLU激活函数
        output = self.dropout2(output)  # 再次随机丢弃
        output = self.fc2(output)  # 最终全连接层
        # 计算log_softmax（对数概率），dim=1表示在类别维度计算
        output = F.log_softmax(output, dim=1)
        return output


# 训练函数
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # 设置为训练模式（启用Dropout和BatchNorm）
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据移至GPU/CPU
        optimizer.zero_grad()  # 清空梯度缓存，防止梯度累积
        output = model(data)  # 前向传播计算输出
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 每隔指定间隔打印训练信息
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:  # 快速验证模式，只跑一个batch
                break


# 测试函数
def test(args, model, device, test_loader):
    model.eval()  # 设置为评估模式（关闭Dropout和BatchNorm）
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测类别（最大概率索引）
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测数
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)  # 计算平均损失
    # 打印测试结果
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='训练批次大小 (默认: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='测试批次大小 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='训练轮数 (默认: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='学习率 (默认: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='学习率衰减系数 (默认: 0.7)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='启用CUDA加速')
    parser.add_argument('--mps', action="store_true", default=False,
                        help="启用MPS加速（Mac专用）")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='快速验证模式（仅跑一个批次）')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子 (默认: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='日志打印间隔（单位：批次）')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='保存当前模型')
    args = parser.parse_args()

    # 设备选择（优先CUDA，其次MPS，最后CPU）
    if args.cuda and not args.mps:
        device = "cuda"
    elif args.mps and not args.cuda:
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)

    # 设置随机种子（保证可重复性）
    torch.manual_seed(args.seed)

    # 数据加载配置
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # 加载MNIST训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
                           transforms.Normalize((0.1307,), (0.3081,))  # 标准化（均值，标准差）
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # 加载MNIST测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # 初始化模型和优化器
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 使用Adadelta优化器

    # 学习率调度器（每epoch衰减一次）
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()  # 更新学习率

    # 保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_rnn.pt")


# 脚本入口
if __name__ == '__main__':
    main()