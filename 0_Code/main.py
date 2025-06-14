"""
    前10 epoch使用的是1e-3的学习率，
    10之后使用的是1e-5的学习率精细优化 ，
    20之后使用的是1e-7的学习率精细优化，

    命名中conv为卷积层使用的卷积核尺寸，hid为隐藏层的神经元数量，Round为当前训练轮次，
    在训练过程中不断备记录训练结果，每两个epoch保存一次模型参数（用来在过拟合发生时进行回溯）
     
"""


import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

ModulFig = {
    "input_size": (188, 188),#输入尺寸(1*3*188*188)
    "conv_size": (3, 3),#卷积核尺寸
    "Hiden_size": 128, #隐藏层尺寸
    "output_size" : 18,
    "dropout": 0.2,
}

class Net(nn.Module):
    def __init__(self, 
        output_size=ModulFig["output_size"], 
        dropout=ModulFig["dropout"], 
        conv_size=ModulFig["conv_size"],
        Hidden_size=ModulFig["Hiden_size"]):
        
        super(Net, self).__init__()#这里定义了模型的结构，需要的话可以适量修改
        # 输入形状: (batch, 3, 188, 188)
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=conv_size, stride=1, padding=(conv_size-1)//2)  # 输出: (batch, 32, 188, 188)， 池化后：(batch, 32, 94, 94)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=conv_size, stride=1, padding=(conv_size-1)//2)  # 输出: (batch, 64, 94, 94)
        #self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))  # 输出: (batch, 128, 1, 94, 94)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 输出: (batch, 64, 47, 47)
        # 计算全连接层输入尺寸
        self.fc1_input_size = 64 * 47* 47  # 需要重新计算
        self.fc1 = nn.Linear(self.fc1_input_size, Hidden_size)  # 全连接层1
        self.fc2 = nn.Linear(Hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Relu(self.conv1(x))
        x = self.pool(x)  # 池化后: (batch, 32, 94, 94)
        x = self.Relu(self.conv2(x))
        #x = self.Relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.fc1_input_size)  # 展平
        x = self.Relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 对应20种花朵的文件夹
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for label in range(18):  # 标签0~17
            sub_dir = os.path.join(root_dir, f"{label}")
            for img_name in os.listdir(sub_dir):
                self.img_paths.append(os.path.join(sub_dir, img_name))
                self.labels.append(label)
                #print(f"Loading {self.img_paths[-1]} as label {label}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 动态加载图像
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, ModulFig["input_size"])  # 统一尺寸
        # 统一转换为Tensor
        img = ToTensor()(img)  # 自动归一化到[0,1]并转为CxHxW
        #img = img.unsqueeze(0)  # 增加batch维度
        #print(img.shape)
        # 应用额外变换
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]
        return img, label


def train_loop(model, dataset, evaluate_dataloader, device, filename_process, filename_model, filename_result, filename_evaluate=None):
    #这里是训练用的循环
    # 训练模型
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range( Idx["epoch"]):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 定义数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size= Idx["batch_size"], shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size= Idx["batch_size"], shuffle=False, num_workers=0)

        #动态调整学习率
        optimizer = optim.Adam(model.parameters(), lr=1e-3 if epoch < 10 else (1e-5 if epoch < 20 else 1e-7))
        
        print(f"\n______Epoch {epoch+1}/{Idx['epoch']}______")
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            print("*", end = "")
            #print(f"processing epoch: {i}/{len(train_dataloader)}")
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #模型测试与该步骤loss存储
        print("", end="\n")
        #在划分的测试集上检验正确率，并在额外的验证集上检验正确率，观察过拟合的发展情况
        reault = evaluate2(model, test_dataloader, evaluate_dataloader, device)
        ac_rate_test, ac_rate_evaluate, testresults_predict, testresults_labels, evaluateresults_predict, evaluateresults_labels = reault
        loss = running_loss / len(train_dataloader)
        print(f"Training loss: {loss:.4f}, ")

            # 将结果写入文件

        with open(filename_process, "a") as f:
            f.write(f"{loss:.4f}  {ac_rate_test:.4f}  {ac_rate_evaluate:.4f}\n")

        with open(filename_result.replace("RoundX.txt", f"Round{epoch}.txt"), "w") as f:
            f.write("Predicted_Labels  True_Labels\n")
            for i in range(len(testresults_predict)):
                f.write(f"{testresults_predict[i]}  {testresults_labels[i]}\n")

        with open(filename_evaluate.replace("RoundX.txt", f"Round{epoch}.txt"), "w") as f:
            f.write("Predicted_Labels  True_Labels\n")
            for i in range(len(evaluateresults_predict)):
                f.write(f"{evaluateresults_predict[i]}  {evaluateresults_labels[i]}\n")

        if epoch % 2 == 0:
            filename_model_withRound = filename_model.replace("RoundX.pth", f"Round{epoch}.pth")
            torch.save(model.state_dict(), filename_model_withRound)
    # 保存模型参数
    torch.save(model.state_dict(), filename_model)


def evaluate(model, test_dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            #print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %.4f %%' % (100.0 * correct / total))
    return 100.0 * correct / total      

def evaluate2(model, test_dataloader, evualuate_dataloader, device):#用于验证训练结果
    '''
    该函数用于验证模型在测试集上的准确率，并返回预测结果和标签。
    参数:
        model: 训练好的模型
        test_dataloader: 测试集数据加载器
        device: 设备（CPU或GPU）
        filename: 用于保存预测结果和真实标签的文件名
    返回:
        ac_rate: 模型在测试集上的准确率
        results: 包含预测结果和真实标签的列表，每个元素为一个元组(predicted, labels)
    '''
    testresults_predict = []
    testresults_labels = []
    evaluateresults_predict = []
    evaluateresults_labels = []

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data in test_dataloader:#在划分的测试集上检验正确率
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            testresults_predict.extend(predicted.cpu().tolist())
            testresults_labels.extend((labels.cpu().tolist()))#将结果输入在results中用于绘制混淆矩阵
        ac_rate_test = 100.0 * correct / total
        print('Accuracy of the network on the test images: %.4f %%' % ac_rate_test)

        correct = 0
        total = 0
        for data in evualuate_dataloader:#在划分的验证集上检验正确率
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            evaluateresults_predict.extend(predicted.cpu().tolist())
            evaluateresults_labels.extend((labels.cpu().tolist()))#将结果输入在results中用于绘制混淆矩阵
        ac_rate_evaluate = 100.0 * correct / total
        print('Accuracy of the network on the evaluate images: %.4f %%' % ac_rate_evaluate)

    return ac_rate_test, ac_rate_evaluate, testresults_predict, testresults_labels, evaluateresults_predict, evaluateresults_labels


def Save_precess(loss, ac_rate ,filename="loss.txt"):
    with open(filename, "a") as f:
        f.write(f"{loss}  {ac_rate}\n")    


Idx = {
    "train_size" : 0.8,
    "batch_size" : 64,
    "lr" : 1e-3,
    "epoch" : 30
}

if __name__ == '__main__':
    # 定义数据集
    print("Synthetic dataset loading...")
    fileroot_process="classifier\\2_RESULT\\convX_hidX_process.txt"#记录优化过程的文件地址
    fileroot_model="classifier\\2_RESULT\\convX_hidX_model_RoundX.pth"#备份训练过程中的模型参数
    fileroot_result="classifier\\2_RESULT\\convX_hidX_result_RoundX.txt.txt"#记录每一轮训练中测试集的预测标签（用于绘制混淆矩阵）
    fileroot_evaluate="classifier\\2_RESULT\\convX_hidX_evaluate_RoundX.txt"#记录每一轮训练中验证集的预测标签（用于观察过拟合情况

    root_dir = "classifier\\1_DATA\\train"
    dataset = MyDataset(root_dir)
    #划分训练集和测试集
    train_size = int(Idx['train_size'] * len(dataset))
    test_size = len(dataset) - train_size

    # 定义模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    for convsize in range(5, 10, 2):#遍历卷积核尺寸
        for hidensize in range(64, 257, 64):#遍历隐藏层节点数
            model = Net(conv_size= convsize, Hidden_size= hidensize).to(device)
            filename_process = fileroot_process.replace("convX_hidX", f"conv{convsize}_hid{hidensize}")
            filename_model = fileroot_model.replace("convX_hidX", f"conv{convsize}_hid{hidensize}")
            filename_result = fileroot_result.replace("convX_hidX", f"conv{convsize}_hid{hidensize}")
            filename_evaluate = fileroot_evaluate.replace("convX_hidX", f"conv{convsize}_hid{hidensize}")
            print(f"\nTraining model with conv_size={convsize}, Hidden_size={hidensize}...")
            '''
            # 加载已训练模型
            loaded_state_dict = torch.load(filename_model, weights_only=True)
            model.load_state_dict(loaded_state_dict)
            '''
            print("loaded successfully.")

            evaluate_dataset = MyDataset("classifier\\1_DATA\\test")
            evaluate_dataloader = DataLoader(evaluate_dataset, batch_size= Idx["batch_size"], shuffle=False, num_workers=0)

            # 训练模型
            train_loop(model, dataset, evaluate_dataloader, device, 
                    filename_process, filename_model, filename_result, filename_evaluate)
            # 定义数据加载器
            #evaluate2(model, evaluate_dataloader, device, loss = 0.0 , filename_result=filename_evaluate)
