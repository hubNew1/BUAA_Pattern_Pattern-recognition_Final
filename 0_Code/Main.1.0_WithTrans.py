"""
加入了transformer机制
将train文件夹整个作为训练集，使用RESULT_6文件夹输出

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
    "conv_size": 5,#卷积核尺寸
    "Hiden_size": 256, #隐藏层尺寸
    "output_size" : 18,
    "dropout": 0.1,
}


class ConvTransBlock(nn.Module):#在么一个卷积层后面接一个Transformer编码器
    def __init__(self, in_channels, out_channels, conv_size, seq_len, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_size, stride=2, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x):# x.shape = (batch, C, H, W)
        x = self.conv(x)  # (batch, C, H, W)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).reshape(b, c, h, w)
        return x

class Net(nn.Module):
    def __init__(self, 
            output_size = ModulFig["output_size"], 
            dropout = ModulFig["dropout"], 
            conv_size = ModulFig["conv_size"], 
            Hidden_size = ModulFig["Hiden_size"]
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=conv_size, stride=2, padding=0)  # 输入: (batch, 3, 188, 188)
        self.pool = nn.MaxPool2d(2, 2)
        self.block2 = ConvTransBlock(32, 64, conv_size, seq_len=47*47, embed_dim=64)
        self.block3 = ConvTransBlock(64, 128, conv_size, seq_len=3*3, embed_dim=128)
        self.seq_len = 4  # 假设序列长度为4
        self.embed_dim = 64  # 嵌入维度
        self.fc1 = nn.Linear(128*3*3,self.seq_len * self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc2 = nn.Linear(self.seq_len * self.embed_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.block2(x))
        x = self.pool(x)
        x = self.relu(self.block3(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))  # (batch, seq_len * embed_dim)
        # reshape为序列 (seq_len, batch, embed_dim)
        x = x.view(x.size(0), self.seq_len, self.embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        x = self.transformer(x)  # (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # (batch, seq_len * embed_dim)
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
        self.cached_images = []  # 新增：用于缓存图像张量

        # 预加载所有图像
        for label in range(18):  # 标签0~17
            sub_dir = os.path.join(root_dir, f"{label}")
            for img_name in os.listdir(sub_dir):
                img_path = os.path.join(sub_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, ModulFig["input_size"])  # 统一尺寸
                img = ToTensor()(img)  # 转换为Tensor并归一化到[0,1]
                if self.transform:
                    img = self.transform(img)
                
                self.img_paths.append(img_path)
                self.labels.append(label)
                self.cached_images.append(img)  # 缓存处理后的图像

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 直接返回缓存的图像和标签
        return self.cached_images[idx], self.labels[idx]


def train_loop(model, train_dataloader, test_dataloader, device, filename_process, filename_model, filename_result, filename_evaluate=None):
    #这里是训练用的循环
    # 训练模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = 2.0

    with open(filename_process, "w") as f:
        f.write("loss  lr  ac_rate_train  ac_rate_test  ac_rate_evaluate\n")
    
    epoch = 0
    while epoch < Idx["epoch"] and loss > 0.01:
        print(f"\n_______Epoch {epoch+1}/{Idx['epoch']}_______")
        epoch += 1
        
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
        train_result = evaluate2(model, train_dataloader, device)
        test_result = evaluate2(model, test_dataloader, device)
        ac_rate_train, trainresults_predict, trainresults_labels = train_result
        ac_rate_test, testresults_predict, testresults_labels = test_result

        print(f"Train accuracy: {ac_rate_train:.4f}%,   Test accuracy: {ac_rate_test:.4f}%")

        loss = running_loss / len(train_dataloader)
        lr = 1e-3 if (loss > 0.2 and epoch<20) else (1e-5 if (loss > 0.07 and epoch<35) else 1e-6 if (loss > 0.03 and epoch<50) else 1e-7)
        optimizer = optim.Adam(model.parameters(), lr= lr)  # 更新学习率
        lr = optimizer.param_groups[0]['lr']
        print(f"Training loss: {loss:.4f}, lr = {lr:.0e}, ")

        # 将结果写入文件
        with open(filename_process, "a") as f:
            f.write(f"{loss:.4f}  {lr:.0e}  {ac_rate_train:.4f}  {ac_rate_test:.4f}\n")

        if epoch%10==0 or epoch==5 or epoch==1:
            with open(filename_result.replace("RoundX.txt", f"Round{epoch}.txt"), "w") as f:
                f.write("Predicted_Labels  True_Labels\n")
                for i in range(len(testresults_predict)):
                    f.write(f"{testresults_predict[i]}  {testresults_labels[i]}\n")

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

def evaluate2(model, test_dataloader, device):#用于验证训练结果
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
        ac_rate = 100.0 * correct / total
        #print('Accuracy of the network on the test images: %.4f %%' % ac_rate)

    return ac_rate, testresults_predict, testresults_labels


def Save_precess(loss, ac_rate ,filename="loss.txt"):
    with open(filename, "a") as f:
        f.write(f"{loss}  {ac_rate}\n")    


Idx = {
    "train_size" : 0.8,
    "batch_size" : 128,
    "lr" : 1e-3,
    "epoch" : 60
}

if __name__ == '__main__':
    # 定义数据集
    print("Synthetic dataset loading...")
    fileroot_process="E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\classifier_2\\RESULT_6.1\\1_convX_hidX_process.txt"
    fileroot_model="E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\classifier_2\\RESULT_6.1\\Model\\convX_hidX_model_RoundX.pth"
    fileroot_result="E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\classifier_2\\RESULT_6.1\\2_convX_hidX_result_RoundX.txt"
    fileroot_evaluate="E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\classifier_2\\RESULT_6.1\\3_convX_hidX_evaluate_RoundX.txt"

    root_dir = "E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\18_flowers\\train"
    dataset = MyDataset(root_dir)


    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # 定义数据加载器
    train_dataloader = DataLoader(
        dataset, 
        batch_size=Idx["batch_size"], 
        shuffle=True, 
        num_workers=0,  # 可以增加worker数量加速数据加载
        pin_memory=True  # 加速数据传到GPU
    )
    print("Train dataset loaded successfully.")

    test_dataset = MyDataset("E:\\0_STUDY!\\7_Pytorch\\SomethingFun\\25-05\\18_flowers\\test")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size= Idx["batch_size"], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    print("test dataset loaded successfully.")
    
    convsize = ModulFig["conv_size"]
    hidensize = ModulFig["Hiden_size"]
    model = Net().to(device)
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
    # 训练模型
    train_loop(model, train_dataloader, test_dataloader, device, 
        filename_process, filename_model, filename_result, filename_evaluate
    )
    # 定义数据加载器
    #evaluate2(model, evaluate_dataloader, device, loss = 0.0 , filename_result=filename_evaluate)
            