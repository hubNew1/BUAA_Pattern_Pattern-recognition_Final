'''
这个文件是用于在不训练模型的情况下调用模型进行预测。
'''

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Main_1_0_WithTrans import Net, evaluate2, MyDataset

Idx = {
    "batch_size" : 20,
}

if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Net().to(device)
    
    ModelFile = "E:/0_STUDY!/7_Pytorch/SomethingFun/25-05/classifier_2/RESULT_6.3/Model/conv5_hid256_model_RoundX.pth"
    loaded_state_dict = torch.load(ModelFile, weights_only=True)
    model.load_state_dict(loaded_state_dict)
    print("...ModelLoaded...")

    
    train_file = "E:/0_STUDY!/7_Pytorch/SomethingFun/25-05/18_flowers/train"
    train_dataset = MyDataset(train_file)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=Idx["batch_size"], 
        shuffle=True, 
        num_workers=0,  # 可以增加worker数量加速数据加载
        pin_memory=True  # 加速数据传到GPU
    )
    print("Train dataset loaded successfully.")
    
    train_result_file = "E:/0_STUDY!/7_Pytorch/SomethingFun/25-05/classifier_2/RESULT_6.3/evaluate/train_result.txt" 
    train_ac, train_predict , train_labels = evaluate2(model, train_dataloader, device)
    print(f"ac on train imgs: {train_ac:.4f}")
    with open(train_result_file, "w") as f:
        f.write("Predicted_Labels  True_Labels\n")
        for i in range(len(train_predict)):
            f.write(f"{train_predict[i]}  {train_labels[i]}\n")
    '''

    test_file = "E:/0_STUDY!/7_Pytorch/SomethingFun/25-05/18_flowers/test"
    test_dataset = MyDataset(test_file)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size= Idx["batch_size"], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    print("test dataset loaded successfully.")

    test_result_file = "E:/0_STUDY!/7_Pytorch/SomethingFun/25-05/classifier_2/RESULT_6.3/evaluate/test_result.txt" 
    test_ac, test_predict, test_labels = evaluate2(model, test_dataloader, device)
    print(f"ac on test imgs: {test_ac}")
    with open(test_result_file, "w") as f:
        f.write("Predicted_Labels  True_Labels\n")
        for i in range(len(test_predict)):
            f.write(f"{test_predict[i]}  {test_labels[i]}\n")
    '''
