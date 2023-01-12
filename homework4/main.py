import pandas as pd
import numpy as np
import torch
from torch import nn

FEATURE_NUMBER = 18
HOUR_PER_DAY = 24

def DataProcess(df):
    x_list, y_list = [], []
    array = np.array(df).astype(float)#设置数据类型

    for i in range(0, array.shape[0], FEATURE_NUMBER):
        for j in range(HOUR_PER_DAY - 9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9,j+9] # 用PM2.5作为标签
            x_list.append(mat)#作为自变量
            y_list.append(label)#作为因变量
    x = np.float32(np.array(x_list))#设置浮点数精度为32bits
    y = np.float32(np.array(y_list))
    return x, y

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()#允许维度变换
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),#激活函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):#forward就是专门用来计算给定输入，得到神经元网络输出的方法
        y_pred = self.linear_relu_stack(x)
        y_pred = y_pred.squeeze()#这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        #y_pred本事一个1行n列的数据，squeeze后就变成了n行
        return y_pred

if __name__ == '__main__':
    df = pd.read_csv('data.csv', usecols=range(2,26)) #去2~25列
    # 将RAINFALL的空数据用0进行填充
    df[df == 'NR'] = 0       
    x, y = DataProcess(df)#数据预处理
    # 输出（3，4）表示矩阵为3行4列
    # shape[0]输出3，为矩阵的行数
    # 同理shape[1]输出列数
    x = x.reshape(x.shape[0], -1)#矩阵转置
    #arr.reshape(m, -1)  # 改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m ）
    #np.arange(16).reshape(2, 8)  # 生成16个自然数，以2行8列的形式显示
    x = torch.from_numpy(x)#用来将数组array转换为张量Tensor（多维向量）
    y = torch.from_numpy(y)
    
    # 划分训练集和测试集
    x_train = x[:3000]
    y_train = y[:3000]
    x_test = x[3000:]
    y_test = y[3000:]
    
    model =  NeuralNetwork(x.shape[1])#shape[1]是获取矩阵的列数，由于是转置之后，原本是行数，样本数

    criterion = torch.nn.MSELoss(reduction='mean')#损失函数的计算方法
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)#定义SGD随机梯度下降法，学习率

    # train
    print('START TRAIN')
    for t in range(2000):
        
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)#获取偏差
        if (t+1) % 50 == 0:
            print(t+1, loss.item())

        optimizer.zero_grad()#在运行反向通道之前，将梯度归零。
        loss.backward()#反向传播计算梯度，否则梯度可能会叠加计算
        optimizer.step()#更新参数
    
    # test
    y_pred_test = model(x_test)
    loss_test = criterion(y_pred_test, y_test)#计算误差
    print('TEST LOSS:', loss_test.item())
    




