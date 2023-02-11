# necessary packages for this homework, you are free to import more.
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import functional as F
from torchvision import transforms

#train MLP in pytorch
#1、pass的作用：在定义空函数或者判断语句不需要执行任何内容的时候 需要用pass语句占位
'''
如以下示例：
def fun():
    pass
def func1(loss):
    if(loss==1):
        pass
'''

'''
训练过程：
1、前向传播
2、反向传播
3、计算损失
4、优化器梯度清零
5、反向传播
优化器的作用：
'''
#一般此步是进行model或者其他变量.train完成
#为什么此步可以封装？ 统一的步骤进行更新
#存在的问题 优化函数不知道哪里调 torch
#传入模型的变量不能使用np
def one_optimize_step(x,gt,optimization,model,cartesian):
    y_pred=model(x)
    loss=cartesian(y_pred,gt)
    optimization.zero_grad()
    loss.backward()
    optimizer.step()
    return loss




if __name__=="__main__":

    #initial parameter
    model=nn.Sequential(
    nn.Linear(1,256),
    nn.ReLU(),
    nn.Linear(256,1)
    )#还需要再初始化一次吗
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adagrad(params= model.parameters(),lr=1e-3)
    x=np.arange(0,6.28,0.01).reshape(-1,1) #reshape操作是为了得到多个数据 能够进行batch操作
    y=np.sin(x) #其shape和x保持一致
    NUM_DATA=x.shape[0]
    #shuffle the data
    shuffle_indx=np.arange(0,NUM_DATA)
    np.random.shuffle(shuffle_indx) #没有返回值
    x=torch.tensor(x[shuffle_indx],dtype=torch.float32)
    y=torch.tensor(y[shuffle_indx],dtype=torch.float32)

    Epoch=100
    Batch=4
    LOG_STEP=10
    epoch_list=[]
    loss_list=[]
    
    for epoch in range(Epoch):
        for start_indx in range(0,NUM_DATA,Batch):
            x_batch=x[start_indx : start_indx + Batch]
            y_batch=y[start_indx : start_indx + Batch]
            
            loss=one_optimize_step(x_batch,y_batch,optimizer,model,criterion)

            loss_val=loss.item()
            loss_list.append(loss_val)
            epoch_list.append(epoch)

        if (epoch+1)%LOG_STEP==0 or epoch==Epoch-1:
            print(
                f"Loss in epoch-{epoch+1:#0{len(str(NUM_DATA))}d}:{loss_val:.2e}"
            )


#定义一个简单的模型训练一步的过程
#输入变量 要使用torch 
#优化器是指定如何更新梯度的  pytorch的封装性 
# 如一步训练的过程 完全可以封装成一个函数 
# 所以 所谓的封装就是指 固有的 非常类似的逻辑 步骤 提取出来
