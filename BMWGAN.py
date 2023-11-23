import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import torch
torch.set_default_tensor_type(torch.DoubleTensor)



data = pd.read_csv(
    r'D:\PycharmProjects\不平衡数据 数据型分类\yeast\yeast-0-2-5-6_vs_3-7-8-9.dat', header=None)
# print(data)
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
labeldata = np.array(le.transform(y_label)).reshape(-1, 1)
columnstestdata = data.shape[1]-1
testdata = pd.concat([data.iloc[:, 0:columnstestdata], pd.DataFrame(labeldata)], axis=1)
testdata.columns = [i for i in range(0, columnstestdata + 1)]

'-----获取某一类数据-----'
# columnstestdata=2
mindata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==0,0:columnstestdata-1])
majdata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==1,0:columnstestdata-1])
real_data=torch.Tensor(np.array(majdata))

####
def bootstrap_variance(data, num_iterations):
    # 初始化一个空数组来存储每次重复采样得到的方差
    variances = []
    mean = []

    # 进行重复采样和方差计算
    for i in range(num_iterations):
        # 从原始样本中有放回地抽取随机样本
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # 计算重复样本的方差
        variance = np.cov(bootstrap_sample)
        mean1=np.mean(bootstrap_sample)
        # 将方差添加到数组中
        variances.append(variance)
        mean.append(mean1)
    # 计算方差的平均值
    bootstrap_variance = np.mean(variances)
    bootstrap_mean = np.mean(mean)
    return bootstrap_variance,bootstrap_mean


# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# data=majdata.values[:,0]
# num_iterations = 100
# estimated_variance,estimated_mean = bootstrap_variance(data, num_iterations)

Mean=[]
Cov=np.zeros((majdata.shape[1],majdata.shape[1]))
# print(Cov)
for i in range(majdata.shape[1]):
    data = majdata.values[:, i]
    num_iterations = 1000
    estimated_variance, estimated_mean = bootstrap_variance(data, num_iterations)
    Mean.append(estimated_mean)
    Cov[i,i]=estimated_variance

print('mean',Mean)
print('cov',Cov)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

'---创建生成器与判别器-----'
'判别器'
class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Discriminator, self).__init__()
        self.disc=nn.Sequential(nn.Linear(input_size,hidden_size),
                                nn.Linear(hidden_size, 512),
                                nn.LeakyReLU(0.1),
                                nn.Linear(512, hidden_size),
                                nn.LeakyReLU(0.1),
                                nn.Linear(hidden_size,output_size),
                                )

    def forward(self,disc_data):
        dic_output=self.disc(disc_data)
        return dic_output
'生成器'
class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        '''input_size 是指输入到生成器数据的维度，可以自定义，
        output_size是指输出到判别器的维度必须和源数据的维度相同，因为此时判别器需要判断是真数据还是假数据'''
        super(Generator, self).__init__()
        self.gen=nn.Sequential(nn.Linear(input_size,64),
                               nn.Linear(64, hidden_size),
                               nn.Linear(hidden_size,256),
                               # nn.Linear(128, 256),
                               nn.Linear(256, hidden_size),
                               # nn.Linear(256, hidden_size),
                               nn.LeakyReLU(0.1),
                               nn.Linear(hidden_size,output_size),
                               )

    def forward(self,gen_data):
        gen_data_output=self.gen(gen_data)
        return gen_data_output


'----概率密度图---'
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

#测试生成器与判别器
'---规定参数----'
'----'
# learning_rate=0.0009
# G_input_size=32
G_input_size=majdata.shape[1]
G_hidden_size=128
G_output_size=columnstestdata
gen=Generator(input_size=G_input_size,hidden_size=G_hidden_size,output_size=G_output_size).to(device)

'-------'
D_input_size=columnstestdata
D_hidden_size=128
D_output_size=1
disc=Discriminator(input_size=D_input_size,hidden_size=D_hidden_size,output_size=D_output_size).to(device)

'---定义优化算法---'
optim_gen=optim.RMSprop(gen.parameters(),lr=0.0001)
optim_disc=optim.RMSprop(disc.parameters(),lr=0.000001)

'---定义损失函数---'
# criterion=nn.BCELoss()   #采样上述损失函数

'''----参数迭代----'''
epochs=500
batch=majdata.shape[0]
# print(batch)
# print(batches)
lossG=[]
lossD=[]
loss_D1=[]
G_mean=[]
G_std=[]
loss_Real=[]
loss_Fake=[]
for epoch in range(epochs):
    '''对数据进行切分，每一次得到batch个数据'''
    '''训练分类器'''
    for i in range(20):
        #train on generator
        # stat=i*batch
        # end=stat+batch
        # '''判别器的损失'''
        # x_real_data=real_data[stat:end]
        optim_disc.zero_grad()
        disc_real_data=disc(real_data)

        #train on fake
        # noise=torch.randn((batch,G_input_size))
        noise = torch.Tensor(np.random.multivariate_normal(Mean, Cov, batch))
        gen_data1=gen(noise)
        gen_data2=disc(gen_data1.detach())
        loss_D=-torch.mean(disc_real_data)+torch.mean(gen_data2)
        # loss_D.append(error_sum.data.numpy())
        loss_D1.append(loss_D.detach().numpy())
        loss_D.backward()
        optim_disc.step()

        for p in disc.parameters():
            p.data.clamp_(-0.01, 0.01)

    lossD.append(loss_D1[-1])
    '''生成器的损失'''
    ##生成器的反向传播
    optim_gen.zero_grad()
    # noise = torch.randn((batch, G_input_size))
    noise = torch.Tensor(np.random.multivariate_normal(Mean, Cov, batch))
    gen_data4 = gen(noise)
    gen_data3=disc(gen_data4)
    loss_G=-torch.mean(gen_data3)
    lossG.append(loss_G.detach().numpy())
    loss_G.backward()
    optim_gen.step()
    with torch.no_grad():
        G_mean.append(np.mean(gen_data4.data.numpy(),axis=0))
        G_std.append(np.cov(gen_data4.data.numpy(),rowvar=False))
    if epoch % 10==0:
        print("mean:{},std:{},Epoch: {}, loss_D:{} ,loss_G:{}"
              .format(G_mean[-1],G_std[-1],epoch,lossD[-1],lossG[-1]))


print('lll',np.mean(lossD))
'''loss函数画图'''
plt.plot(lossG,c='green',label='loss G')
plt.title('Loss Function')
plt.plot(lossD,c='red',label='loss D')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(linestyle='-.')  #设置网格
plt.xticks(range(0, 210, 20))
plt.show()

# print(torch.randn((batch, G_input_size)).shape)
# print(torch.Tensor(np.random.multivariate_normal(Mean, Cov, batch)).shape)
# genedata=gen(torch.randn((batch, G_input_size))).detach().numpy()
result=[]
number1=mindata.shape[0]-majdata.shape[0]
print(number1)
for i in range(1000):
    genedata=gen(torch.Tensor(np.random.multivariate_normal(Mean, Cov, batch))).detach().numpy()
    # genedata = gen(torch.randn((batch, G_input_size))).detach().numpy()
    # print(genedata.shape)
    for j in genedata:
         # print(len(result))
         if len(result)>=number1:
             break
         result.append(j)
# print(len(result))


df1=pd.DataFrame(result)
df12=pd.DataFrame([1]*number1)
print(df12)
df3=pd.concat([df1,df12],axis=1)
df3.columns = [i for i in range(0, columnstestdata + 1)]
# print(df3)
# print(testdata)
df4=pd.concat([testdata,df3],axis=0)
print(df4)
# print(testdata.values.shape)
# print(df3.values.shape)
# df5=np.stack((testdata.values,df3.values))
# print(df5)
df1.to_csv(r'D:\PycharmProjects\BWGAN\数据\maj.csv',index=False)
df4.to_csv(r'D:\PycharmProjects\BWGAN\数据\last.csv',index=False)
# print('cov',cov1)
# pd.DataFrame(mean1).to_csv(r'D:\PycharmProjects\pythonProject2\sci3\数据\mean1.csv',index=False)
# pd.DataFrame(cov1).to_csv(r'D:\PycharmProjects\pythonProject2\sci3\数据\cov1.csv',index=False)
