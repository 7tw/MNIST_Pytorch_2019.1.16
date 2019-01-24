# coding=utf-8
# 配置库
import torch
from torch import nn #nn就是Neural NetWork，神经网络
from torch.autograd import Variable #从“自动求导”包中引入“变量”
from torchvision import datasets


# load model
# 保存模型
# torch.save(model.state_dict(), './cnn.pth')

# 定义卷积神经网络模型
class Cnn(nn.Module):   #表示Cnn继承了nn.Module
    def __init__(self, in_dim, n_class):  # 28x28x1
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(  #.conv，conv是卷积的意思
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),  # 输入数据大小是28x28xin_dim。经过这层后，输出大小是28x28x6。维度没变是因为padding=1，即采用了same填充。
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 经过这层后，缩小为14 x 14 x 6。
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 上层的14x14x6经过这层，变成10 x 10 x 16维度。
            nn.ReLU(True), nn.MaxPool2d(2, 2))  # 经过这层后缩小为5x5x16

        self.fc = nn.Sequential(    #.fc，fc是全连接
            nn.Linear(400, 120),  # 400 = 5 * 5 * 16，上一个conv最后的输出作为这里的输入
            nn.Linear(120, 84),     #120，84就是自定义的了。故此推断这应该是个LeNet-5.
            nn.Linear(84, n_class))     #最终输出一个n_class维的（即n类）

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)  # 400 = 5 * 5 * 16, 
        out = self.fc(out)
        return out


# 打印模型
print(Cnn)

model = Cnn(1, 10)  # 图片大小是28x28x1, 然后输出有10个class
# cnn = torch.load('./cnn.pth')['state_dict']
model.load_state_dict(torch.load('./cnn.pth'))

# 识别
print(model)
test_data = datasets.MNIST(root='./data', train=False, download=True)
with torch.no_grad():
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:20] / 255.0
    # 原注释：shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    # 说白了，就是[:20]取前20个值；然后/255的意思是把RGB压缩成灰度。0-255是RGB，而0-1是灰度
test_y = test_data.test_labels[:20]
print(test_x.size())
test_output = model(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'predict result')
print(test_y[:10].numpy(), 'real result')
