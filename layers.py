import numpy as np


class Data():#处理各种数据
    def __init__(self, name, batch_size):#文件名name,图片数量batch_size,一次批处理的量
        with open(name, 'rb')as f:
            data = np.load(f)
            self.x = data[0]#加载图片的数据(所有）
            self.y = data[1]#加载所有的图片对应的标签，对应的字母为数字
            self.l = len(self.x)#计算图片的个数
            self.batch_size = batch_size#一次要进行的的数据
            self.pos = 0#标记读取的位置,这位置是总体的位置，不是一次批处理的位置

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        l = self.l#记录总长度
        if pos + bat >= l:#当这次批处理的长度数量不足时，重新开始
            ret = (self.x[pos:l], self.y[pos:l])#
            self.pos = 0#重设图片位置
            index = list(range(l))#图片序列
            np.random.shuffle(index)#重新排序
            self.x = self.x[index]#
            self.y = self.y[index]#
        else:#继续进行
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])#提取数据元组
            self.pos += self.batch_size#
        return ret, self.pos#返回数据，与标签

    def backward(self, d):
        pass


class FullyConnect():#这是个输入17X17的，输出26的网络层
    def __init__(self, l_x, l_y):
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)#初始化网络26*289
        self.bias = np.random.randn(l_y, 1)#初始化偏置
        self.lr = 0#初始化学习率

    def forward(self, x):
        self.x = x#这里x为输入的训练数据，不带标签的，1024*289（289=17*17）
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])#一次批量处理的结果，一个数代表结果，weights为26*289，xx为289*1的向量
        return self.y#返回结果向量1024*26

    def backward(self, d):#反向传播网络d为26*1
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]#
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]#axis＝0表示按列相加，axis＝1表示按照行的方向相加
        self.db = np.sum(d, axis=0) / self.x.shape[0]#axis＝0表示按列相加，axis＝1表示按照行的方向相加
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        self.weights -= self.lr * self.dw#更新权重
        self.bias -= self.lr * self.db#
        return self.dx#反向传播


class Sigmoid():#构造函数，这里默认为一层，实际和隐藏层结合的
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx


class QuadraticLoss():#损失函数
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x#这里的X为结果每次批预测的结果向量
        self.label = np.zeros_like(x)#产生一个零向量，维度和x一致，为1024*26
        for a, b in zip(self.label, label):
            a[b] = 1.0#a为一个含有26个元素的向量
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2#计算损失函数，是一个平均值

        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]#反向函数
        return self.dx


class Accuracy():#说白了，就是
    def __init__(self):
        pass

    def forward(self, x, label):#这里的x为，label为
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])#argmax为取xx的向量里最大数的位置，若位置和ll的值一致，则为1，否则为0
        self.accuracy = 1.0 * self.accuracy / x.shape[0]#取平均值
        return self.accuracy

