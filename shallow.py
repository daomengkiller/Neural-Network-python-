from layers import *


def main():
    datalayer1 = Data('train.npy', 10000)  # 输入训练数据，一次批量为1024
    datalayer2 = Data('validate.npy', 10000)  # 输入验证数据，一次性验证10000，
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers.append(Sigmoid())  # 将全连接层和sigmoid层对接，其实是同一层
    losslayer = QuadraticLoss()  # 实例化损失函数
    accuray = Accuracy()  # 实例化准确函数
    for layer in inner_layers:  # 循环处理学习率为1000
        layer.lr = 1000.0
    epochs = 20  # 迭代次数
    for i in range(epochs):  # 循环输出
        print('epochs:', i)  #
        losssum = 0  #
        iters = 0
        while True:#每次迭代
            data, pos = datalayer1.forward()  # 从数据层，提取数据，返回一次批量的训练数据和对应的标签组成的data（即data包含两组）
            x, label = data  # 从数据分离训练和标签x为1024*289，label为1024*1，数字代表字母，例如25为z
            for layer in inner_layers:
                x = layer.forward(x)#注意这里是将x处理又返回x，连接层的处理，第一次为x为1024*26，第二次为1024*26
            loss = losslayer.forward(x, label)#计算损失值，为后面的处理
            losssum += loss#每次批处理的损失总和
            iters += 1#批处理个数
            d = losslayer.backward()#损失函数反向值，问题在这
            for layer in inner_layers[::-1]:
                d = layer.backward(d)
            if pos == 0:
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                accu = accuray.forward(x, label)#此为更新了x的值
                print('loss:', losssum / iters)
                print('accuracy:', accu)
                break


if __name__ == '__main__':
    main()
