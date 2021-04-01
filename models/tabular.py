import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 在一个一维函数上训练一个生成对抗网络

class Tabular:
    def __init__(self, data):
        self.data = data
        # 一维数据的个数
        self.n_inputs = self.data.shape[1]
        # 隐空间的维度
        self.latent_dim = 6
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator()
        self.gan_model = self.define_gan(self.discriminator, self.generator)

    # 定义独立的判别器模型
    def define_discriminator(self):
        model = Sequential()
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.n_inputs))
        model.add(Dense(1, activation='sigmoid'))
        # 编译模型
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # 定义独立的生成器模型
    def define_generator(self):
        model = Sequential()
        model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(Dense(self.n_inputs, activation='linear'))
        return model

    # 定义合并的生成器和判别器模型，来更新生成器
    def define_gan(self, discriminator, generator):
        # 将判别器的权重设为不可训练
        discriminator.trainable = False
        # 连接它们
        model = Sequential()
        # 加入生成器
        model.add(generator)
        # 加入判别器
        model.add(discriminator)
        # 编译模型
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    # 用生成器生成 n 个假样本和类标签
    def generate_fake_samples(self, generator, latent_dim, n):
        # 在隐空间中生成点
        x_input = self.generate_latent_points(latent_dim, n)
        # 预测输出值
        X = generator.predict(x_input)
        # 创建类标签
        y = np.zeros((n, 1))
        return X, y

    # 生成隐空间中的点作为生成器的输入
    def generate_latent_points(self, latent_dim, n):
        # 在隐空间中生成点
        x_input = np.random.randn(latent_dim * n)
        # 为网络调整一个 batch 输入的维度大小
        x_input = x_input.reshape(n, latent_dim)
        return x_input

    # 生成 n 个真实样本和类标签
    def generate_real_samples(self, n):
        assert len(self.data) > n
        # 从样本集中随机选取n个样本
        idx = np.random.choice(range(len(self.data)), n, replace=False)
        X = self.data[idx, :]
        # 生成类标签
        y = np.ones((n, 1))
        return X, y

    # 评估判别器并且绘制真假点
    def summarize_performance(self, epoch, n=100):
        # 准备真实样本
        x_real, y_real = self.generate_real_samples(n)
        # 在真实样本上评估判别器
        _, acc_real = self.discriminator.evaluate(x_real, y_real, verbose=0)
        # 准备假样本
        x_fake, y_fake = self.generate_fake_samples(self.generator, self.latent_dim, n)
        # 在假样本上评估判别器
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # 总结判别器性能
        print(epoch, acc_real, acc_fake)
        # 绘制真假数据的散点图
        pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
        pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
        pyplot.show()

    def train(self, n_epochs=10000, n_batch=128, n_eval=2000):
        # 用一半的 batch 数量来训练判别器
        half_batch = int(n_batch / 2)

        # 手动遍历 epoch
        for i in range(n_epochs):
            # 准备真实样本
            x_real, y_real = self.generate_real_samples(half_batch)
            # 准备假样本
            x_fake, y_fake = self.generate_fake_samples(self.generator, self.latent_dim, half_batch)
            # 更新判别器
            self.discriminator.train_on_batch(x_real, y_real)
            self.discriminator.train_on_batch(x_fake, y_fake)
            # 在隐空间中准备点作为生成器的输入
            x_gan = self.generate_latent_points(self.latent_dim, n_batch)
            # 为假样本创建反标签
            y_gan = np.ones((n_batch, 1))
            # 通过判别器的误差更新生成器
            self.gan_model.train_on_batch(x_gan, y_gan)
            # 为每 n_eval epoch 模型做评估
            if (i + 1) % n_eval == 0:
                self.summarize_performance(i)

    def generate(self, size=100):
        '''size can be list'''
        x_fake, y_fake = self.generate_fake_samples(self.generator, self.latent_dim, size)
        pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
        pyplot.show()
        return x_fake, y_fake


if __name__ == '__main__':

    data_size = 1000
    # 生成 [-0.5, 0.5] 范围内的输入值
    X1 = np.random.rand(data_size) - 0.5
    # 生成输出值 X^2
    X2 = np.square(X1)
    # 堆叠数组
    X1 = X1.reshape(data_size, 1)
    X2 = X2.reshape(data_size, 1)
    X = np.hstack((X1, X2))
    # print("X", X)

    tabular = Tabular(X)
    tabular.train()
    tabular.generate()
    tabular.generate()
    tabular.generate()



