import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import scipy.stats
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 在一个一维函数上训练一个生成对抗网络

class Tabular:
    def __init__(self, data, categorical_cols=None):
        self.categorical_cols = categorical_cols or []

        self.data = self.pre_process(np.array(data))
    

        self.columns = data.columns
        # 一维数据的个数
        self.n_inputs = self.data.shape[1]
        # 隐空间的维度
        self.latent_dim = 10
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator()
        self.gan_model = self.define_gan(self.discriminator, self.generator)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    # 定义独立的判别器模型
    def define_discriminator(self):
        model = Sequential()
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.n_inputs))
        model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # 编译模型
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    # 定义独立的生成器模型
    def define_generator(self):
        model = Sequential()
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
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
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    # 用生成器生成 n 个假样本和类标签
    def generate_fake_samples(self, generator, latent_dim, n):
        # 在隐空间中生成点
        x_input = self.generate_latent_points(latent_dim, n)
        # 预测输出值
        X = generator.predict(x_input)
        # 创建类标签
        y = np.zeros((n, 1))
        return X.astype(np.float32), y.astype(np.float32)

    # 生成隐空间中的点作为生成器的输入
    def generate_latent_points(self, latent_dim, n):
        # 在隐空间中生成点
        x_input = np.random.randn(latent_dim * n)
        # 为网络调整一个 batch 输入的维度大小
        x_input = x_input.reshape(n, latent_dim)
        return x_input

    # 生成 n 个真实样本和类标签
    def generate_real_samples(self, n):
        assert len(self.data) >= n
        # 从样本集中随机选取n个样本
        idx = np.random.choice(range(len(self.data)), n, replace=False)
        X = self.data[idx, :]
        # 生成类标签
        y = np.ones((n, 1))
        return X.astype(np.float32), y.astype(np.float32)

    # 评估判别器并且绘制真假点
    def summarize_performance(self, epoch, n=50):
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
        # # 绘制真假数据的散点图
        # pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
        # pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
        # pyplot.show()

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
        df = pd.DataFrame(x_fake)
        df.columns = self.columns

        df = self.post_process(df)

        return df

    def pre_process(self, data):
        return data

    def post_process(self, data):
        return data



    # dataFrame, js散度评估，只可以处理数值型
    def js_evaluate(self, data1, data2):
        dataNp1 = np.array(data1)
        dataNp2 = np.array(data2)
        return js_div(dataNp1.flatten(), dataNp2.flatten(), num_bins=20)


def js_divergence(p, q):
    M = (p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

def js_div(arr1, arr2, num_bins):
    max0 = max(np.max(arr1), np.max(arr2))
    min0 = min(np.min(arr1), np.min(arr2))
    bins = np.linspace(min0-1e-4, max0-1e-4, num=num_bins)
    PDF1 = pd.cut(arr1, bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2, bins).value_counts() / len(arr2)
    return js_divergence(PDF1.values, PDF2.values)


if __name__ == '__main__':

    data_size = 200
    # 生成 [-0.5, 0.5] 范围内的输入值
    X1 = np.random.rand(data_size) - 0.5
    # 生成输出值 X^2
    X2 = np.square(X1)
    # 堆叠数组
    X1 = X1.reshape(data_size, 1)
    X2 = X2.reshape(data_size, 1)
    X = np.hstack((X1, X2))
    # print("X", X)
    js = js_div(X1.flatten(), X.flatten(), num_bins=20)
    print("js", js)
    # print("js", js_div(X, X_pre, num_bins=10))



