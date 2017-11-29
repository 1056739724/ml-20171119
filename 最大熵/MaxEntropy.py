#!/usr/bin/env python
#-*- coding:utf-8 -*-
import math
from collections import defaultdict

class MaximumEntropy:
    def __init__(self):
        self.iter = 40 #训练最多迭代次数
        self.threshold = 1e-2 #阈值,用来判断是否收敛
        self.labSet = set([])#存放label集合
        self.xyCount = defaultdict(int)#map对应(xi,y)和数量
        self.pxy = 0.0 #联合分布
        self.samples = [] #训练数据集
        self.M = 0 #每条数据最大特征数量,用于求参数时的迭代还有参数w初始化用到
        self.w = []#参数数组
        self.last_w = []#上一轮参数数组
        self.model_Epf = []  #对模型的期望
        self.sample_Epf = []  #对经验的期望
        self._xyID = {};  # 对(x,y)对做的顺序编号(ID), Key是(xi,yi)对,Value是ID

    def loadData(self, fileName):
        with open(fileName, "rb") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                sample = line.split("\t")
                if len(sample) < 2:
                    print "至少应该有一个特征和标签"
                else:
                    label = sample[0]
                    features = sample[1:]
                    self.labSet.add(label)
                    self.samples.append(sample)
                    if len(features) > self.M:
                        self.M = len(features)
                    for feature in features:
                        self.xyCount[(feature, label)] += 1

    def calcuZw(self, sample):
        # 规范化因子,计算Zw(x)
        Zw = 0.0
        for label in self.labSet:
            sum = 0.0
            for feature in sample:
                if (feature, label) in self.xyCount:
                    sum += self.w[self._xyID[(feature, label)]]
            Zw += math.exp(sum)
        return Zw

    def calcuPyx(self, sample):
        # 计算条件概率p(y/x)
        results = []
        Zw = self.calcuZw(sample)
        for label in self.labSet:
            sum = 0.0
            for feature in sample:
                if (feature, label) in self.xyCount:
                    sum += self.w[self._xyID[(feature, label)]]
            pyx = 1.0 / Zw * math.exp(sum)
            results.append((label, pyx))
        return results

    def initParameter(self):
        # 初始化参数
        self.w = [0.0] * len(self.xyCount)
        self.last_w = [0.0] * len(self.xyCount)
        self.model_Epf = [0.0] * len(self.xyCount) #对模型的期望
        self.sample_Epf = [1.0/len(self.samples)] * len(self.xyCount)#对经验的期望

        for i, xy in enumerate(self.xyCount):
            self._xyID[xy] = i;
        for sample in self.samples:
            label = sample[0]
            features = sample[1:]
            self.pxy = 1.0 / len(self.samples)#默认样本不重复,故经验分布一样
            for feature in features:
                self.sample_Epf[self._xyID[(feature, label)]] += self.pxy

    def convergence(self):
         for last_w, w in zip(self.last_w,self.w):
            if math.fabs(w - last_w) > self.threshold:
                return False

    def model_ep(self):
        for sample in self.samples:
            features = sample[1:]
            results = self.calcuPyx(features)
            for label, pyx in results:
                for xi in features:
                    if (xi, label) in self.xyCount:
                        self.model_Epf[self._xyID[(xi, label)]] += 1.0 * pyx/ len(self.samples)

    def trainModel(self):
        self.initParameter()
        for j in range(self.iter):
            self.last_w = self.w #保存上一次的权重
            self.model_ep()
            for i,w in enumerate(self.last_w):
                self.w[i] += 1.0 / self.M * math.log(self.sample_Epf[i]/self.model_Epf[i])
            #判断是否收敛
            if self.convergence():
                break

    def predictResult(self,input):
        #预测样本的类别,input是样本输入,预测新样本属于每个类别的概率
        X = input.strip().split("\t")
        results = self.calcuPyx(X)
        print results

if __name__ == '__main__':
    maximumEntropy = MaximumEntropy ()
    #读取数据集
    maximumEntropy.loadData('data.txt')
    #训练模型
    maximumEntropy.trainModel()
    maximumEntropy.predictResult("sunny\thot\thigh\tFALSE")


