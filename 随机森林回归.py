import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from scipy import stats
import matplotlib.pyplot as plt

data_diabetes=load_diabetes()
data=data_diabetes['data']
target=data_diabetes['target']
feature_names=data_diabetes['feature_names']
X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=999)#划分训练集和测试集


class DecisionNode():
    def __init__(self, x_data, y_label, dimension, value=None,):
        self.x_data = x_data#数据
        self.y_label = y_label#标签
        self.dimension = dimension#属性下标
        self.value = value#阙值
        self.left = None#左子树
        self.right = None#右子树



class MetaLearner():
    def __init__(self,max_depth,min_samples):
        self.root = None#根节点
        self.max_depth = 5
        self.min_samples=20

    def fit(self, x_train, y_train):
        #计算信息熵
        a=self.max_depth
        b=self.min_samples
        def entropy(y_label):

            if len(y_label)==0:
                return 0.0
            else:
                mean=sum(y_label)/len(y_label)
                ent = 0.0
                for i in range(len(y_label)):
                    ent+=(y_label[i]-mean)**2
                ent=ent/len(y_label)
                return ent

        #划分数据集，根据属性和阙值将数据划分为左子树和右子树
        def split(x_data, y_label, dimension, value):
            index_left = (x_data[:, dimension] <= value)#小于阙值的为左子树
            #print(index_left)
            index_right = (x_data[:, dimension] > value)#大于阙值的为右子树
            return x_data[index_left], x_data[index_right], y_label[index_left], y_label[index_right]

        # 根据一个属性划分数据集 遍历所有维度的特征，不断寻找一个合适的划分数值，找到能把熵降到最低的那个特征和数值
        def one_split(x_data, y_label):

            best_gain = 0  # 最佳增益
            best_dimension = -1  # 最佳属性
            best_value = -1  # 最佳阙值
            # 计算最佳增益
            for d in range(x_data.shape[1]):
                sorted_index = list(set(x_data[:, d]))  # 每次循环取不同列，argsort的作用是从小到大排序
                for i in sorted_index:
                    x_left, x_right, y_left, y_right = split(x_data, y_label, d, i)
                    p_left = len(x_left) / len(x_data)
                    p_right = len(x_right) / len(x_data)

                    ent = entropy(y_label) -p_left * entropy(y_left) - p_right * entropy(y_right)
                    #更新增益
                    if ent > best_gain:
                        best_gain = ent
                        best_dimension = d
                        best_value = i
            return best_gain, best_dimension, best_value

        # 建树
        def create_tree(x_data, y_label,depth=0):
            if depth<a:
                ent, dim, value = one_split(x_data, y_label)
                x_left, x_right, y_left, y_right = split(x_data, y_label, dim, value)#划分
                node = DecisionNode(x_data, y_label, dim, value)#定义节点
                if ent < 0.000000001 or len(x_data)<=b: #直到信息熵接近0时，停止划分
                    return node
            #迭代继续划分
                node.left = create_tree(x_left, y_left,depth+1)
                node.right = create_tree(x_right, y_right,depth+1)
                return node

        self.root = create_tree(x_train, y_train)
        return self

    def predict(self, x_predict):
        def travel(x_data, node):
            p = node
            if x_data[p.dimension] <= p.value and p.left:
                pred = travel(x_data, p.left)
            elif x_data[p.dimension] > p.value and p.right:
                pred = travel(x_data, p.right)
            else:
                pred = np.mean(p.y_label)
                #pred = counter.most_common(1)[0][0]  # 数据对应的属性
            return pred

        ans = []
        #print(x_predict)
        for data in x_predict:
            #print(data)
            ans.append(travel(data, self.root))
        #print(ans)
        return np.array(ans)


class RandomForest():
    def __init__(self, n_estimators=10, min_samples=2,max_depth=5):
        self.n_estimators = n_estimators#决策树的棵树
        self.min_samples = min_samples#最少样本值
        self.max_depth=max_depth#最大深度
        self.estimators_ = []

    def Forestconstruct(self, data):
        for _ in range(self.n_estimators):
            tree = MetaLearner(min_samples=self.min_samples,max_depth=self.max_depth)#取我们写的决策树
            sub_data = self.sampling(data)#去样本值
            tree.fit(sub_data[:, :-1], sub_data[:, -1])#对样本值进行训练
            self.estimators_.append(tree)
    #有放回采样
    def sampling(self, data):
        n_samples, n_features = data.shape
        n_features -= 1
        sub_data = np.copy(data)
        random_f_idx = np.random.choice(
            n_features, size=int(np.sqrt(n_features)), replace=False
        )
        f_idx = [i for i in range(n_features) if i not in random_f_idx]
        random_data_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sub_data = sub_data[random_data_idx]
        sub_data[:, f_idx] = 0
        return sub_data

    def fit(self, X_train, y_train):
        data = np.c_[X_train, y_train]
        self.Forestconstruct(data)
        del data

    def predict(self, X_test):
        r_pred = np.array([tree.predict(X_test) for tree in self.estimators_]).T
        return np.array([stats.mode(y_pred)[0][0] for y_pred in r_pred])

    def loss(self,y_test,X_test):
        y=self.predict(X_test)
        #print(y)
        #print(y_test)
        MSE=sum(np.square(y-y_test))/len(y_test)#测试误差MSE
        y=np.sort(y)
        y_test=np.sort(y_test)
        plt.plot(y,y_test)#画出理论值与真实值的折线图
        #plt.show()
        return MSE

m=RandomForest()
m.fit(X_train,y_train)
result=m.predict(X_test)
print(y_test)#真实值
print(result)#预测值
m.loss(y_test,X_test)

#画出决策树棵树和误差的关系图
MSE=[]#取MSE为误差
for i in range(1,15):
    mt=RandomForest(n_estimators=i)
    mt.fit(X_train,y_train)
    resultt=mt.predict(X_test)
    MSE.append(np.sum((y_test-resultt)**2)/len(y_test))
n=[i for i in range(1,15)]
plt.plot(n,MSE,color="green")
plt.show()

#画出拟合曲线
n = np.arange(0, X_test.shape[0], 1)
plt.figure(figsize=(8, 5))#设置图片格式
plt.plot(n , result, c='r', label='prediction', lw=2)  # 画出拟合曲线
plt.plot(n , y_test, c='b', label='true', lw=2)  # 画出拟合曲线
plt.axis('tight') #使x轴与y轴限制在有数据的区域
plt.title("RandomForestRegressor" )
plt.show()