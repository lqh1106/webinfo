# Lab 1-2 豆瓣数据的个性化检索与推荐

## 实验任务

本实验中，我们利用现有的豆瓣 Movie&Book 的 tag 信息、豆瓣电影与书的评分记录以及用户间的社交关系，判断用户的偏好，基于 kNN 的评分预测方法和线性拟合方法对用户交互过的 item（电影、书籍）进行基于得分预测的排序，并实现了基于关系网的推荐算法。

## 实验内容

1. 数据处理：对豆瓣 Movie&Book 的 tag 信息、豆瓣电影与书的评分记录以及用户间的社交关系进行处理，得到用户的偏好信息。
2. 评分排序：基于 kNN 的评分预测方法和线性拟合方法对用户交互过的 item 进行基于得分预测的排序。
3. 推荐算法：基于关系网的推荐算法。

## 实验方法

### 数据处理

### 评分排序

假定用户偏好不与时间有关，即如果用户给某一类的书籍/电影打高分，我们就认为他会给此类相似的书籍/电影打高分。此处我们给出两种计算预测评分的方法：基于 kNN 的评分预测方法和线性拟合方法。

#### kNN

kNN (k-Nearest Neighbors) 方法是一种基于实例的有监督学习方法，它的基本思想是如果一个样本在特征空间中的 k 个最相似的样本中的大多数属于某一个类别，则该样本也属于这个类别。我们基于该算法，计算待预测书籍与用户评价过的书籍的相似度，然后根据相似度计算预测评分：

对于用户评价过的某一本书籍/电影，我们可以找到与其最为相似的 k 本书籍/电影，计算该用户对这 k 本书籍/电影的评分的加权平均值作为预测评分。这里我们使用余弦相似度作为相似度的度量，并作为电影评分的权重。

$$
\text{rating} = \frac{\sum_{i=1}^{k} \text{rating}_i \cdot \text{similarity}_i}{\sum_{i=1}^{k} \text{similarity}_i}
$$

其中 $\text{rating}_i$ 是用户对第 $i$ 本书籍/电影的评分，$\text{similarity}_i$ 是用户评价过的书籍/电影与第 $i$ 本书籍/电影的余弦相似度。

本实验中我们改变 k 的大小，根据该式计算预测评分，然后使用测试集计算 MSE 和 NDCG ，并探究与k 值大小的关系。

#### 线性拟合

我们将用户对书籍/电影的评分看作是书籍/电影的特征的线性组合，即

$$
\text{rating} = \sum_{i=1}^{n} w_i \cdot \text{feature}_i
$$

此时我们调用 sklearn 库划分训练集和测试集，划分比例为9:1，使用训练集训练线性回归模型，然后使用测试集计算 MSE 和 NDCG。

### 推荐算法

推荐算法的基本思想是，如果用户 A 与用户 B 有相似的兴趣，那么用户 A 喜欢的书籍/电影，用户 B 也可能喜欢。我们可以简单地将 A 的阅览记录作为 B 的推荐列表。这里我们使用余弦相似度作为相似度的度量，计算用户 A 阅览的某本书对用户 B 的推荐评分为：

$$
rating = \frac{\sum_{i=1}^{k} \text{rating}_i \cdot \text{similarity}_i}{\sum_{i=1}^{k} \text{similarity}_i} \cdot r_{A, i}
$$

即在用户 B 的阅览记录中，找到与用户 A 待推荐书籍最为相似的 k 个书籍/电影，计算用户 B 对这 k 个书籍/电影的评分的加权平均值，然后用用户 A 对这本书的评分加权，得到该书籍最终的推荐评分。最后依据推荐评分排序，得到推荐列表，返回给用户 B 即可。

## 实验结果

我们使用 MSE 和 NDCG 作为评价指标，计算两种方法的结果得到以下表格：

||MSE|NDCG|
|---|---|---|
| kNN | 0.6537304444906865 | 0.7991238148683567 |
| Linear Regression | 0.6429722291005368 | 0.8012059320420349 |

在这里我们可以看到，线性回归的效果略好于 kNN 方法。NDGC 的值在 0.8 左右，说明我们的预测效果还是不错的。

对 kNN 方法，我们改变 k 的大小，得到以下图像：

![kNN](./img/knn.png)

我们可以看到，随着 k 的增大，MSE 值逐渐减小，NDCG 值逐渐增大，说明 k 的增大对预测效果有一定的提升。（这块按你跑出来的实验结果改一下）

## 代码实现

### 数据处理

```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取 CSV 文件
file_path = 'lab1-2/data/book_tag.csv'
data = pd.read_csv(file_path)

# 假设 CSV 文件有两列：'Book' 和 'Tags'
# 将 'Tags' 列中的标签合并为一个字符串
data['Tags'] = data['Tags'].apply(lambda x: ' '.join(eval(x)))

# 初始化 TfidfVectorizer
vectorizer = TfidfVectorizer()

# 对 'Tags' 列进行 TF-IDF 分析
tfidf_matrix = vectorizer.fit_transform(data['Tags'])

# 获取词汇表
feature_names = vectorizer.get_feature_names_out()

# 将 TF-IDF 矩阵转换为 DataFrame，并添加书的 ID 列
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.insert(0, 'Book', data['Book'])

# 输出文档和词的向量化表达

# 保存结果到 CSV 文件
tfidf_df.to_csv('lab1-2/data/tfidf_result.csv', index=False)
```

### 评分排序

#### kNN 

代码见 `knn.py`

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import random
from tqdm import tqdm

def sort_by_rate(answer_ratings,target_ratings):


    sorted_indices = np.argsort(answer_ratings)[::-1]
    sorted_ratings = [target_ratings[i] for i in sorted_indices]
    
    return sorted_ratings

def dcg(scores):
    return np.sum([(2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores)])

def ndcg(target_sort, answer_sort):
    ndcg = []
    for i in range(len(target_sort)):
        dcg_val = dcg(answer_sort[:i+1])
        idcg_val = dcg(target_sort[:i+1])
        ndcg.append(dcg_val / idcg_val if idcg_val > 0 else 0)
    return ndcg

with open('./data/sim_score.json', 'r') as f:
    data = json.load(f)
    


total_ndcg = 0
total_len = len(data)
for key ,item in tqdm(data.items()):
    try:
        if len(item) == 1:
            continue
        target_ratings = []
        answer_ratings = []
        ratings = []
        similarities = []
        id = []
        for j in item:
            if len(j[2])==1:
                continue
            id.append(j[0])
            target_ratings.append(j[1])
            rating = []
            similaritie = []
            for i in j[2]:
                if i[1] > 0.99:
                    continue
                rating.append(i[0])
                similaritie.append(i[1])
            ratings.append(rating)
            similarities.append(similaritie)
        if target_ratings == []:
            continue    
        
        answer_ratings = [np.dot(r,s)/sum(s) for r,s in zip(ratings,similarities)]
        
        answer_sort = sort_by_rate(answer_ratings,target_ratings)
        target_sort = sorted(target_ratings,reverse=True)
        ndcg_score = ndcg(target_sort, answer_sort)
        
        total_ndcg += ndcg_score[-1]
        
    except:
        total_len -= 1
        continue
    
print(f'NDCG Score: {total_ndcg/total_len}')
```

#### 线性拟合

输入：数据集 data，相似度列表和评分列表，对相似度列表作归一化处理得到特征集 X，评分列表作为标签集 y。训练后计算 MSE 和 NDCG。

```python
target_ratings = []
ratings = []
similarities = []
id = []


for key ,item in data.items():
    for j in item:
        if len(j[2])==1:
            continue
        id.append(j[0])
        target_ratings.append(j[1])
        rating = []
        similaritie = []
        for i in j[2]:
            if i[0] == id[-1]:
                continue
            rating.append(i[2])
            similaritie.append(i[1])
        ratings.append(rating)
        similarities.append(similaritie)
        

    
max_len = max(len(r) for r in ratings)
ratings_padded = np.array([r + [int(0)] * (max_len - len(r)) for r in ratings])
similarities_padded = np.array([s + [int(0)] * (max_len - len(s)) for s in similarities])

features = np.hstack(similarities_padded)

features = [np.dot(r, s) / sum(s) for r, s in zip(ratings_padded, similarities_padded)]
features = np.array(features).reshape(-1, 1)
target_ratings = np.array(target_ratings)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target_ratings, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 计算 NDCG
total_ndcg = 0
for key ,item in tqdm(data.items()):
    if len(item) == 1:
        continue
    target_ratings = []
    answer_ratings = []
    ratings = []
    similarities = []
    id = []
    for j in item:
        if len(j[2])==1:
            continue
        id.append(j[0])
        target_ratings.append(j[1])
        rating = []
        similaritie = []
        for i in j[2]:
            if i[1] > 0.99:
                continue
            rating.append(i[0])
            similaritie.append(i[1])
        ratings.append(rating)
        similarities.append(similaritie)
    if target_ratings == []:
        continue    

    ratings_padded = np.array([r + [int(0)] * (max_len - len(r)) for r in ratings])
    similarities_padded = np.array([s + [int(0)] * (max_len - len(s)) for s in similarities])
    new_features = [np.dot(r, s) / sum(s) for r, s in zip(ratings_padded, similarities_padded)]
    new_features = np.array(new_features).reshape(-1, 1)
    answer_ratings = model.predict(np.array(new_features))
    
    answer_sort = sort_by_rate(answer_ratings,target_ratings)
    target_sort = sorted(target_ratings,reverse=True)
    ndcg_score = ndcg(target_sort, answer_sort)
    total_ndcg += ndcg_score[-1]
    

print(f'NDCG Score: {total_ndcg/len(data)}')
```

### 推荐算法

```python
import pandas as pd
import ast
import json
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 读取CSV文件
book_score = pd.read_csv('../lab1-1/dataset/book_score.csv')
movie_score = pd.read_csv('../lab1-1/dataset/movie_score.csv')

# 读取倒排索引字典
with open('./data/sim_score.json', 'r', encoding='utf-8') as f:
    inverted_index = json.load(f)

# 倒排索引字典记录 user_id -> { { book_id, cosine_similarity, rate}, ...}
# 根据 user_id，使用knn算法对书籍/电影进行推荐，输出结果为 {book_id, rate}的有序列表
def knn_recommend(user_id, k, is_book):
    # 读取数据
    if is_book:
        score = book_score
    else:
        score = movie_score

    # 读取用户评分信息
    user_score = score[score['User'] == user_id]

    # 读取用户的朋友信息
    with open('./data/Contacts.txt', 'r') as f:
        contacts = f.readlines()
    contacts = [contact.strip().split(':') for contact in contacts]
    contacts = {contact[0]: contact[1].split(',') for contact in contacts}

    # 计算用户的朋友的评分信息
    friends_score = []
    print(user_id)
    for friend in contacts[user_id]:
        friend_score = score[score['User'] == int(friend)]
        if friend_score.empty:
            continue
        friends_score.append(friend_score)

    # 计算用户的朋友的评分信息的倒排索引
    friends_inverted_index = {}
    for friend_score in friends_score:
        for index, row in friend_score.iterrows():
            book_id = row['Book'] if is_book else row['Movie']
            rate = row['Rate']
            if book_id in user_score['Book'].values: # 如果用户已经评分过该书籍，则不考虑
                continue
            if book_id in friends_inverted_index:
                friends_inverted_index[book_id].append(rate)
            else:
                friends_inverted_index[book_id] = [rate]

    # 计算朋友的评分书籍与用户的相似度
    for book_id in friends_inverted_index:
        # 计算 book_id 在user_id评价过的书籍中的相似度
        cosine_similarity = 0
        srate = 0
        for i in inverted_index[user_id][2]:
            if i[0] == book_id:
                cosine_similarity = i[1]
                srate = i[2]
                break
        friends_inverted_index[book_id] = cosine_similarity * srate * np.mean(friends_inverted_index[book_id])
    
    print (friends_inverted_index)
    sorted_friends_inverted_index = sorted(friends_inverted_index.items(), key=lambda x: x[1], reverse=True)
    return sorted_friends_inverted_index[:k]
    

print(knn_recommend('34894527', 5, True)) # 以用户34894527为例，推荐5本书籍
```

## 运行截图

### 评分排序

#### kNN

![kNN](./img/knn.png)  // 路径名改一下

#### 线性拟合

![Linear Regression](./img/linear_regression.png)


## 实验总结

在这个实验中，我们使用了两种方法对用户的评分进行预测，得到了不错的结果。但是这里我们只使用了用户的评分信息，实际上我们还可以使用用户的社交关系信息，以及时间戳等信息来提高预测的准确性。