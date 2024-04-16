from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# 使用remove去除正文以外的信息
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes')) # 从20个新闻组数据集中获取数据  有403 Fofbidden 问题
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
max_features = 1000
# 将文本数据变换为向量
tf_vectorizer = CountVectorizer(max_features=max_features,stop_words='english')
tf = tf_vectorizer.fit_transform(data.data)
n_topics = 20
model = LatentDirichletAllocation(n_components=n_topics)
model.fit(tf)
print(model.components_) # 各主题包含的单词的分布
print(model.transform(tf)) # 使用主题描述的文本
