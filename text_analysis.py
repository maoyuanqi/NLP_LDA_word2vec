import jieba
from gensim.models import word2vec
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report


#read in excel dataset
text_data  = pd.read_excel("/Users/maoyuanqi/Downloads/datasets/data3.xlsx",index_col=None)
#data cleaning
text_data['label'] = text_data['label'].replace(['正面'],1)
text_data['label'] = text_data['label'].replace(['负面'],0)
text_data = text_data.sort_values('label')
#append each text from label into lines in txt


#train-test data split

review_list = []
for review in text_data['evaluation']:
    review_list.append(review)

outcome_labels = []
for label in text_data['label']:
    outcome_labels.append(label)

x_train,x_test,y_train,y_test = train_test_split(review_list,outcome_labels,train_size=0.8,random_state=0)

#use total data to construct model
#cut words
#self define some should-be-recognized words

jieba.suggest_freq(['好评','差评','性价比高','lj','jia','性价比极高','性价比超高','满意','很好','不错'],True)
segments = []
for each_review in review_list:
    segments.append(' '.join(jieba.cut(each_review)))
print(segments)
#fed data into work2vec
with open("reviews_segmented.txt",'w') as file:
    for seg_review in segments:
        file.write(seg_review)


#fed into model
sentences = word2vec.LineSentence('reviews_segmented.txt')
model = Word2Vec(sentences , size=300,  min_count=10)
model.save('word2vec.model')

#total vacbs
vocabs = model.wv.vocab


req_count = 10
for key in model.wv.similar_by_word('不错', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break
req_count = 10
print("---------------")
for key in model.wv.similar_by_word('好', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

req_count = 10
print("---------------")

for key in model.wv.similar_by_word('垃圾', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break
req_count = 10
print("---------------")
for key in model.wv.similar_by_word('差评', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

#construct raw input to classifier
train_data_vectors = []

#construct words of x-train


for each_review in x_train:
    #对每一个review都有自己的word list
    words_list_x_train = []
    #对里面的词，都放到word list里
    for each_word in jieba.cut_for_search(each_review):
        words_list_x_train.append(each_word)
    #当前review的所有词都匹配到vector上，然后求平均作为input向量；
    temp_vector = np.array([model[word] for word in words_list_x_train if word in model])
    #添加刚刚那条review的向量
    train_data_vectors.append(temp_vector.mean(axis=0))

#construct RF for classifer
forest = RandomForestClassifier(n_estimators = 100, random_state=42)
forest = forest.fit(train_data_vectors, y_train)
svm = svm.LinearSVC()
svm.fit(train_data_vectors, y_train)


##test on test dataset
test_data_vectors = []

#construct words of x-train
for each_review in x_test:
    #对每一个review都有自己的word list
    words_list_x_test = []
    #对里面的词，都放到word list里
    for each_word in jieba.cut_for_search(each_review):
        words_list_x_test.append(each_word)
    #当前review的所有词都匹配到vector上，然后求平均作为input向量；
    temp_vector = np.array([model[word] for word in words_list_x_test if word in model])
    #添加刚刚那条review的向量
    test_data_vectors.append(temp_vector.mean(axis=0))

#查看分类结果 on test data using decision tree
result_decision_tree = forest.predict(test_data_vectors)
result_SVM = svm.predict(test_data_vectors)
print("Random Forest report ----------")
print(classification_report(y_test, result_decision_tree, target_names=['negative','positive']))
print("\n")
print("SVM report ----------")
print(classification_report(y_test, result_SVM, target_names=['negative','positive']))






