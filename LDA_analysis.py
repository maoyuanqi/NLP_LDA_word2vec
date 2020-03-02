import jieba
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim import corpora



#read in excel dataset
text_data  = pd.read_excel("/Users/maoyuanqi/Downloads/datasets/data3.xlsx",index_col=None)
#data cleaning
text_data['label'] = text_data['label'].replace(['正面'],1)
text_data['label'] = text_data['label'].replace(['负面'],0)
text_data = text_data.sort_values('label')
#append each text from label into lines in txt


#train-test data split
#positivereview_list = []

pos_review_list = []
for review in text_data[text_data['label'] == 1]['evaluation']:
    pos_review_list.append(review)

neg_review_list = []
for review in text_data[text_data['label'] == 0]['evaluation']:
    neg_review_list.append(review)


#use total data to construct model


#cut words
#suggest some frequencies

jieba.suggest_freq(['差评','性价比高','lj','jia','性价比极高','性价比超高','满意','很好'],True)
pos_segments = []
for each_review in pos_review_list:
    pos_segments.append(' '.join(jieba.cut(each_review)))
#print(pos_segments)
#delete stop words

with open("/Users/maoyuanqi/Desktop/stoplists.txt",'r') as f:
    stop_words = f.readlines()

stop_words = [x.strip() for x in stop_words]

#print(stop_words)
for i in range(0,len(pos_segments)):
    for should_delete_word in stop_words:
        if(should_delete_word in pos_segments[i]):
            pos_segments[i] = pos_segments[i].replace(should_delete_word,"")
        else:
            pass
#fed data into work2vec
with open("pos_reviews_segmented.txt",'w') as file:
    for seg_review in pos_segments:
        file.write(seg_review)

neg_segments = []
for each_review in neg_review_list:
    neg_segments.append(' '.join(jieba.cut(each_review)))
#delete stop words
for i in range(0,len(neg_segments)):
    for should_delete_word in stop_words:
        if(should_delete_word in neg_segments[i]):
            neg_segments[i] = neg_segments[i].replace(should_delete_word,"")
        else:
            pass
# fed data into work2vec
with open("neg_reviews_segmented.txt", 'w') as file:
    for seg_review in neg_segments:
        file.write(seg_review)
#fed into model

pos_dict = corpora.Dictionary([word.split() for word in pos_segments])
pos_corpus = [pos_dict.doc2bow(i) for i in [word.split() for word in pos_segments]]
pos_lda_model = LdaModel(pos_corpus,num_topics=10,id2word=pos_dict)
print("positive topics:")
for i in range(10):
    print(pos_lda_model.print_topic(i))
print("__________________________")
##negative
print("negative topics:")
neg_dict = corpora.Dictionary([word.split() for word in neg_segments])
neg_corpus = [neg_dict.doc2bow(i) for i in [word.split() for word in neg_segments]]
neg_lda_model = LdaModel(neg_corpus,num_topics=10,id2word=neg_dict)
for i in range(10):
    print(neg_lda_model.print_topic(i))
print("__________________________")