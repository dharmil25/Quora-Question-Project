import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
eng_stopwords = set(stopwords.words('english'))


class Quora_class():
    def __init__(self):
        self.chill = None

    def load_data(self,path):
        return pd.read_csv(path,delimiter='\t',encoding='utf-8')[:50000].dropna(inplace=True)
    
    def common_words(self,x):
        q1, q2 = x
        return len(set(str(q1).lower().split()) & set(str(q2).lower().split()))

    def words_count(self,question):
        return len(str(question).split())

    def length(self,question):
        return len(str(question))

    def vect(self,train_df):
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, max_df=0.5)
        all_questions = pd.concat([train_df["question1"], train_df["question2"]], ignore_index=True)
        all_Q = vectorizer.fit_transform(all_questions.values)
        return all_Q
         

    def feature_engineering(self,train_df):
        train_df['q1_words_num'] = train_df['question1'].map(self.words_count)
        train_df['q2_words_num'] = train_df['question2'].map(self.words_count)
        train_df['q1_length'] = train_df['question1'].map(self.length)
        train_df['q2_length'] = train_df['question2'].map(self.length)
        train_df['common_words'] = train_df[['question1', 'question2']].apply(self.common_words, axis=1)
        all_Q = self.vect(train_df)
        Q1 = all_Q[0:all_Q.shape[0]/2]
        Q2 = all_Q[all_Q.shape[0]/2:]
        train_df['tf_idf_dot_product'] = pd.Series(np.array([np.dot(Q1[i,:], Q2[i,:].T).A[0,0] for i in range(Q1.shape[0])])).values
        return train_df

        

    
