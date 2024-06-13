#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import gc
from  lightgbm import LGBMClassifier,log_evaluation,early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# In[2]:


path='data/IND-WhoIsWho/'
path2 = 'data/IND-WhoIsWho/'

#sample: Iki037dt dict_keys(['name', 'normal_data', 'outliers'])
with open(path+"train_author.json") as f:
    train_author=json.load(f)
#sample : 6IsfnuWU dict_keys(['id', 'title', 'authors', 'abstract', 'keywords', 'venue', 'year'])   
with open(path+"pid_to_info_all.json") as f:
    pid_to_info=json.load(f)
#efQ8FQ1i dict_keys(['name', 'papers'])
with open(path2+"ind_test_author_filter_public.json") as f:
    valid_author=json.load(f)
with open(path2+"ind_test_author_submit.json") as f:
    submission=json.load(f)


# In[3]:


#转成dataframe
train_author_df = pd.DataFrame(train_author.values())

normal_data_df = train_author_df['normal_data'].apply(pd.Series).stack().reset_index(level=1, drop=True)
normal_data_df = normal_data_df.to_frame('papers')
normal_data_df['label'] = 1
normal_data_df = normal_data_df.join(train_author_df['name'])

outliers_df = train_author_df['outliers'].apply(pd.Series).stack().reset_index(level=1, drop=True)
outliers_df = outliers_df.to_frame('papers')
outliers_df['label'] = 0
outliers_df = outliers_df.join(train_author_df['name'])

train = pd.concat([normal_data_df, outliers_df], ignore_index=True)
print(train.shape)
print(train['label'].value_counts())
print(train.head())

test_author_df = pd.DataFrame(valid_author.values())

normal_data_df = test_author_df['papers'].apply(pd.Series).stack().reset_index(level=1, drop=True)
normal_data_df = normal_data_df.to_frame('papers')
test = normal_data_df.join(test_author_df['name'])
print(test.shape)
print(test.head())

paper_df = pd.DataFrame(pid_to_info.values())
paper_df.columns = ['papers', 'title', 'authors', 'abstract', 'keywords', 'venue', 'year']
print(paper_df.shape)
print(paper_df.head())


# In[4]:


# 提取每个作者的所有文章的关键词，计算TopK关键词重合度
def get_keywords_overlap(author_papers, paper_keywords, top_k=10):
    all_keywords = []
    for paper in author_papers:
        if paper in paper_keywords:
            all_keywords.extend(paper_keywords[paper])
    top_keywords = pd.Series(all_keywords).value_counts().head(top_k).index
    current_keywords = set(paper_keywords.get(author_papers[-1], []))
    return len(set(top_keywords) & current_keywords) / len(top_keywords) if len(top_keywords)>0 else 0

# 创建包含每篇论文关键词的字典
paper_keywords = paper_df.set_index('papers')['keywords'].to_dict()

# 获取每个作者的所有文章的关键词
author_papers_dict = train.groupby('name')['papers'].apply(list).to_dict()

# 计算训练集和测试集的关键词重合度
train['keywords_overlap'] = train.apply(
    lambda row: get_keywords_overlap(author_papers_dict.get(row['name'], []), paper_keywords), axis=1
)

author_papers_dict = test.groupby('name')['papers'].apply(list).to_dict()
test['keywords_overlap'] = test.apply(
    lambda row: get_keywords_overlap(author_papers_dict.get(row['name'], []), paper_keywords), axis=1
)


# In[5]:


#获取第一第二作者，并且给他映射
from sklearn.preprocessing import LabelEncoder
def extract_authors(authors_list):
    if not authors_list:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    first_author = authors_list[0]['name'] if authors_list else np.nan
    first_author_org = authors_list[0]['org'] if authors_list else np.nan
    if len(authors_list) > 2:
        third_author = authors_list[2]['name']
        third_author_org = authors_list[2]['org']
        second_author = authors_list[1]['name']
        second_author_org = authors_list[1]['org']
    elif len(authors_list) > 1:
        third_author = np.nan
        third_author_org = np.nan
        second_author = authors_list[1]['name']
        second_author_org = authors_list[1]['org']
    else:
        third_author = np.nan
        third_author_org = np.nan
        second_author = np.nan
        second_author_org = np.nan
    return pd.Series([first_author, first_author_org, second_author, second_author_org, third_author, third_author_org])

paper_df[['first_author', 'first_author_org', 'second_author', 'second_author_org', 'third_author', 'third_author_org']] = paper_df['authors'].apply(extract_authors)
columns_to_encode = ['first_author', 'first_author_org','second_author', 'second_author_org', 'third_author', 'third_author_org']
le = LabelEncoder()
for col in columns_to_encode:
    paper_df[col] = le.fit_transform(paper_df[col])


# In[6]:


data = pd.concat([train,test]).reset_index(drop=True)


# In[7]:


data = data.merge(paper_df, on='papers', how='left')
data.shape


# In[8]:


#！todo：做一些预处理，停用词的process等
data['venue'] = data['venue'].replace(np.nan, '')
data['full_text'] = data['title'] + data['abstract'] + data['keywords'].apply(lambda x: ', '.join(x)) + data['venue']


# In[9]:


#基础特征
data['title_length'] = data['title'].apply(lambda x: len(x))
data['abstract_length'] = data['abstract'].apply(lambda x: len(x))
data['keywords_length'] = data['keywords'].apply(lambda x: len(x))
data['authors_length'] = data['authors'].apply(lambda x: len(x))

data['year'] = data['year'].replace('', np.nan).astype(float)
data['year'] = data['year'].fillna(2000)
data['year'] = data['year'].astype(int)


# In[10]:


from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
def w2v_emb(df, f1, f2,emb):
    emb_size=emb
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    model = Word2Vec(sentences, vector_size=emb_size, window=5, min_count=1, sg=0, hs=0, epochs=32,workers=1,seed=2023)

    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv.key_to_index:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    return tmp


# In[11]:


w2v_name_title = w2v_emb(data,'name','title',32)
w2v_name_abstract = w2v_emb(data,'name','abstract',64)
w2v_name_venue = w2v_emb(data,'name','venue',32)
w2v_name_keywords = w2v_emb(data,'name','keywords',32)


# In[12]:


data = data.merge(w2v_name_title, on='name', how='left')
data = data.merge(w2v_name_abstract, on='name', how='left')
data = data.merge(w2v_name_venue, on='name', how='left')
data = data.merge(w2v_name_keywords, on='name', how='left')


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def tfidf_vec(df, f1, f2, emb_size=None):
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False).agg({f2: list})
    tmp['{}_{}_list'.format(f1, f2)] = tmp[f2].apply(lambda x: ' '.join([str(i) for i in x]))
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    vectorizer = TfidfVectorizer(max_features=emb_size)
    emb_matrix = vectorizer.fit_transform(sentences).toarray()
    
    for i in range(emb_matrix.shape[1]):
        tmp['{}_{}_tfidf_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del tmp[f2]
    del vectorizer, emb_matrix, sentences
    return tmp.drop(columns=['{}_{}_list'.format(f1, f2)])


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def count2vec(df, f1, f2, emb_size=None):
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False).agg({f2: list})
    tmp['{}_{}_list'.format(f1, f2)] = tmp[f2].apply(lambda x: ' '.join([str(i) for i in x]))
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    vectorizer = CountVectorizer(max_features=emb_size)
    emb_matrix = vectorizer.fit_transform(sentences).toarray()
    
    for i in range(emb_matrix.shape[1]):
        tmp['{}_{}_count_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    
    del vectorizer, emb_matrix, sentences
    del tmp[f2]
    return tmp.drop(columns=['{}_{}_list'.format(f1, f2)])


# In[15]:


stats = data.groupby('name')[['title_length', 'abstract_length', 'keywords_length', 'authors_length']].agg(['mean', 'std', 'max', 'min','sum','median'])
stats.columns = stats.columns.map('_'.join).str.strip('_')
stats = stats.reset_index()
for col in ['title_length', 'abstract_length', 'keywords_length', 'authors_length']:
    stats[col +'_ptp'] = stats[col +'_max'] - stats[col +'_min']
    stats[col +'_corr'] = stats[col +'_std'] / (stats[col +'_mean']+1e-5)
stats.head()


# In[16]:


data = pd.merge(data, stats, how='left', on='name')
data.shape


# In[24]:


import pandas as pd
# 定义处理函数
def clean_authors(authors):
    cleaned_authors = []
    for author in authors:
        cleaned_author = f"{author['name']} ({author['org']})"
        cleaned_authors.append(cleaned_author)
    return ', '.join(cleaned_authors)
# 应用处理函数到DataFrame的authors列
paper_df['authors'] = paper_df['authors'].apply(clean_authors)


# In[25]:


paper_df['venue'] = paper_df['venue'].replace(np.nan, '')
paper_df['keywords'] = paper_df['keywords'].apply(lambda x: ', '.join(x))
paper_df['full_text'] = paper_df['title'] + paper_df['abstract'] + paper_df['authors'] + paper_df['keywords']

le = LabelEncoder()
paper_df['authors'] = le.fit_transform(paper_df['authors'])


# In[26]:


from sklearn.decomposition import TruncatedSVD

def lsa_emb(df, f1, f2, emb_size=100):
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False).agg({f2: list})
    tmp['{}_{}_list'.format(f1, f2)] = tmp[f2].apply(lambda x: ' '.join([str(i) for i in x]))
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    
    vectorizer = CountVectorizer(max_features=emb_size)
    X = vectorizer.fit_transform(sentences)
    # 动态调整n_components以确保其小于特征数
    n_components = min(emb_size, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    emb_matrix = svd.fit_transform(X)
    for i in range(n_components):
        tmp['{}_{}_lsa_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del vectorizer, emb_matrix, sentences, X, svd
    del tmp[f2]
    return tmp.drop(columns=['{}_{}_list'.format(f1, f2)])

lsa_first_author = lsa_emb(paper_df,'first_author','full_text',32)
lsa_first_author_org = lsa_emb(paper_df,'first_author_org','full_text',32)
lsa_authors = lsa_emb(paper_df,'authors','full_text',32)
lsa_second_author = lsa_emb(paper_df,'second_author','full_text',16)
lsa_second_author_org = lsa_emb(paper_df,'second_author_org','full_text',16)

lsa_third_author = lsa_emb(paper_df,'third_author','full_text',16)
lsa_third_author_org = lsa_emb(paper_df,'third_author_org','full_text',16)

paper_df = paper_df.merge(lsa_first_author, on='first_author', how='left')
paper_df = paper_df.merge(lsa_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(lsa_authors, on='authors', how='left')
paper_df = paper_df.merge(lsa_second_author, on='second_author', how='left')
paper_df = paper_df.merge(lsa_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(lsa_third_author, on='third_author', how='left')
paper_df = paper_df.merge(lsa_third_author_org, on='third_author_org', how='left')
import gc
del lsa_first_author
del lsa_first_author_org
del lsa_second_author
del lsa_second_author_org
del lsa_authors
gc.collect()


# In[27]:


w2v_first_author = w2v_emb(paper_df,'first_author','full_text',32)
w2v_first_author_org = w2v_emb(paper_df,'first_author_org','full_text',32)
w2v_authors = w2v_emb(paper_df,'authors','full_text',32)
w2v_second_author = w2v_emb(paper_df,'second_author','full_text',16)
w2v_second_author_org = w2v_emb(paper_df,'second_author_org','full_text',16)

w2v_third_author = w2v_emb(paper_df,'third_author','full_text',16)
w2v_third_author_org = w2v_emb(paper_df,'third_author_org','full_text',16)


# In[28]:


paper_df = paper_df.merge(w2v_first_author, on='first_author', how='left')
paper_df = paper_df.merge(w2v_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(w2v_second_author, on='second_author', how='left')
paper_df = paper_df.merge(w2v_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(w2v_third_author, on='third_author', how='left')
paper_df = paper_df.merge(w2v_third_author_org, on='third_author_org', how='left')
paper_df = paper_df.merge(w2v_authors, on='authors', how='left')
import gc
del w2v_first_author
del w2v_first_author_org
del w2v_second_author
del w2v_second_author_org
del w2v_authors
gc.collect()


# In[29]:


tfidf_first_author = tfidf_vec(paper_df,'first_author','full_text',32)
tfidf_first_author_org = tfidf_vec(paper_df,'first_author_org','full_text',32)
tfidf_authors = tfidf_vec(paper_df,'authors','full_text',32)
tfidf_second_author = tfidf_vec(paper_df,'second_author','full_text',16)
tfidf_second_author_org = tfidf_vec(paper_df,'second_author_org','full_text',16)

tfidf_third_author = tfidf_vec(paper_df,'third_author','full_text',16)
tfidf_third_author_org = tfidf_vec(paper_df,'third_author_org','full_text',16)

paper_df = paper_df.merge(tfidf_first_author, on='first_author', how='left')
paper_df = paper_df.merge(tfidf_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(tfidf_authors, on='authors', how='left')
paper_df = paper_df.merge(tfidf_second_author, on='second_author', how='left')
paper_df = paper_df.merge(tfidf_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(tfidf_third_author, on='third_author', how='left')
paper_df = paper_df.merge(tfidf_third_author_org, on='third_author_org', how='left')
import gc
del tfidf_first_author
del tfidf_first_author_org
del tfidf_second_author
del tfidf_second_author_org
del tfidf_authors
gc.collect()


# In[30]:


count2vec_first_author = count2vec(paper_df,'first_author','full_text',32)
count2vec_first_author_org = count2vec(paper_df,'first_author_org','full_text',32)
count2vec_authors = count2vec(paper_df,'authors','full_text',32)
count2vec_second_author = count2vec(paper_df,'second_author','full_text',16)
count2vec_second_author_org = count2vec(paper_df,'second_author_org','full_text',16)
count2vec_third_author = count2vec(paper_df,'third_author','full_text',16)
count2vec_third_author_org = count2vec(paper_df,'third_author_org','full_text',16)

paper_df = paper_df.merge(count2vec_first_author, on='first_author', how='left')
paper_df = paper_df.merge(count2vec_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(count2vec_authors, on='authors', how='left')
paper_df = paper_df.merge(count2vec_second_author, on='second_author', how='left')
paper_df = paper_df.merge(count2vec_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(count2vec_third_author, on='third_author', how='left')
paper_df = paper_df.merge(count2vec_third_author_org, on='third_author_org', how='left')
import gc
del count2vec_first_author
del count2vec_first_author_org
del count2vec_second_author
del count2vec_second_author_org
del count2vec_authors
gc.collect()


# In[33]:


#venue
w2v_first_author = w2v_emb(paper_df,'first_author','venue',16)
w2v_first_author_org = w2v_emb(paper_df,'first_author_org','venue',16)
w2v_authors = w2v_emb(paper_df,'authors','venue',16)
w2v_second_author = w2v_emb(paper_df,'second_author','venue',8)
w2v_second_author_org = w2v_emb(paper_df,'second_author_org','venue',8)
w2v_third_author = w2v_emb(paper_df,'third_author','venue',8)
w2v_third_author_org = w2v_emb(paper_df,'third_author_org','venue',8)


paper_df = paper_df.merge(w2v_first_author, on='first_author', how='left')
paper_df = paper_df.merge(w2v_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(w2v_authors, on='authors', how='left')
paper_df = paper_df.merge(w2v_second_author, on='second_author', how='left')
paper_df = paper_df.merge(w2v_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(w2v_third_author, on='third_author', how='left')
paper_df = paper_df.merge(w2v_third_author_org, on='third_author_org', how='left')

del w2v_first_author
del w2v_first_author_org
del w2v_second_author
del w2v_second_author_org
del w2v_authors
gc.collect()

tfidf_first_author = tfidf_vec(paper_df,'first_author','venue',16)
tfidf_first_author_org = tfidf_vec(paper_df,'first_author_org','venue',16)
tfidf_authors = tfidf_vec(paper_df,'authors','venue',16)
tfidf_second_author = tfidf_vec(paper_df,'second_author','venue',8)
tfidf_second_author_org = tfidf_vec(paper_df,'second_author_org','venue',8)
tfidf_third_author = tfidf_vec(paper_df,'third_author','venue',8)
tfidf_third_author_org = tfidf_vec(paper_df,'third_author_org','venue',8)

paper_df = paper_df.merge(tfidf_first_author, on='first_author', how='left')
paper_df = paper_df.merge(tfidf_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(tfidf_authors, on='authors', how='left')
paper_df = paper_df.merge(tfidf_second_author, on='second_author', how='left')
paper_df = paper_df.merge(tfidf_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(tfidf_third_author, on='third_author', how='left')
paper_df = paper_df.merge(tfidf_third_author_org, on='third_author_org', how='left')
import gc
del tfidf_first_author
del tfidf_first_author_org
del tfidf_second_author
del tfidf_second_author_org
del tfidf_authors
gc.collect()

count2vec_first_author = count2vec(paper_df,'first_author','venue',16)
count2vec_first_author_org = count2vec(paper_df,'first_author_org','venue',16)
count2vec_authors = count2vec(paper_df,'authors','venue',16)
count2vec_second_author = count2vec(paper_df,'second_author','venue',8)
count2vec_second_author_org = count2vec(paper_df,'second_author_org','venue',8)
count2vec_third_author = count2vec(paper_df,'third_author','venue',8)
count2vec_third_author_org = count2vec(paper_df,'third_author_org','venue',8)

paper_df = paper_df.merge(count2vec_first_author, on='first_author', how='left')
paper_df = paper_df.merge(count2vec_first_author_org, on='first_author_org', how='left')
paper_df = paper_df.merge(count2vec_authors, on='authors', how='left')
paper_df = paper_df.merge(count2vec_second_author, on='second_author', how='left')
paper_df = paper_df.merge(count2vec_second_author_org, on='second_author_org', how='left')
paper_df = paper_df.merge(count2vec_third_author, on='third_author', how='left')
paper_df = paper_df.merge(count2vec_third_author_org, on='third_author_org', how='left')
import gc
del count2vec_first_author
del count2vec_first_author_org
del count2vec_second_author
del count2vec_second_author_org
del count2vec_authors
gc.collect()


# In[34]:


col = [i for i in paper_df.columns if i not in ['title', 'authors', 'abstract', 'keywords', 'venue', 'year','full_text',
       'first_author', 'first_author_org', 'second_author',
       'second_author_org', 'third_author', 'third_author_org']]


# In[36]:


data = data.merge(paper_df[col], on='papers', how='left')


# In[37]:


data['keywords'] = data['keywords'].apply(lambda x: ', '.join(x))


# In[38]:


#自身语义embeddding
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
def tfidf_fea(df,f, size):
    vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                strip_accents='unicode',
                analyzer = 'word',
                ngram_range=(1,3),
                min_df=0.05,
                max_df=0.95,
                sublinear_tf=True,
                max_features=32
    )
    paper_tfid = vectorizer.fit_transform([i for i in df[f]])
    dense_matrix = paper_tfid.toarray()
    df_tfidf = pd.DataFrame(dense_matrix)
    tfid_columns = [ f + f'_tfid_{i}' for i in range(len(df_tfidf.columns))]
    df_tfidf.columns = tfid_columns
    return df_tfidf
tfidf_name_title = tfidf_fea(data,'title',16)
tfidf_name_abstract = tfidf_fea(data,'abstract',32)
tfidf_name_venue = tfidf_fea(data,'venue',16)
tfidf_name_keywords = tfidf_fea(data,'keywords',16)

data = pd.concat([data,tfidf_name_title], axis=1)
data = pd.concat([data,tfidf_name_abstract], axis=1)
data = pd.concat([data,tfidf_name_venue], axis=1)
data = pd.concat([data,tfidf_name_keywords], axis=1)


# In[39]:


train = data[data['label'].notna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)
feat = [i for i in train.columns if i not in ['label', 'papers', 'title', 'name','authors','abstract','keywords','venue','full_text']]
print(train.shape)
print(test.shape)
print(len(feat))


# In[42]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
# 自定义函数进行模型训练和预测
def fit_and_predict(model, train_feats, test_feats, name=0):
    X = train_feats[feat].copy()
    y = train_feats['label'].copy()
    test_X = test_feats[feat].copy()
    groups = train_feats['name'].values  # 按name列进行分组
    oof_pred_pro = np.zeros((len(X), 2))
    test_pred_pro = np.zeros((5, len(test_X), 2))

    # 5折分组交叉验证
    gkf = GroupKFold(n_splits=5)

    for fold, (train_index, valid_index) in enumerate(gkf.split(X, y.astype(str), groups=groups)):
        print(f"name:{name}, fold:{fold}")

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  eval_metric="auc", early_stopping_rounds=100, verbose=100)

        oof_pred_pro[valid_index] = model.predict_proba(X_valid)
        # 将数据分批次进行预测
        test_pred_pro[fold] = model.predict_proba(test_X)

    print(f"roc_auc: {roc_auc_score(y.values, oof_pred_pro[:, 1])}")

    # 特征重要性
    importances = model.feature_importances_
    feature_names = feat

    # 按重要性降序排列
    sorted_idx = np.argsort(importances)[::-1]
    importances = importances[sorted_idx]
    feature_names = np.array(feature_names)[sorted_idx]

    # 返回预测结果和特征重要性
    return oof_pred_pro, test_pred_pro, importances, feature_names

# 定义LightGBM参数
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 12,
    "learning_rate": 0.05,
    "n_estimators": 3072,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "verbose": -1,
    "random_state": 1996,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees": True,
    'num_leaves': 64,
    "verbose": -1,
    "max_bin": 255,
}

# 训练LightGBM模型
lgb_oof_pred_pro, lgb_test_pred_pro, lgb_importances, lgb_feature_names = fit_and_predict(
    model=LGBMClassifier(**lgb_params), train_feats=train, test_feats=test, name='lgb'
)

# 定义XGBoost参数
xgb_params = {
    "objective": "binary:logistic",
    "metric": "auc",
    "max_depth": 12,
    "learning_rate": 0.05,
    "n_estimators": 3072,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "random_state": 1996,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "use_label_encoder": False,
}

# 训练XGBoost模型
xgb_oof_pred_pro, xgb_test_pred_pro, xgb_importances, xgb_feature_names = fit_and_predict(
    model=XGBClassifier(**xgb_params), train_feats=train, test_feats=test, name='xgb'
)


# In[43]:


train['pred'] = lgb_oof_pred_pro[:,1]*0.2 + xgb_oof_pred_pro[:,1]*0.8
result = train[['name','pred','label']]


# In[45]:


author_errors = result.groupby('name')['label'].sum()
total_errors = author_errors.sum()
weights = author_errors / total_errors
aucs = []
for name, group in result.groupby('name'):
    auc = roc_auc_score(group['label'], group['pred'])
    aucs.append((name, auc))
aucs = pd.Series(dict(aucs))
weighted_auc = (aucs * weights).sum()
weighted_auc


# In[66]:


path='data/IND-WhoIsWho/sub/'


# In[67]:


test_preds=lgb_test_pred_pro.mean(axis=0)[:,1]*0.2 + xgb_test_pred_pro.mean(axis=0)[:,1]*0.8


# In[68]:


cnt=0
for id,names in submission.items():
    for name in names:
        submission[id][name]=test_preds[cnt]
        cnt+=1
with open(path+'testb_baseline_795.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)


# In[ ]:




