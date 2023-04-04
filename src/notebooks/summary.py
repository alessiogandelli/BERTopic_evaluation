#%%
import sys
sys.path.append("/Users/alessiogandelli/dev/internship/BERTopic_evaluation/src/utils")
from supervised import Supervised
from supervised import get_test_dataset
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %%

df = get_test_dataset('./../../data/simple_supervised/')
df_no_hash = df.copy()
df_no_hash['text'] = df_no_hash['text'].str.replace(r'#\S+', '', case=False)

df_pol = get_test_dataset('./../../data/politics_supervised/')
df_pol_no_hash = df_pol.copy()
df_pol_no_hash['text'] = df_pol_no_hash['text'].str.replace(r'#\S+', '', case=False)

df_us = get_test_dataset('./../../data/US_supervised/')
df_us_no_hash = df_us.copy()
df_us_no_hash['text'] = df_us_no_hash['text'].str.replace(r'#\S+', '', case=False)
# %%
## SIMPLE DATASET
#gsdmm = Supervised(df, 'GSDMM')
#climatebert = Supervised(df, 'climatebert/distilroberta-base-climate-f')
#tweetclass = Supervised(df, 'louisbetsch/tweetclassification-bf-model')
nmf = Supervised(df, 'NMF')
bert = Supervised(df, 'all-MiniLM-L6-v2')
openai_eval = Supervised(df, 'openai')

#%%
bert_no_hash = Supervised(df_no_hash, 'all-MiniLM-L6-v2', n_iter= 10)
openai_no_hash = Supervised(df_no_hash, 'openai',n_iter= 10)
nmf_no_hash = Supervised(df_no_hash, 'NMF',n_iter= 1)

#%% 
## POLITICS DATASET

nmf_pol = Supervised(df_pol, 'NMF')
bert_pol = Supervised(df_pol, 'all-MiniLM-L6-v2', n_iter=10)
openai_pol = Supervised(df_pol, 'openai', n_iter=10)

#%%
nmf_pol_no_hash = Supervised(df_pol_no_hash, 'NMF')
bert_pol_no_hash = Supervised(df_pol_no_hash, 'all-MiniLM-L6-v2', n_iter=10)
openai_eval_pol_no_hash = Supervised(df_pol_no_hash, 'openai', n_iter=10)


#%%
## US POLITICS DATASET

nmf_us = Supervised(df_us, 'NMF')
bert_us = Supervised(df_us, 'all-MiniLM-L6-v2', n_iter=10)
openai_us = Supervised(df_us, 'openai', n_iter=10)

#%%
#remove nan in text 

nmf_us_no_hash = Supervised(df_us_no_hash, 'NMF')
bert_us_no_hash = Supervised(df_us_no_hash, 'all-MiniLM-L6-v2', n_iter=10)
openai_us_no_hash = Supervised(df_us_no_hash, 'openai', n_iter=10)


# %%

def get_avg_accuracy(model):
    acc = []
    ts = []
    n_my_topics = []
    n_topics = []
    
    for test in model.accuracy:
        #remove min topic size from accuracy['accuracy']
        if('min_topic_share' in test['accuracy']):
            del test['accuracy']['min_topic_share']

        acc.append(np.mean(list(test['accuracy'].values())))
        ts.append(np.mean(list(test['topic_share'].values())))
        n_my_topics.append(len(test['topic_share']))
        n_topics.append(len(test['accuracy']))

    
    return {'acc': np.mean(acc), 'ts': np.mean(ts), 'n_my_topics': np.mean(n_my_topics), 'n_topics': np.mean(n_topics)}


# %%

acc_bert_simple = get_avg_accuracy(bert)
acc_bert_no_hash_simple = get_avg_accuracy(bert_no_hash)
acc_bert_pol = get_avg_accuracy(bert_pol)
acc_bert_pol_no_hash = get_avg_accuracy(bert_pol_no_hash)
acc_bert_us = get_avg_accuracy(bert_us)
acc_bert_us_no_hash = get_avg_accuracy(bert_us_no_hash)
# %%
acc_openai_simple = get_avg_accuracy(openai_eval)
acc_openai_no_hash_simple = get_avg_accuracy(openai_no_hash)
acc_openai_pol = get_avg_accuracy(openai_pol)
acc_openai_pol_no_hash = get_avg_accuracy(openai_eval_pol_no_hash)
acc_openai_us = get_avg_accuracy(openai_us)
acc_openai_us_no_hash = get_avg_accuracy(openai_us_no_hash)
# %%
acc_nmf_simple = get_avg_accuracy(nmf)
acc_nmf_no_hash_simple = get_avg_accuracy(nmf_no_hash)
acc_nmf_pol = get_avg_accuracy(nmf_pol)
acc_nmf_pol_no_hash = get_avg_accuracy(nmf_pol_no_hash)
acc_nmf_us = get_avg_accuracy(nmf_us)
acc_nmf_us_no_hash = get_avg_accuracy(nmf_us_no_hash)






# %%
# one line for each model
df_acc = pd.DataFrame()
df_acc = df_acc.append({'model': 'bert', 'dataset': 'simple', 'acc': acc_bert_simple['acc'], 'ts': acc_bert_simple['ts'], 'n_topics': acc_bert_simple['n_topics'], 'n_my_topics': acc_bert_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'bert', 'dataset': 'simple_no_hash', 'acc': acc_bert_no_hash_simple['acc'], 'ts': acc_bert_no_hash_simple['ts'], 'n_topics': acc_bert_no_hash_simple['n_topics'], 'n_my_topics': acc_bert_no_hash_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'bert', 'dataset': 'politics', 'acc': acc_bert_pol['acc'], 'ts': acc_bert_pol['ts'], 'n_topics': acc_bert_pol['n_topics'], 'n_my_topics': acc_bert_pol['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'bert', 'dataset': 'politics_no_hash', 'acc': acc_bert_pol_no_hash['acc'], 'ts': acc_bert_pol_no_hash['ts'], 'n_topics': acc_bert_pol_no_hash['n_topics'], 'n_my_topics': acc_bert_pol_no_hash['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'bert', 'dataset': 'us_politics', 'acc': acc_bert_us['acc'], 'ts': acc_bert_us['ts'], 'n_topics': acc_bert_us['n_topics'], 'n_my_topics': acc_bert_us['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'bert', 'dataset': 'us_politics_no_hash', 'acc': acc_bert_us_no_hash['acc'], 'ts': acc_bert_us_no_hash['ts'], 'n_topics': acc_bert_us_no_hash['n_topics'], 'n_my_topics': acc_bert_us_no_hash['n_my_topics']}, ignore_index=True)

df_acc = df_acc.append({'model': 'openai', 'dataset': 'simple', 'acc': acc_openai_simple['acc'], 'ts': acc_openai_simple['ts'], 'n_topics': acc_openai_simple['n_topics'], 'n_my_topics': acc_openai_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'openai', 'dataset': 'simple_no_hash', 'acc': acc_openai_no_hash_simple['acc'], 'ts': acc_openai_no_hash_simple['ts'], 'n_topics': acc_openai_no_hash_simple['n_topics'], 'n_my_topics': acc_openai_no_hash_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'openai', 'dataset': 'politics', 'acc': acc_openai_pol['acc'], 'ts': acc_openai_pol['ts'], 'n_topics': acc_openai_pol['n_topics'], 'n_my_topics': acc_openai_pol['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'openai', 'dataset': 'politics_no_hash', 'acc': acc_openai_pol_no_hash['acc'], 'ts': acc_openai_pol_no_hash['ts'], 'n_topics': acc_openai_pol_no_hash['n_topics'], 'n_my_topics': acc_openai_pol_no_hash['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'openai', 'dataset': 'us_politics', 'acc': acc_openai_us['acc'], 'ts': acc_openai_us['ts'], 'n_topics': acc_openai_us['n_topics'], 'n_my_topics': acc_openai_us['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'openai', 'dataset': 'us_politics_no_hash', 'acc': acc_openai_us_no_hash['acc'], 'ts': acc_openai_us_no_hash['ts'], 'n_topics': acc_openai_us_no_hash['n_topics'], 'n_my_topics': acc_openai_us_no_hash['n_my_topics']}, ignore_index=True)

df_acc = df_acc.append({'model': 'nmf', 'dataset': 'simple', 'acc': acc_nmf_simple['acc'], 'ts': acc_nmf_simple['ts'], 'n_topics': acc_nmf_simple['n_topics'], 'n_my_topics': acc_nmf_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'nmf', 'dataset': 'simple_no_hash', 'acc': acc_nmf_no_hash_simple['acc'], 'ts': acc_nmf_no_hash_simple['ts'], 'n_topics': acc_nmf_no_hash_simple['n_topics'], 'n_my_topics': acc_nmf_no_hash_simple['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'nmf', 'dataset': 'politics', 'acc': acc_nmf_pol['acc'], 'ts': acc_nmf_pol['ts'], 'n_topics': acc_nmf_pol['n_topics'], 'n_my_topics': acc_nmf_pol['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'nmf', 'dataset': 'politics_no_hash', 'acc': acc_nmf_pol_no_hash['acc'], 'ts': acc_nmf_pol_no_hash['ts'], 'n_topics': acc_nmf_pol_no_hash['n_topics'], 'n_my_topics': acc_nmf_pol_no_hash['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'nmf', 'dataset': 'us_politics', 'acc': acc_nmf_us['acc'], 'ts': acc_nmf_us['ts'], 'n_topics': acc_nmf_us['n_topics'], 'n_my_topics': acc_nmf_us['n_my_topics']}, ignore_index=True)
df_acc = df_acc.append({'model': 'nmf', 'dataset': 'us_politics_no_hash', 'acc': acc_nmf_us_no_hash['acc'], 'ts': acc_nmf_us_no_hash['ts'], 'n_topics': acc_nmf_us_no_hash['n_topics'], 'n_my_topics': acc_nmf_us_no_hash['n_my_topics']}, ignore_index=True)



#round to 2 decimals
df_acc = df_acc.round(2)


df_acc.groupby(['model']).mean()

# %%

acc_gsdmm = get_avg_accuracy(gsdmm)
acc_climatebert = get_avg_accuracy(climatebert)
acc_tweetclass = get_avg_accuracy(tweetclass)


df_broken = pd.DataFrame()

#gsdmm climatebert tweetclass  these 3 models 

df_broken = df_broken.append({'model': 'gsdmm', 'dataset': 'simple', 'acc': acc_gsdmm['acc'], 'ts': acc_gsdmm['ts'], 'n_topics': acc_gsdmm['n_topics'], 'n_my_topics': acc_gsdmm['n_my_topics']}, ignore_index=True)
df_broken = df_broken.append({'model': 'climatebert', 'dataset': 'simple', 'acc': acc_climatebert['acc'], 'ts': acc_climatebert['ts'], 'n_topics': acc_climatebert['n_topics'], 'n_my_topics': acc_climatebert['n_my_topics']}, ignore_index=True)
df_broken = df_broken.append({'model': 'tweetclass', 'dataset': 'simple', 'acc': acc_tweetclass['acc'], 'ts': acc_tweetclass['ts'], 'n_topics': acc_tweetclass['n_topics'], 'n_my_topics': acc_tweetclass['n_my_topics']}, ignore_index=True)

df_broken = df_broken.round(2)

df_broken


# %%

import openai

def get_topics_label(model):

    topics = list(model.get_topic_info()['Topic'])
    topic_words = model.get_topics()
    labels = {}

    for topic in topics:
        tweets = model.get_representative_docs(topic)
        prompt = "you are a tweet labeler, you are given representative words from a topic and three representative tweets, given all these information give a short label for the topic, starts all with topic:."
        words = [word[0] for word in topic_words[0]]


        topic_label = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content":prompt},
                    {"role": "user", "content": 'words'+ str(words)},
                    {"role": "user", "content": 'tweet'+ tweets[0]},
                    {"role": "user", "content": 'tweet'+tweets[1]},
                    {"role": "user", "content": 'tweet'+tweets[2]},
                ]
        )
        labels[topic] = topic_label['choices'][0]['message']['content']
    
    return labels
# %%
# save df_Acc
df_acc.to_csv('supervised_Accuracy.csv')
# %%

models = [nmf, bert, openai_eval, 
          nmf_no_hash, bert_no_hash, openai_no_hash,
          nmf_pol, bert_pol, openai_pol,
            nmf_pol_no_hash, bert_pol_no_hash, openai_eval_pol_no_hash,
            nmf_us, bert_us, openai_us,
            nmf_us_no_hash, bert_us_no_hash, openai_us_no_hash]
         
# %%

# save to file pickle in only one file
import pickle
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)




# %%
# read models 
with open('models.pkl', 'rb') as f:
    modelli = pickle.load(f)
# %%

models = {'bert': bert_pol.accuracy[0],
          'openai': openai_pol.accuracy[0],
          'nmf': nmf_pol.accuracy[0]['accuracy']}

df = pd.DataFrame()
df['bert'] = bert_pol.accuracy[0]['accuracy']
df['openai'] = openai_pol.accuracy[0]['accuracy']
df['nmf'] = nmf_pol.accuracy[0]['accuracy']

df_share = pd.DataFrame()

df_share['bert_topic_share'] = bert_pol.accuracy[0]['topic_share']
df_share['openai_topic_share'] = openai_pol.accuracy[0]['topic_share']
df_share['nmf_topic_share'] = nmf_pol.accuracy[0]['topic_share']

fig = df_share.T.plot.bar()


fig.set_ylabel('Accuracy')
fig.set_title('Accuracy different models')

fig.set_ylim(0,1)

fig.set_xticklabels(['bert', 'openai', 'nmf'])

# yticks
fig.set_yticks(np.arange(0, 1.1, 0.1))

#grid evrey 0,1
fig.grid(axis='y', alpha=0.5)

#rotate xticks
fig.set_xticklabels(fig.get_xticklabels(), rotation=0)




#add yticks every 0.1
#plt.yticks(np.arange(0, 1.1, 0.1))
# legend outside 
fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
