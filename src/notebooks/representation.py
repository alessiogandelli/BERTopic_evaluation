#%%
import openai

def get_topics_label(model):

    topics = list(model.get_topic_info()['Topic'])
    topic_words = model.get_topics()
    labels = {}

    for topic in topics:
        tweets = model.get_representative_docs(topic)
        prompt = "you are a tweet labeler, you are given representative words from a topic and three representative tweets, give more weight to the words, given all these information give a short label for the topic (max 10 words), starts all with topic:."
        words = [word[0] for word in topic_words[topic]]


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



#%%
#simple 
labels_bert_simple = get_topics_label(bert.model)
label_openai_simple = get_topics_label(openai_eval.model)
labels_bert_simple_nohash = get_topics_label(bert_no_hash.model)
label_openai_simple_nohash = get_topics_label(openai_no_hash.model)

# %%
df_label = pd.DataFrame.from_dict(labels_bert_simple, orient='index', columns=['bert_simple'])
df_label['openai_simple'] = pd.DataFrame.from_dict(label_openai_simple, orient='index', columns=['openai_simple'])
df_label['bert_simple_nohash'] = pd.DataFrame.from_dict(labels_bert_simple_nohash, orient='index', columns=['bert_simple_nohash'])
df_label['openai_simple_nohash'] = pd.DataFrame.from_dict(label_openai_simple_nohash, orient='index', columns=['openai_simple_nohash'])

# %%

# pol dataset 
labels_bert_pol = get_topics_label(bert_pol.model)
labels_openai_pol = get_topics_label(openai_pol.model)
labels_bert_pol_nohash = get_topics_label(bert_pol_no_hash.model)
labels_openai_pol_nohash = get_topics_label(openai_eval_pol_no_hash.model)

# %%
