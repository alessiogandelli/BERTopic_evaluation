#%%
# read jsonl from file
import jsonlines
from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
import os 
from dotenv import load_dotenv
import openai
from bertopic.representation import OpenAI
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder
from hdbscan import HDBSCAN
import cohere
from bertopic.representation import Cohere
import numpy as np
import nltk
nltk.download('words')
load_dotenv()



#%%
# list filenames in directory
df[df['topic'] == 'trump']


def get_test_dataset(path: str):
    df = pd.DataFrame(columns=['text', 'lang', 'topic'])

    for file in os.listdir(path): # read files in directory
        if file.endswith(".jsonl"):
            with jsonlines.open(os.path.join(path,file)) as reader: # open file
                data = list(reader)
                for batch in data:  # every line contain 100 tweets
                    for tweet in batch['data']:
                        df.loc[tweet['id']] = [tweet['text'], tweet['lang'], file[:-6]]

    # remove hastags 
    #df['text'] = df['text'].str.replace(r'#\S+', '', case=False)

    df['text'] = df['text'].str.replace(r'RT', '', case=False)
    df['text'] = df['text'].str.replace(r'\n', '', case=False)
    df['text'] = df['text'].str.replace(r'http\S+', '', case=False) # remove urls

    df = df[df['lang'] == 'en'] # remove non english tweets



    def generate_tweet( n_words=10):
        words = set(nltk.corpus.words.words())
        words = [w for w in words if w.islower()]
        return ' '.join(np.random.choice(words, n_words))

    for i in range(100):
        df.loc[i] = [generate_tweet(), 'en', 'random']

    return df


def get_topic_model(tweets, embedding_model, nr_topics= 'auto'):

    
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)# avoid words that are frequent in all topics
    representation_model = KeyBERTInspired()
    #representation_model = Cohere(co, delay_in_seconds=10)
    #representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)


    topic_model = BERTopic( 
                            umap_model=umap_model, 
                            hdbscan_model=hdbscan_model,
                            vectorizer_model=vectorizer_model,
                            ctfidf_model=ctfidf_model,
                            #representation_model=representation_model,
                            embedding_model = embedding_model,
                            calculate_probabilities=False, 
                            nr_topics= nr_topics,
                            language='english',
                            top_n_words=15,
                            n_gram_range=(1, 2),
                            min_topic_size=15 # more documents ->  higher this number 
                            )
                        

    topic, probs = topic_model.fit_transform(tweets)

    return topic_model


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        documents = [text.replace("\n", " ") for text in documents]
        
        embs = self.embedding_model.Embedding.create(input = documents, model="text-embedding-ada-002")['data']
        return np.array([np.array(emb['embedding']) for emb in embs])
        #return embs


# %%
path = './../data'
df = get_test_dataset(path)
docs = df['text'].tolist()

# %%

bert_embedder =  SentenceTransformer('all-MiniLM-L6-v2')
bert_model = get_topic_model(docs, bert_embedder)

bert_model.get_topic_info()
# %%
openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = CustomEmbedder(openai).embed(documents=docs)

#%%
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
openai_model = BERTopic( 
                        vectorizer_model=vectorizer_model,
                        ctfidf_model=ctfidf_model,
                        nr_topics= 6,
                        min_topic_size=90
)
topic ,probs = openai_model.fit_transform(docs, embeddings=embeddings)

openai_model.get_topic_info()





# random tweet generator 

# %%
df['openai_topics'] = topic


# %%

df
# %%
df[df['topic'] == 'btc'].value_counts('openai_topics')
df[df['topic'] == 'formula1'].value_counts('openai_topics')
# %%



def evaluate(df):
    topics = df['topic'].unique()
    results = {}

    for topic in topics:
        res = df[df['topic'] == topic].value_counts('openai_topics')
        print(res)
        first = res.iloc[0]
        second = sum(res.iloc[1:])
        results[topic] = first / (first + second)

    return results
# %%
