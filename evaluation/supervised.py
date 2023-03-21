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
import seaborn as sns

nltk.download('words')
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")    

#%%

def get_test_dataset(path: str):
    print('getting dataset')
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

    # def generate_tweet( n_words=10):
    #     words = set(nltk.corpus.words.words())
    #     words = [w for w in words if w.islower()]
    #     return ' '.join(np.random.choice(words, n_words))

    # for i in range(100):
    #     df.loc[i] = [generate_tweet(), 'en', 'random']

    return df


class Supervised:

    def __init__(self, dataset):
        self.df = dataset
        self.docs = self.df['text'].tolist()
        self.embeddings = None
        self.embedder = None
        self.nr_topics = 'auto'
        self.min_topic_size = 50
        self.model = None
        self.results = None

    # given name of the model start the evaluation
    def evaluate(self, name):
        print('evaluate', name)

        if(name == 'openai'):
            embs = openai.Embedding.create(input = self.docs, model="text-embedding-ada-002")['data']
            self.embeddings = np.array([np.array(emb['embedding']) for emb in embs])
            
        
        else:
            self.embedder = SentenceTransformer(name)
            self.embeddings = self.embedder.encode(self.docs)


        self.get_topic_model()
        self.accuracy()

            


    # 
    def get_topic_model(self):

        vectorizer_model = CountVectorizer(stop_words="english")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        model = BERTopic( 
                                vectorizer_model =   vectorizer_model,
                                ctfidf_model      =   ctfidf_model,
                                nr_topics        =   self.nr_topics,
                                min_topic_size   =   self.min_topic_size,
                                embedding_model  =   self.embedder
        )
        topics ,probs = model.fit_transform(self.docs, embeddings = self.embeddings)

        self.model = model
        self.df['my_topics'] = topics
        self.df['my_probs'] = probs

        return model

    def accuracy(self):
        df = self.df
        topics = df['topic'].unique()
        results = {}

        for topic in topics:
            res = df[df['topic'] == topic].value_counts('my_topics')
            first = res.iloc[0]
            second = sum(res.iloc[1:])
            results[topic] = first / (first + second)
        
        sns.heatmap(pd.crosstab(df['topic'], df['my_topics']), annot=True, cmap="YlGnBu", fmt='g')


        self.results = results

        return results
    
    def visualize_documents(self):
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(self.embeddings)
        return self.model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings)


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        documents = [text.replace("\n", " ") for text in documents]
        
        embs = self.embedding_model.Embedding.create(input = documents, model="text-embedding-ada-002")['data']
        return np.array([np.array(emb['embedding']) for emb in embs])
        #return embs



#%%
df = get_test_dataset('./../data')


#%%
bert = Supervised(df)
bert.evaluate('all-MiniLM-L6-v2')





#%%
openai_eval = Supervised(df)
openai_eval.evaluate('openai')
openai_eval.visualize_documents()




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




# %%
# heatmap between topic and my_topics 


# %%
