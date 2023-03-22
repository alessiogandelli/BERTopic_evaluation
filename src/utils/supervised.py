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

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")    

#%%
# given the path of a directory containing jsonl files, (tweets extracted with twarc2)
# return a dataframe with the tweets
# every jsonl file is a topic

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

def preprocess_text(text):

    text = re.sub(r'http\S+', '', text) # remove urls
    text = text.lower()                 # lowercase
    text = re.sub(r'@\S+', '', text)        # remove mentions
    text = re.sub(r'#', '', text)       # remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = re.sub("(\\d|\\W)+"," ",text)        # remove numbers
    text = text.strip()                 # remove whitespaces
    text = contractions.fix(text)           # expand contractions
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))]) # remove stopwords
    text = text.replace('amp', '')   # remove amp
    text = ' '.join([word for word in text.split() if len(word) > 2])#remove 2 letter words

    return text



class Supervised:

    def __init__(self, dataset, name, nr_topics = 'auto', min_topic_size = 50):
        self.df = dataset.copy()
        self.docs = self.df['text'].tolist()
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size

        self.embeddings = None
        self.embedder = None
        self.model = None
        self.accuracy = None                 # accracy result
        self.accuracy_no_outliers = None     # accuracy without outliers 
        self.name = name
        self.topic_share = None

        self.evaluate()             # create embeddings 
        self.get_accuracy()                 # get accuracy

    # given name of the model compute the embeddings, if name is openai use openai api 
    # else only sentence tranformer are allowed
    def evaluate(self):
        print('evaluate', self.name)

        if(self.name == 'openai'):
            embs = openai.Embedding.create(input = self.docs, model="text-embedding-ada-002")['data']
            self.embeddings = np.array([np.array(emb['embedding']) for emb in embs])
            self.get_topic_model()            # create model
            self.get_accuracy()
        
        elif(self.name == 'NMF'):
            self.get_NMF()

        else:
            self.embedder = SentenceTransformer(self.name)
            self.embeddings = self.embedder.encode(self.docs)
            self.get_topic_model()            # create model
            self.get_accuracy()

    # create topic model with bertopic and update the dataframe with the inferred topics 
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

    def get_NMF(self):
        self.df['preprocessed'] = df['text'].apply(preprocess_text)
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=3, ngram_range=(1,2))
        dtm = tfidf.fit_transform(self.df['preprocessed'])


        nmf_model = NMF(n_components= len(self.df['topic'].unique()), random_state=42)

        topics = nmf_model.fit_transform(dtm)



        self.model = nmf_model
        self.df['my_topics'] = topics.argmax(axis=1)


    def get_accuracy(self):
        df = self.df
        topics = df['topic'].unique()
        my_topics = df['my_topics'].unique()
        results = {}
        results_no_outliers = {}
        topic_share = {}

        # every my_topic should have >90 % pf the documents of topic
        for topic in my_topics:
            res = df[df['my_topics'] == topic].value_counts('topic') 
            topic_share[topic] = round(res[0]/ sum(res) ,2)

        # compute accuracy for each topic
        if (min(topic_share.values()) < 0.85):
            print('topic share is too low, proprablt accuracy is not meaningful for ', min(topic_share))
            print('check the heatmap ')
        results['min_topic_share'] = min(topic_share.values())

        for topic in topics:
            res = df[df['topic'] == topic].value_counts('my_topics') 
            first = res.iloc[0] if res.index[0] != -1 else res.iloc[1]                       # i'm assuming that out of the possible label the right one is the biggest 
            missed = sum(res.iloc[1:]) if res.index[0] != -1 else sum(res) - res.iloc[1]    # sum of the other labels
            try :
                outliers = res.loc[-1]
            except:
                outliers = 0

            
            
            results[topic] = first / (first + missed)
            results_no_outliers[topic] = first / (first + missed - outliers)  # bertopic mark the outliers with -1, i do not consider them while computing accuracy
        
        
        self.accuracy = results
        self.accuracy_no_outliers = results_no_outliers
        self.topic_share = topic_share

        return results
    
    def visualize_documents(self):
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(self.embeddings)
        return self.model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings)

    def visualize_heatmap(self):
        return sns.heatmap(pd.crosstab(self.df['topic'], self.df['my_topics']), annot=True, cmap="YlGnBu", fmt='g')

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
