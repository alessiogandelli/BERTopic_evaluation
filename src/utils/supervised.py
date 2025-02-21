#%%
# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import jsonlines
from dotenv import load_dotenv
import os 
# bertopic
from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder

from sentence_transformers import SentenceTransformer
import openai
from hdbscan import HDBSCAN
from umap import UMAP
import tensorflow_hub

from gsdmm import MovieGroupProcess

# preprocessing 
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text

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

# this is required for traditional methods 
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

    def __init__(self, dataset, name, nr_topics = 'auto', min_topic_size = 50, n_iter=1):
        self.df = dataset.copy()
        self.docs = self.df['text'].tolist()
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.n_iter = n_iter

        self.embeddings = None
        self.embedder = None
        self.model = None
        self.accuracy = []                 # accracy result
        self.accuracy_no_outliers = []     # accuracy without outliers 
        self.name = name
        self.topic_share = []

        self.evaluate()             # create embeddings 
        self.get_accuracy()                 # get accuracy

    # given name of the model compute the embeddings, if name is openai use openai api 
    # else only sentence tranformer are allowed
    def evaluate(self):
        print('evaluate', self.name)
        model = ''


        if(self.name == 'openai'):
            embs = openai.Embedding.create(input = self.docs, model="text-embedding-ada-002")['data']
            self.embeddings = np.array([np.array(emb['embedding']) for emb in embs])
            model = 'bertopic'           # create model
           
        
        elif(self.name == 'NMF'):
            #self.get_NMF()
            model = 'nmf'
        
        elif(self.name == 'GSDMM'):
            #self.get_GSDMM()
            model = 'gsdmm'

        elif(self.name == 'USE'):
            embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            self.embedder = embedding_model
            self.get_topic_model() 
            model = 'bertopic'           # create model

        else:
            self.embedder = SentenceTransformer(self.name)
            self.embeddings = self.embedder.encode(self.docs)
            model = 'bertopic'           # create model
            #self.get_topic_model()            # create model

        
        for i in range(self.n_iter):
            if model == 'bertopic':
                self.get_topic_model(i)            # create model
                #self.get_accuracy()                 # get accuracy

            elif model == 'nmf':
                self.get_NMF(n = i)
                #self.get_accuracy()                 # get accuracy

            elif model == 'gsdmm':
                self.get_GSDMM(n = i)
                #self.get_accuracy()                 # get accuracy
            

    # create topic model with bertopic and update the dataframe with the inferred topics 
    def get_topic_model(self, n ):

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
        self.df['my_topics_'+str(n)] = topics
        #self.df['my_probs'] = probs

        return model

    def get_NMF(self, n,  max_df = 0.95, min_df = 3, ngram_range = (1,2)  ):
 

        self.df['preprocessed'] = self.df['text'].apply(preprocess_text)
        tfidf = TfidfVectorizer(stop_words='english', max_df = max_df, min_df = min_df, ngram_range = ngram_range)
        dtm = tfidf.fit_transform(self.df['preprocessed'])


        nmf_model = NMF(n_components= len(self.df['topic'].unique()), random_state=42)

        topics = nmf_model.fit_transform(dtm)



        self.model = nmf_model
        self.df['my_topics_'+str(n)] = topics.argmax(axis=1)

    def get_GSDMM(self,n = 0, alpha = 0.1, beta = 0.1, n_iters = 30):

        if self.nr_topics == 'auto':
            self.nr_topics = len(self.df['topic'].unique())

        mgp = MovieGroupProcess(K= self.nr_topics, alpha= alpha, beta=beta, n_iters=n_iters)
        lemmatizer = nltk.WordNetLemmatizer()
        vectorizer = TfidfVectorizer()

        self.df['tokens'] = self.df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
        self.df['tokens'] = self.df['tokens'].apply(lambda x: nltk.word_tokenize(x))
        self.df['tokens'] = self.df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        docs = self.df['tokens'].tolist()
        vocab = set(x for doc in docs for x in doc)
        n_terms = len(vocab)
        y = mgp.fit(docs, n_terms)

        self.model = mgp
        self.df['my_topics_'+str(n)] = y


    def get_accuracy(self):
        df = self.df


        for i in range(self.n_iter):

            topics = df['topic'].unique()
            my_topics = df['my_topics_'+str(i)].unique()
            results = {}
            results_no_outliers = {}
            topic_share = {}
            # every my_topic should have >77 % pf the documents of topic
            for topic in my_topics:
                res = df[df['my_topics_'+str(i)] == topic].value_counts('topic') 
                if topic != -1:
                    topic_share[topic] = round(res[0]/ sum(res) ,2)
                    

            # compute accuracy for each topic
            if (min(topic_share.values()) < 0.7):
                print('topic share is too low, proprablt accuracy is not meaningful for ', min(topic_share))
                print('check the heatmap ')
            results['min_topic_share'] = min(topic_share.values())

            for topic in topics:
                res = df[df['topic'] == topic].value_counts('my_topics_'+str(i))
                first = res.iloc[0] if res.index[0] != -1 else res.iloc[1]                       # i'm assuming that out of the possible label the right one is the biggest 
                missed = sum(res.iloc[1:]) if res.index[0] != -1 else sum(res) - res.iloc[1]    # sum of the other labels
                try :
                    outliers = res.loc[-1]
                except:
                    outliers = 0

                
                
                results[topic] = first / (first + missed)
                results_no_outliers[topic] = first / (first + missed - outliers)  # bertopic mark the outliers with -1, i do not consider them while computing accuracy
            
            
            self.accuracy.append({
                'accuracy': results,
                'accuracy_no_outliers': results_no_outliers,
                'topic_share': topic_share
            }) 
            

        return results
    
    def visualize_documents(self):
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(self.embeddings)
        return self.model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings)

    def visualize_heatmap(self, n=0):
        return sns.heatmap(pd.crosstab(self.df['topic'], self.df['my_topics_'+str(n)]), annot=True, cmap="YlGnBu", fmt='g').set_title(self.name)
    
    def visualize_min_topic_share(self):

        # create a figure and axis
        min_topic_size = [acc['accuracy']['min_topic_share'] for acc in self.accuracy]

        fig, ax = plt.subplots()
        #bar plot
        ax.bar(range(len(min_topic_size)), min_topic_size) 

        ax.set_xlabel('iteration')
        ax.set_ylabel('min topic share')
        ax.set_title( self.name)

        # y lim 0-1 
        ax.set_ylim(0,1)

        # x ticks
        ax.set_xticks(range(len(min_topic_size)))

        



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
# find differences between columns 
