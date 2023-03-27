#%%
from top2vec import Top2Vec
import pandas as pd
import jsonlines

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


#%%
df = get_test_dataset('/Users/alessiogandelli/dev/internship/BERTopic_evaluation/data/simple_supervised')
# %%
model = Top2Vec(df['text'].values, speed="deep-learn", workers=4)

# %%
import tensorflow_hub
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# %%

from tweetopic import DMM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import topicwizard
# Creating a vectorizer for extracting document-term matrix from the
# text corpus.
vectorizer = CountVectorizer(min_df=15, max_df=0.1)

# Creating a Dirichlet Multinomial Mixture Model with 30 components
dmm = DMM(n_components=30, n_iterations=100, alpha=0.1, beta=0.1)

# Creating topic pipeline
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("dmm", dmm),
])

# Fitting the pipeline to the corpus
topics = pipeline.fit(df['text'].values)

# %%
#get nan texts 
nan_texts = df[df['text'].isna()]
# %%

from gsdmm import MovieGroupProcess

# %%
mgp = MovieGroupProcess(K=5, alpha=0.1, beta=0.1, n_iters=30)

# find nan in corpus
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

lemmatizer = nltk.WordNetLemmatizer()
vectorizer = TfidfVectorizer()


df['tokens'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
df['tokens'] = df['tokens'].apply(lambda x: nltk.word_tokenize(x))
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
docs = df['tokens'].tolist()
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
y = mgp.fit(docs, n_terms)

df['my_topic'] = y


import seaborn as sns

# confusion matrix for my topic
sns.heatmap(pd.crosstab(df['topic'], df['my_topic']), annot=True, fmt="d", cmap="YlGnBu")

# %%
