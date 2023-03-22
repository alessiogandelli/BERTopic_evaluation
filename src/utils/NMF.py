#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text


from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

import re
import string

from supervised import get_test_dataset



# %%

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




#%%
path = './../../data/simple_supervised/'
df = get_test_dataset(path)
df['preprocessed'] = df['text'].apply(preprocess_text)

#%%
df['no_hashtags'] = df['preprocessed'].str.replace(r'#\S+', '', case=False)





#%%
tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=3, ngram_range=(1,2))
dtm = tfidf.fit_transform(df['preprocessed'])
dtm = tfidf.fit_transform(df['no_hashtags'])


# %%
nmf_model = NMF(n_components=5, random_state=42)

topics = nmf_model.fit_transform(dtm)
# %%


for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    print('\n')
# %%


df['my_topics'] = topics.argmax(axis=1)
# %%
sns.heatmap(pd.crosstab(df['topic'], df['my_topics']), annot=True, cmap="YlGnBu", fmt='g')
# %%
