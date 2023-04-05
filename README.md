# Evaluate topic modeling

in this repository, we evaluate the the following topic modeling methods, using both a Supervised and Unsupervised methods

[presentation](https://docs.google.com/presentation/d/1mujMJvgsh6InJW2cF6Mef4cKrJcSXA6lLysWpDhZ17Y/edit?usp=sharing)



- [LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [NMF](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization)
- [BERTopic](https://arxiv.org/abs/2203.05794)
- [Top2Vec](https://arxiv.org/abs/2008.09470)

for the bertopic model, I used different sentence-transformer for computing the embeddings:

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [openai embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)
- [ClimateBert](https://huggingface.co/climatebert/distilroberta-base-climate-f)
- [tweetclassification](https://huggingface.co/louisbetsch/tweetclassification-bf-model)


[OCTIS](https://github.com/MIND-Lab/OCTIS) library and starting from the work of @MaartenGr :

# Unsupervised 

## Metrics 
for each of these models, we compute the following metrics:
- npmi: degree of association between the top words in a topic.
- umass:  how often two words appear together.
- diversity: how distinct the topics are from each other
- computational time: time to compute the model


## Datasets used

climate: 1669 preprocessed tweets with the hastag #cop22:
- removed: retweets, links, punctuation, #cop22 hashtag, #climatechange hashtag
- only english 

TODO: add the other datasets

## Results
TODO: add the results

## Methods 
for each model we fitted the model with different parameters, in particular
- number of topics: from 10 to 50 with a step of 5 

all these test have been computed 3 times and then averaged


# Supervised 

## Datasets

Tweets have been extracted using twarc2, only english and without retweets. Then for the analysis er used two version of the same dataset: with and without hashtags.

### simple supervised 
1093 labeled tweets of 5 different topics identified by an hashtag (#Bitcoin, #stormydaniels, #UkraineRussianWar, #SaudiArabianGP, #climatechange)
```
twarc2 search --limit 200 "#stormydaniels -is:retweet lang:en " trump.jsonl
twarc2 search --limit 200 "#Bitcoin -is:retweet lang:en" btc.jsonl  
twarc2 search --limit 200 "#socialscience -is:retweet lang:en" socialscience.jsonl
twarc2 search --limit 200 "#UkraineRussianWar -is:retweet lang:en" war.jsonl
twarc2 search --limit 200 "#SaudiArabianGP -is:retweet lang:en" formula1.jsonl 
```

### politics supervised
1492 labeled tweets of 7 politics-related hashtags

```
twarc2 search --limit 200 "#IndictArrestAndConvictTrump -is:retweet lang:en" trump.jsonl    
twarc2 search --limit 200 "#kabul -is:retweet lang:en" kabul.jsonl   
twarc2 search --limit 200 "#BidenHarris2024 -is:retweet lang:en" biden_harris2024.jsonl  
twarc2 search --limit 200 "#belarus -is:retweet lang:en" belarus.jsonl   
twarc2 search --limit 200 "#taiwan -is:retweet lang:en" taiwan.jsonl     
twarc2 search --limit 200 "#KamalaHarris -is:retweet lang:en" kamala.jsonl   
```

## metrics 
- Accuracy: for each known topic look at the biggest of inferred topics and divide by number of tweets in that topic.
- Accuracy no outliers: in the Bertopic case the label -1 refers to outliers.
- Min_topic_share: same as accuracy but in the opposite direction, after having computed it for all of my_topics we take the minimum

## parameters
```
BERTopic : ( nr_topics = 'auto', min_topic_size = 50)
NMF : (max_df = 0.95,  min_df = 3, ngram_range = (1,2) )
GSDMM : (alpha = 0.1,  min_df = 0.1, n_iters = 30 )
````

## Results

TODO: add the results