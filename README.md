# Evaluate topic modeling

in this repository, we evaluate the the following topic modeling methods, using both a Supervised and Unsupervised methods



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

## Metrics 
for each of these models, we compute the following metrics:
- npmi: degree of association between the top words in a topic.
- umass: degree of semantic similarity between the top words in a topic.
- diversity: how distinct the topics are from each other
- computational time: time to compute the model


## Datasets used

climate: 1669 preprocessed tweets with the hastag #cop22:
- removed: retweets, links, punctuation, #cop22 hashtag, #climatechange hashtag
- only english 

TODO: add the other datasets

## Results
you can see the results in [this](https://github.com/alessiogandelli/BERTopic_evaluation/tree/main/myresults) folder

## Methods 
for each model we fitted the model with different parameters, in particular
- number of topics: from 10 to 50 with a step of 5 

all these test have been computed 3 times and then averaged


## Next steps
Note that these evaluation metrics tells just a part of the story.
In order to better evaluate these models we need to investigate more in a supervised way:

we have to define a test dataset already labeled: 
First, we will use a general tweets dataset with several different topics( e.g. politics, sports, technology ) in order to see how it perform on simple tasks. 
Then, we will take a topic( or more) with several subtopics, to see how it perform in this context .


### generate supervised dataset

#### simple supervised 

twarc2 search --limit 200 "#stormydaniels -is:retweet lang:en " trump.jsonl
twarc2 search --limit 200 "#Bitcoin -is:retweet lang:en" btc.jsonl  
twarc2 search --limit 200 "#socialscience -is:retweet lang:en" socialscience.jsonl
twarc2 search --limit 200 "#UkraineRussianWar -is:retweet lang:en" war.jsonl
twarc2 search --limit 200 "#SaudiArabianGP -is:retweet lang:en" formula1.jsonl 

#### politics supervised
twarc2 search --limit 200 "#IndictArrestAndConvictTrump -is:retweet lang:en" trump.jsonl    
twarc2 search --limit 200 "#kabul -is:retweet lang:en" kabul.jsonl   
twarc2 search --limit 200 "#BidenHarris2024 -is:retweet lang:en" biden_harris2024.jsonl  
twarc2 search --limit 200 "#belarus -is:retweet lang:en" belarus.jsonl   
twarc2 search --limit 200 "#taiwan -is:retweet lang:en" taiwan.jsonl     
twarc2 search --limit 200 "#KamalaHarris -is:retweet lang:en" kamala.jsonl   