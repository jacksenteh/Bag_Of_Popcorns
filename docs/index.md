---
layout: default
title: Sentiment analysis on IMDB movie reviews (Kaggle)
---

# Data Wrangling
The dataset consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis and was 
split into:
* labeledTrainData.tsv
* testData.tsv
* unlabeledTrainData.tsv (extra training set with no labels)

All three dataset have the following properties:
* The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. 
* For each movie, there contain multiple reviews with no more than 30 reviews in total.
* labeledTrain and testData contain 25,000 review each with labeled and unlabeled sentiments.

With that in mind, we can first assess the data quality to determine the level of data engineer required 
(e.g., data wrangling, cleansing and preparation).
```bash
import re
import nltk
import sklearn
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt

from pprint              import pprint
from bokeh.io            import output_notebook, show
from bokeh.models        import ColumnDataSource
from bokeh.palettes      import Spectral6
from bokeh.plotting      import figure
from lib.data_preprocess import Preprocess
from nltk.corpus         import stopwords 
from nltk.tokenize       import word_tokenize
from nltk.stem           import WordNetLemmatizer
from bs4                 import BeautifulSoup
from PIL                 import Image
from wordcloud           import WordCloud, STOPWORDS, ImageColorGenerator
from tqdm                import tqdm_notebook as tqdm
from scipy               import stats

# plot setting
output_notebook()
sns.set_style('whitegrid')
param = {'figure.figsize': (16, 8),
         'axes.titlesize': 18,
         'axes.labelsize': 16}
plt.rcParams.update(param)

# ntlk setting
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# define the dataset path
train_path = 'dataset/labeledTrainData.tsv'
test_path  = 'dataset/testData.tsv'

# read in the dataset
pp                = Preprocess()
train_df, test_df = pp.read_file(train_path, test_path, sep='\t')

# check for missing value
print('Training set:')
pp.check_na(train_df)
print()

print('Test set:')
pp.check_na(test_df)

``` 

#### Output:
![missing_val](images/missing_val.png)

Nice, both the train and test dataset is complete. However, the datatype of *id* is an object rather than int. 
Due to the lack of information, it is difficult for us to assess the validity of this column.
But since the this column will not affect the model accuracy, we can just check the uniqueness and disregard it for now.

The next tasks is to process the reviews, text normalization and check the data distribution. 
To clean the reviews, we can take the following steps:
* converting all letters to lower case.
* remove excessive white spaces.
* remove non-words (e.g., punctuation and weird symbols).
* remove html tags.

```bash
# convert it to lower space
def text_process(df):
    # convert it to lower space
    df.review = df.review.str.lower()

    # pattern
    nw_removal  = lambda review: re.sub(r'[^A-Za-z\s]' , ' ', review)
    spc_removal = lambda review: re.sub(r'[\s]+'      , ' ', review)
    tag_removal = lambda review: BeautifulSoup(review, 'lxml').get_text()

    # replace words or substrings which match the pattern
    df.review = df.review.apply(tag_removal)
    df.review = df.review.apply(nw_removal)
    df.review = df.review.apply(spc_removal)
    
    return df


# clean the review
train_df = text_process(train_df)
test_df  = text_process(test_df)

```

After we clean the reviews, the next thing we can do is to convert each word back to it's root.
This can greatly reduce the vocabulary size and help the models identify the root expression of each review.
The techniques that used to find the root are called:

* **Stemming:** “The process of reducing inflection of words to their root forms such as mapping a group of words into a word stem even if the steam is not a valid word in language. “ 
* **Lemmatization:** “The process of ensuring the root word belongs to the language”. For example, (run, ran, running) -> run.

For more details about the stemming and lemmatization techniques, please visit this [tutorial](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python).
In short, the two techniques above are used for text mining (e.g., to extract high-quality information from text) applications.
The applications include **text categorization, text clustering, concept/entity extraction, production of granular taxonomies, sentiment analysis, document summarization, and entity relation modeling**.
 
In this project, we will use the lemmatization techniques to get the root word in proper english language.

```bash
def lemmatize(review):
    lemmatization = WordNetLemmatizer()
    stop_words    = set(stopwords.words('english'))
    
    # tokenize, lemmatize and stop words removal
    tokens            = word_tokenize(review)
    lemmatized_tokens = list(map(lambda x: lemmatization.lemmatize(x, pos='v'), tokens))
    meaningful_tokens = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    
    return meaningful_tokens
    
train_df['review_tokenized'] = list(map(lemmatize, train_df.review.values.copy()))
test_df['review_tokenized']  = list(map(lemmatize, test_df.review.values.copy()))

```

#### Output:
With all the text processing and normalization done, we can check the data distribution to gain som insights on the reviews.
Here, I group the reviews into positive and negative group and utilize TF-IDF to locate the important words that represent each group
(code could be found in [here](https://github.com/jacksenteh/Bag_Of_Popcorns/blob/master/Train.ipynb)).

Most frequent + interesting words in **positive** reviews
![good_wdcloud](images/good_wdcloud_1.png)

Most frequent + interesting words in **negative** reviews:
![bad_wdcloud](images/bad_wdcloud_1.png)

***Please remember that the **size** of the word is equal to the **frequency** of the word.*** 

If you look closely into the negative wordcloud, you might find that when words like 'terrible', 'stupid', 'dead' appears in the review, it usually is a negative review. 
But this is just a speculation based on the current observation, as a data scientist we need to use statistical method to prove this.

The next thing we can check is the distribution of word counts on each review.
![word_dist](images/words_distribution.png)

From the distribution plot above, we can see each review tends to have 50 to 100 words.
Up until this point, we had successfully reduced the number of words in the overall reviews and managed to lemmatize words back to its original roots. 


### Test output
![bigram_phrases](images/bigram.png)

## Word Embedding
It is safe to said we had completed the basics of text normalization. 
The next task to do is word embedding (e.g., converting word to vector space).
In each reviews usually contains multiple phrases. For example, "big apple is a nickname for new york".
When a human read this sentence, they can quickly identify two phrases within the sentence, 'big apple' and 'new york'.
But to a machine learning model, it will not be able to recognize these easily. 
Hence, we can use *Phrases* function from gensim library to identify the phrases.

```bash
from gensim.models import Word2Vec, Phrases

bigrams  = Phrases(sentences=train_df.review_tokenized.values)
trigrams = Phrases(sentences=bigrams[train_df.review_tokenized.values])

```  
After that we can convert each word into vectore by using *Word2Vec*.

```bash
%%time
from gensim.models import Word2Vec


embedded_size = 300
embedding = Word2Vec(sentences=trigrams[bigrams[all_reviews]],
                     size=embedded_size, window=5, min_count=3)

print('Vocabulary size: {}'.format(len(embedding.wv.vocab)))
```
 
 
# Conclusion
While working on this projects, I am struggling whether I should add in the test dataset while performing text normalization and word embedding.
This is because I tried to stimulate an real world environment where you will be not be able to obtain the test dataset while training the model.
After a long self-debate, I figured I can always extend my dataset with reviews from other websites, such as rottentomatoes and letterboxd, so it makes sense to treat the test dataset as additional dataset and include it. 
If any readers do not agree with this rationale, feel free to exclude the test dataset (any suggestions are welcome).