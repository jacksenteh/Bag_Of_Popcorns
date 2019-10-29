---
layout: default
title: Sentiment analysis on IMDB movie reviews (Kaggle)
---

# Introduction
This is a  sentiment analysis task where the aim is to classify each review's sentiment by estimating the IMDB rating on each movie. 
To make it clearer, the objective is to estimate the movie rating based on each review. 
The review sentiment will be classified as 1 if the estimated rating is >= 7, otherwise 0.

In order to achieve this aim, the objectives are defined as follow:
* **Data Wrangling:** Clean the text (e.g. remove html tags, symbols, etc.), text normalization (e.g., lemmatization), stop word removal, etc.
* **Word Embedding:** Convert each word to vector space.
* **Model Development:** Develop and train a deep learning model to classify each review's sentiment.
* **Optimization:** Check the deeo learning model performance and determine the optimization needed to improve the model performance.

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
The first thing we can check is the distribution of word counts on each review. 
![word_dist](images/words_distribution.png)

From the distribution plot above, we know the most of the reviews had around 120 words. 
This is useful for the later process when we need to pad sequence to the same length.

## Word Embedding
Up until this point, we had successfully cleaned most of the reviews and managed to lemmatize words back to it's original roots.
It is safe to said we had completed the basic of text normalization. 
Hence, we can move forward to word embedding task (e.g., converting word to vector space).
In each reviews usually contains multiple phrases. For example, "big apple is a nickname for new york".
When a human read this sentence, they can quickly identify two phrases within the sentence, 'big apple' and 'new york'.
But to a machine learning model, it will not be able to recognize these easily. 
Hence, we can use *Phrases* function from gensim library to identify the phrases.

```bash
from gensim.models import Word2Vec, Phrases

bigrams  = Phrases(sentences=train_df.review_tokenized.values)
trigrams = Phrases(sentences=bigrams[train_df.review_tokenized.values])

```  

### Test output
![bigram_phrases](images/bigram.png)

After that we can build our embedding matrix by using *Word2Vec*, convert the sentence to vector and set each vector to have same size. 
The embedding matrix basically contain the vectors that represent each word.

```bash
%%time
from gensim.models import Word2Vec

embedded_size = 300
embedding = Word2Vec(sentences=trigrams[bigrams[all_reviews]],
                     size=embedded_size, window=5, min_count=3)

print('Vocabulary size: {}'.format(len(embedding.wv.vocab)))

train  , valid   = pp.train_valid_split(train_df[:25000], split=0.95, seed=0)
train_x, valid_x = train.review_tokenized.values.copy(), valid.review_tokenized.values.copy()
train_y, valid_y = train.sentiment.values.copy()       , valid.sentiment.values.copy()

def vectorized_sentence(reviews, vocab):
    print('Vectorizing words.....', end='\r')
    
    keys           = list(vocab.keys())
    filter_unknown = lambda word  : vocab.get(word, None) is not None
    encode         = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized     = list(map(encode, reviews))
    
    print('Vectorizing words..... (done)')
    
    return vectorized


vocab   = embedding.wv.vocab
train_x = vectorized_sentence(trigrams[bigrams[train_x]], vocab)
valid_x = vectorized_sentence(trigrams[bigrams[valid_x]], vocab)

pad_length  = 150 
train_pad_x = pad_sequences(sequences=train_x, maxlen=pad_length, padding='post')
valid_pad_x = pad_sequences(sequences=valid_x, maxlen=pad_length, padding='post')

```

# Model Development
The model we can use for this problem is a RNN model, specifically the Bidirectional + LSTM (Long Short Term Memory).
Developing the model is relatively straightforward with *keras*. 
```bash
import tensorflow as tf

from keras.models            import Sequential
from keras.layers            import Dense, Dropout, Bidirectional, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers        import Adam
from keras.callbacks         import TensorBoard, LearningRateScheduler, Callback


def Bidirectional_LSTM(embedding_matrix, input_length):
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1], 
                                input_length=input_length,
                                weights=[embedding_matrix],
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def lr_scheduler(epochs):
    lr = 0.03
    
    if epochs > 10  : lr = 0.01;
    elif epochs > 20: lr = 0.003;
    elif epochs > 30: lr = 0.001;
        
    tf.summary_scalar('learning_rate', data=lr, step=epochs)
    
    return lr

class LossCallback(Callback):
    def on_train_begin(self, logs={}):
        self.history = []
        
    def on_batch_end(self, batch, logs={}):
        self.history.append(logs.get('loss'))

net   = Bidirectional_LSTM(embedding.wv.vectors, pad_length)
optim = Adam(lr=0.001)
net.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
net.summary()
```

### Training
After the build the deep learning model, we can just set the training parameters (i.e., batch size and epochs) and start training.
```bash
epochs       = 60
batch_size   = 200
log_dir      = '/content/sample_data/'
tb_callbacks = TensorBoard(log_dir=log_dir + 'accuracy/')
ls_callbacks = LossCallback()

print('Training....', end='\r')

history = net.fit(x=train_pad_x, 
                y=train_y, 
                validation_data=(valid_pad_x, valid_y), 
                batch_size=batch_size, 
                epochs=epcohs,
                verbose=0,
                callbacks=[tb_callbacks, ls_callbacks])

print('Training.... (Done)')
print('Average Loss:', np.average(ls_callbacks.history))
```


 
 
# Conclusion
While working on this projects, I am struggling whether I should add in the test dataset while performing text normalization and word embedding.
This is because I tried to stimulate an real world environment where you will be not be able to obtain the test dataset while training the model.
After a long self-debate, I figured I can always extend my dataset with reviews from other websites, such as rottentomatoes and letterboxd, so it makes sense to treat the test dataset as additional dataset and include it. 
If any readers do not agree with this rationale, feel free to exclude the test dataset (any suggestions are welcome).