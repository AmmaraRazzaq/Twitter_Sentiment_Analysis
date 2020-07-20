# Twitter Sentiment Analysis using Logistic Regression
In this project sentiment analysis of tweets is performed by implementing a logistic regression model that classifies tweets into positive or negative sentiment. Sentiment analyzer
is built in the following steps.

1. Preprocessing of tweets
2. Building frequency dictionary
3. Extracting features
4. Training logistic regression model from scratch.
5. Testing logistic regression model

### Dataset
NLTK library provides Twitter Samples dataset that can be downloaded using `nltk.download('twitter_samples')`. This dataset consists of 10,000 tweets among which 5000 tweets have
positive sentiment and 5000 tweets have negative sentiment.

### Libraries Used
* Pandas, Numpy
* NLTK: Natural Language Processing (NLP) library 

## 0. Split Dataset into Training and Test Dataset
Data is split into 80% training data and 20% test dataset. The training data consists of 50% positive and 50% negative tweets and similarly test data set consists of half 
positive and half negative tweets.

## 1. Preprocess Tweets
Data needs to be preprocessed before feeding into a machine learning algorithm. In an NLP project, preprocessing is done in following steps. 

**1. Removing hyperlinks, twitter marks and styles** 
  * As these donot contribute towards sentiment of a tweet.

**2. Tokenizing and lowercasing**
  * Tokenizing means splitting the string into a list of words without blanks or tabs.

**3. Removing stopwords and punctuation**
  * Stopwords are the words that donot add significant meaning to the text.

**4. Stemming**
  * Stemming means reducing the word to its most general form, or stem. It helps in reducing the size of vocabulory.

## 2. Build Frequencies
Before extracting the features, a frequency dictionary is built using tweets and labels in training data set. Frequency dictionary consists of unique words
in the corpus of tweets after their preprocessing and their corresponding positive and negative frequencies. Positive frequency refers to the number of times a word has appeared
in positive tweets and negative frequency refers to the number of times a word has appeared in negative tweets. Frequency dictionary maps (word, label) pairs to frequencies.
Label is 1 for tweets with positive sentiment and 0 for tweets with negative sentiment.

## 3. Extract Features
A machine learning model expects features in numerical form. The frequency dictionary is used to represent features in numerical form from text data. 

Feature for a tweet = `[1, sum of positive frequencies, sum of negative frequencies]`

A feature for a tweet is a 1x3 vector, where first element is 1 representing the bias, second element is the sum of positive frequencies for every word in the tweet that is 
found in frequency dictionary as (word,1) pair and the third element is the sum of negative frequencies for every word in the tweet that appears in fequency dictionary as 
(word,0) pair.

Features from all the tweets in training data are stored in a mx3 matrix where m is the number of tweets in training dataset.

## 4. Train Logistic Regression Model
In this step, a logistic regression model is trained from scratch. A gradient descent function is used to minimize the cost of training.

1. Initialize the weight vector to zeros <img src="https://render.githubusercontent.com/render/math?math=\theta = [0,0,0]">
2. Apply logistic regression function `h(z)` on tweets to get a prediction.
    * <img src="https://render.githubusercontent.com/render/math?math=h(z) = \frac{1}{1 %2B \exp^{-z}}">
    * <img src="https://render.githubusercontent.com/render/math?math=z = \theta_0 x_0 %2B \theta_1 x_1 %2B \theta_2 x_2 %2B ... \theta_N x_N">
3. Calculate the cost function `J(theta)` using the current weight vector and prediction.
    * <img src="https://render.githubusercontent.com/render/math?math=J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) %2B (1-y^{(i)})\log (1-h(z(\theta)^{(i)}))\tag{5}">
4. update weight vector according to the learning rate alpha `theta = theta - alpha*gradient`
    * <img src="https://render.githubusercontent.com/render/math?math=\theta_j = \theta_j - \alpha \times \nabla_{\theta_j}J(\theta)">
5. Repeat steps 2,3, and 4 for a set number of iterations. 

Use the final weight vector to make predictions.

## 5. Test Logistic Regression
The trained model is used to predict the sentiment of test data. If the prediction value is greater than 0.5, the sentiment is labeled as positive sentiment and if the prediction
value is less than 0.5 the sentiment is labeled as negative sentiment.

The accuracy of the model is determined by calucalating how many tweets have been classified correctly as compared to total number of tweets.

* <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^m (y_{hat}^{(i)} == y^{(i)})/m">
    
    * where, <img src="https://render.githubusercontent.com/render/math?math=y_{hat}^{(i)}"> is predicted label of <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> tweet.
    
    * and <img src="https://render.githubusercontent.com/render/math?math=y^{(i)}"> is actual label of <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> tweet.
    
    * m is total number of tweets.

## Result
The model is 99.5% accurate.  

