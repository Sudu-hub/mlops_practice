import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os
import logging

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logger.debug('nltk download successfully')
except Exception as e:
    logger.error('pls watch nltk error carefully and debug that')
    raise

def lemmatization(text):
    try:
        lemmatizer= WordNetLemmatizer()

        text = text.split()

        text=[lemmatizer.lemmatize(y) for y in text]

        return " " .join(text)
    except SyntaxError:
        print('learn syntax first')
    
    except Exception as e:
        print(e)

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        print(e)

def removing_numbers(text):
    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        print(e)

def lower_case(text):
    try:
        text = text.split()

        text=[y.lower() for y in text]

        return " " .join(text)
    except Exception as e:
        print(e)

def removing_punctuations(text):
    ## Remove punctuations
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except Exception as e:
        print(e)

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print(e)

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        print(e)

def normalize_text(df):
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug('traformed text successfully and saved in df')
        return df
    except Exception as e:
        logger.error('not saved in df')
        raise

def normalized_sentence(sentence):
    try:
        sentence= lower_case(sentence)
        sentence= remove_stop_words(sentence)
        sentence= removing_numbers(sentence)
        sentence= removing_punctuations(sentence)
        sentence= removing_urls(sentence)
        sentence= lemmatization(sentence)
        return sentence
    except Exception as e:
        print(e)


def save_data(data_path, traindata, testdata):
    try:
        if traindata.empty or testdata.empty:
            raise ValueError('Train and test data is empty')
            
        os.makedirs(data_path, exist_ok=True)
        traindata.to_csv(os.path.join(data_path,'train_processed.csv'))
        testdata.to_csv(os.path.join(data_path,'test_processed.csv'))

    except Exception as e:
        print(e)


def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        if train_data.empty:
            raise ValueError('Train data is Empty')
        test_data = pd.read_csv('./data/raw/test.csv')
        if test_data.empty:
            raise ValueError('Test data is Empty')
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        data_path = os.path.join('data','processed')
        save_data(data_path, train_data,test_data)
        logger.debug('all of saved')
    
    except Exception as e:
        logger.error('pls do it again and save data')
        raise

if __name__ == '__main__':
    main()