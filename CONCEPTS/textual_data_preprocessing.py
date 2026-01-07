# IMPORTING THE LIBRARIES :
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print(stopwords.words('english'))




# STEMMING : IT IS THE PROCESS OF REDUCING A WORD TO ITS ROOT WORD.
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    