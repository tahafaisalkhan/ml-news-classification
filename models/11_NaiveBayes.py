# %% [markdown]
# # Model 1: MultiNomial Naive Bayes

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re


# %%
data = pd.read_csv('../data/FINAL_DATASET.csv')

data.info()
print()
data.head()

# %% [markdown]
# ## Preprocessing

# %%
with open('../data/stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    return text

def normalize_unicode(text):
    return re.sub(r'[٠١٢٣٤٥٦٧٨٩]', '', text) 

def normalize_urdu(text):
    text = re.sub(r'[؁؂؃؄؅؆؇؈؉؊؋،؛؟]', '', text)  
    text = re.sub(r'[آإأٱ]', 'ا', text)
    text = re.sub(r'[ىېۍ]', 'ی', text)
    text = re.sub(r'[ۀہ]', 'ہ', text) 
    text = re.sub(r'[ؤو]', 'و', text) 
    text = re.sub(r'[ءئ]', 'ی', text)
    return text

def tokenize_text(text):
    return re.findall(r'\w+', text) 

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

def lemmatize_custom(word):
    if word.endswith('نا') or word.endswith('تے'):
        return word[:-2] 
    elif word.endswith('ا') or word.endswith('ی'):
        return word[:-1]  
    return word

def lemmatize_text(tokens):
    return [lemmatize_custom(word) for word in tokens]

def remove_non_informative_words(tokens):
    return [word for word in tokens if len(word) > 1]  


def preprocess_urdu_text(text):
    text = clean_text(text)  
    text = normalize_unicode(text) 
    text = normalize_urdu(text) 
    tokens = tokenize_text(text) 
    tokens = remove_stopwords(tokens) 
    tokens = lemmatize_text(tokens)
    tokens = remove_non_informative_words(tokens)
    return ' '.join(tokens) 

data['title'] = data['title'].apply(preprocess_urdu_text)
data['content'] = data['content'].apply(preprocess_urdu_text)
data['combined'] = data['title'] + " " + data['content']


# %%
X = data['content'] 
y = data['gold_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
class BagOfWords:
    def __init__(self):
        self.vocabulary = {}  
        self.vocab_size = 0

    def fit(self, corpus):

        unique_words = set()

        for sentence in corpus:
            words = sentence.split() 
            unique_words.update(words)  

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(unique_words))}
        self.vocab_size = len(self.vocabulary)

    def vectorize(self, sentence):

        vector = [0] * self.vocab_size
        
        words = sentence.split()

        for word in words:
            if word in self.vocabulary:
                index = self.vocabulary[word]
                vector[index] += 1

        return vector

# %%
model = BagOfWords()
model.fit(X_train)

vector_training = [model.vectorize(doc) for doc in X_train]
vector_test = [model.vectorize(doc) for doc in X_test]


# %% [markdown]
# ## Implementation

# %%
class MultiNaiveBayes:
    def __init__(self):
        self.classprobs = {}  
        self.wordprobs = {}  
        self.vocab_size = 0    
        self.bow = BagOfWords()                

    def fit(self, X, y):
        self.bow.fit(X)         
        X_vectorized = [self.bow.vectorize(doc) for doc in X] 
        self.vocab_size = self.bow.vocab_size 
        n_docs = len(X)
        
        unique_classes = np.unique(y)  
        class_counts = {c: 0 for c in unique_classes}
        word_counts = {c: np.zeros(self.vocab_size) for c in unique_classes}

        for i in range(n_docs):
            c = y.iloc[i] 
            class_counts[c] += 1
            word_counts[c] += X_vectorized[i]

        for c in unique_classes:
            self.classprobs[c] = class_counts[c] / n_docs
            total_words_in_class = np.sum(word_counts[c])
            self.wordprobs[c] = (word_counts[c] + 1) / (total_words_in_class + self.vocab_size)
    
    def predict(self, X):
        X_vectorized = np.array([self.bow.vectorize(doc) for doc in X])
        predictions = []
        
        for doc_vec in X_vectorized:
            class_scores = {}
            for c in self.classprobs:
                log_prob_c = np.log(self.classprobs[c])
                log_prob_x_given_c = np.sum(doc_vec * np.log(self.wordprobs[c]) + (1 - doc_vec) * np.log(1 - self.wordprobs[c]))
                class_scores[c] = log_prob_c + log_prob_x_given_c
            
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
        
        return predictions


# %% [markdown]
# # Evaluation

# %%
model = MultiNaiveBayes()
model.fit(X_train, y_train) 

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
confusionmatrix = confusion_matrix(y_test, predictions)

print("Manually Implemented Naive Bayes:")
print("Accuracy: ", accuracy*100)
print("Precision: ", precision*100)
print("Recall: ", recall*100)
print("F1 Score: ", f1*100)
print(confusionmatrix)


