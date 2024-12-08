# %% [markdown]
# # Model 2: Neural Network

# %%
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


# %%
data = pd.read_csv('../data/FINAL_DATASET.csv')
data.set_index("id", inplace=True)
data.head()

# %% [markdown]
# # Preprocessing

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
train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

# %%
MAX_VOCAB = 6000
word_counter = Counter(" ".join(train_data["content"]).split())
vocab = [word for word, freq in word_counter.most_common(MAX_VOCAB)]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}


# %%
def vectorize_text(text, vocab, word_to_idx):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector

class MinMaxNormalizer:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


train_vectors = np.array([vectorize_text(text, vocab, word_to_idx) for text in train_data["content"]])
val_vectors = np.array([vectorize_text(text, vocab, word_to_idx) for text in val_data["content"]])
test_vectors = np.array([vectorize_text(text, vocab, word_to_idx) for text in test_data["content"]])
    
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data["gold_label"])
val_labels = label_encoder.transform(val_data["gold_label"])
test_labels = label_encoder.transform(test_data["gold_label"])


# %%
train_dataset = TensorDataset(torch.tensor(train_vectors, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(val_vectors, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(test_vectors, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# # Implementation and Training

# %%
class ArticleClassifier(nn.Module):
    def __init__(self, input_dims, output_dims, dropout_prob=0.4):
        super(ArticleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 1024),  
            nn.ReLU(),                  
            nn.Linear(1024, 128),        
            nn.ReLU(),                  
            nn.Linear(128, 64),           
            nn.ReLU(),
            nn.Linear(64, 32),            
            nn.ReLU(),
            nn.Linear(32, output_dims)  
        )

    def forward(self, x):
        return self.model(x)


# %%
input_size = MAX_VOCAB
num_classes = len(label_encoder.classes_)
model = ArticleClassifier(input_size, num_classes).to("cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    model.eval()
    val_loss = 0
    val_correct_predictions = 0
    val_samples = 0

    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_correct_predictions += (val_predictions == y_val_batch).sum().item()
            val_samples += y_val_batch.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct_predictions / val_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2%}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()


# %% [markdown]
# # Evaluation

# %%
@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()  
    total_correct = 0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    for X_batch, y_batch in data_loader:
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(y_batch.tolist())
        predicted_labels.extend(predictions.tolist())

        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    accuracy = total_correct / total_samples

    return accuracy, true_labels, predicted_labels


# %%
_,all_labels, all_preds = evaluate(model, test_loader)


# %%
cr = classification_report(all_labels, all_preds,output_dict=True)
cm = confusion_matrix(all_labels, all_preds)

print(cr)
print(cm)

# %%
print(f"Accuracy: {cr['accuracy']:.2%}")
print(f"F1-Score: {cr['macro avg']['f1-score']:.2%}")
print(f"Recall: {cr['macro avg']['recall']:.2%}")


