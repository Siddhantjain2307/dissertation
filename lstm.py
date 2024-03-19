import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data
human_data = pd.read_table('human_data.txt')

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Apply k-mers conversion
human_data['words'] = human_data['sequence'].apply(lambda x: getKmers(x))
human_data = human_data.drop('sequence', axis=1)

# Join the k-mers to form sentences
human_texts = [' '.join(words) for words in human_data['words']]

# Encode the labels
label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(human_data.iloc[:, 0].values)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(human_texts)
X = tokenizer.texts_to_sequences(human_texts)

# Pad sequences to ensure uniform length
max_sequence_length = max([len(x) for x in X])
X = pad_sequences(X, maxlen=max_sequence_length)

# Define the number of folds for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform cross-validation
for train_index, test_index in skf.split(X, y_data):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    # Define the LSTM model
    embedding_dim = 100
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(128))  # Adjust units as needed
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Calculate mean metrics across folds
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print("Mean accuracy = %.3f \nMean precision = %.3f \nMean recall = %.3f \nMean f1 = %.3f" % (mean_accuracy, mean_precision, mean_recall, mean_f1))
