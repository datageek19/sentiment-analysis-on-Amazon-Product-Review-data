import glob
import csv
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Softmax
from keras.preprocessing import sequence
import time
import pickle
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


stopwords = set(stopwords.words('english'))

# == config options =====
stopword_removal = True  # True, False
classwise_sampling = 'equal'  # 'equal' - take equal number of samples from each class, 'all' - take all samples

max_features = 1024
maxlen = 128
# =======================

np.random.seed(81)
num_classes = 3

# file structure not same
# some ratings missing
# some titles replaced by dummy titles
# some titles missing

# deleted rows with missing ratings

x = []
y = []
star_counts = [0, 0, 0, 0, 0]

file_name = 'file_1.csv'

print('reading file')
trf0 = time.time()
x_file_name = 'x_preproc_{}.pkl'.format(stopword_removal)
y_file_name = 'y_preproc.pkl'


skip_reading = False
if os.path.exists(x_file_name) and os.path.exists(y_file_name):
    skip_reading = True

if not skip_reading:
    with open(file_name, 'rt') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()   # ignore column headers
        for row in csv_reader:
            review_title = row[2]
            review_text = row[1]
            review_stars = row[0]
            text = review_title + ' ' + review_text
            tokenized = word_tokenize(text.lower())
            if stopword_removal:
                # filtered = [i for i in tokenized if i not in stopwords]
                filtered = list(filter(lambda a: a not in stopwords, tokenized))
            else:
                filtered = tokenized
            x.append(filtered)      # currently concatenating. Might be better to keep them separate
            review_stars = int(review_stars)
            star_counts[review_stars-1] += 1
            y.append(review_stars)
    print('read in {} s'.format(time.time() - trf0))

    # print counts
    for i in range(len(star_counts)):
        print("Number of reviews with {} stars: {}".format(i+1, star_counts[i]))

    # combine 1 and 2
    # combine 4 and 5

    print("\nCombining classes to convert it to a 3-class problem.")
    print("Bad: 1- and 2-star")
    print("Neutral: 3-star")
    print("Good: 4- and 5-star")

    y = np.array(y)
    y[y == 1] = 0
    y[y == 2] = 0
    y[y == 3] = 1
    y[y == 4] = 2
    y[y == 5] = 2

    pickle.dump(x, open(x_file_name), 'wb')
    pickle.dump(y, open(y_file_name), 'wb')

else:
    x = pickle.load(open(x_file_name, 'rb'))
    y = pickle.load(open(y_file_name, 'rb'))
    print("Preprocessed files available, reading from stored file.")


class_counts = [np.sum([y == 0]), np.sum([y == 1]), np.sum([y == 2])]

print('\nCounts')
print('Bad:\t\t{}'.format(class_counts[0]))
print('Neutral:\t{}'.format(class_counts[1]))
print('Good:\t\t{}'.format(class_counts[2]))

indices = np.arange(len(x))
classwise_split_indices = []
for class_num in range(3):
    class_indices = indices[y == class_num]
    if classwise_sampling == 'equal':
        class_indices = np.random.choice(class_indices, min(class_counts), replace=False)
    elif classwise_sampling == 'all':
        pass
    else:
        raise Exception("Invalid classwise_sampling choice.")
    train_indices, test_indices = train_test_split(class_indices, test_size=0.4)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5)
    classwise_split_indices.append([train_indices, test_indices, val_indices])

classwise_split_indices = np.array(classwise_split_indices)
train_indices = np.concatenate(classwise_split_indices[:, 0])
val_indices = np.concatenate(classwise_split_indices[:, 1])
test_indices = np.concatenate(classwise_split_indices[:, 2])

x = np.array(x)
y = np.array(y)
y = keras.utils.to_categorical(y, num_classes=3)

x_train = x[train_indices]
x_val = x[val_indices]
x_test = x[test_indices]

y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]


tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
#tokenizer.fit_on_sequences(x_train)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

#x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
#x_val = tokenizer.sequences_to_matrix(x_val, mode='binary')
#x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# x_train = tokenizer.sequences_to_matrix(x_train, mode='tfidf')
# x_val = tokenizer.sequences_to_matrix(x_val, mode='tfidf')
# x_test = tokenizer.sequences_to_matrix(x_test, mode='tfidf')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

model.fit(x_train, y_train,
          batch_size=64,
          epochs=16,
          validation_data=(x_val, y_val),
          callbacks=[earlystop])#, tensorboard])
