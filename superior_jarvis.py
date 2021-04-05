# used a dictionary to represent an intents JSON file
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
              "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"],
             },
             {"tag": "age",
              "patterns": ["how old are you?", "when is your birthday?", "when was you born?"],
              "responses": ["I am 24 years old", "I was born in 1996", "My birthday is July 3rd and I was born in 1996", "03/07/1996"]
             },
             {"tag": "date",
              "patterns": ["what are you doing this weekend?",
              "do you want to hang out some time?", "what are your plans for this week"],
              "responses": ["I am available all week", "I don't have any plans", "I am not busy"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "what are you called?", "who are you?"],
              "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "g2g", "see ya", "adios", "cya"],
              "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
             }
]}

import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Dense, Dropout
# print('Imports successfull')

nltk.download("punkt")
nltk.download("wordnet")

# Inicializing lemmatizer to get steam of words
lemmatizer = WordNetLemmatizer()

# Each list to create
words = []
classes = []
doc_x = []
doc_y = []

# Loop throught all the intents
# tokenize each pattern and append token to words, the pattern and
# the associated tag to their associated list

for intent in data["intents"]:
  if intent["tag"] not in classes:
    classes.append(intent["tag"])
  for pattern in intent["patterns"]:
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    doc_x.append(pattern)
    doc_y.append(intent["tag"])

# Lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Sortering the vocab and classes in alphabetical order and taking the
# set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

"""
print(words)
print(classes)
print(doc_x)
print(doc_y)

"""

# List for training data
training = []
out_empty= [0] * len(classes)

# Creating the bag of words model
for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if text else bow.append(0)

    # mark the index of class that current pattern is associated to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    # add the one hot encoder BoW and associated classes to training
    training.append([bow, output_row])

# Shuffle the data and convert into an array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the features and target labels
train_x = np.array(list(training[:, 0]),)
train_y = np.array(list(training[:, 1]))

# Define some parameters
input_shape = (len(train_x[0]),)
output_shape = len(train_y[0])
epochs = 200

# The deep learning model
model = Sequential()
model.add(layers.Dense(128, input_shape = input_shape, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(output_shape, activation = 'softmax'))

adam = tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6)


model.compile(loss= 'categorial_crossentropy', optimizer= adam, metrics = ['accuracy'])

# model.fit not working
#model.fit(x = train_x, y = train_y, epochs=epochs)

# print(model.summary())

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result


while True:
    menssage = input('')
    intends = pred_class(menssage, words, classes)
    result = get_response(intends, data)
    print(result)

