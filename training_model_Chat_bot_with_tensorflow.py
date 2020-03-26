import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

st = LancasterStemmer()


def function_of_clean_json(intents):
	words = []
	words_new = []
	classes = []
	documents = []
	ignore_words = ['?']

	for intent in intents['intents']:
	    for pattern in intent['patterns']:

	        w = nltk.word_tokenize(pattern)
	        words.extend(w)
	        documents.append((w, intent['tag']))

	        if intent['tag'] not in classes:
	            classes.append(intent['tag'])


	for w in words:
		if w not in ignore_words:
			words_new.append(st.stem(w.lower()))

	words = sorted(list(set(words_new)))
	classes = sorted(list(set(classes)))

	print (len(documents), "documents")
	print (len(classes), "classes", classes)
	print (len(words), "unique stemmed words", words)
	return documents, classes, words


def function_of_transform_words_toTensors(documents, classes, words):
	training = []
	output = []
	output_empty = [0] * len(classes)

	for doc in documents:
		bag = []
		pattern_words = doc[0]
		pattern_words = [st.stem(word.lower()) for word in pattern_words]
		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)

		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1

		training.append([bag, output_row])

	random.shuffle(training)
	training = np.array(training)

	train_x = list(training[:, 0])
	train_y = list(training[:, 1])

	print(train_x, train_y)

	return train_x, train_y, output


def function_open_and_read_json():
	with open("intents.json") as json_data:
		intents = json.load(json_data)
	print(intents)
	return intents


def train_a_model(train_x, train_y):

	tf.reset_default_graph()

	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)

	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

	model.fit(train_x, train_y, n_epoch=1500, batch_size=8, show_metric=True)
	model.save('model.tflearn')


documents, classes, words = function_of_clean_json(function_open_and_read_json())

train_x, train_y, output = function_of_transform_words_toTensors(documents, classes, words)

train_a_model(train_x, train_y)


pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))
