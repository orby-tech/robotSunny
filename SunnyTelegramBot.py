import telebot
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import pickle
import warnings


warnings.filterwarnings('ignore')
tag = ''
string = "dfgdfgdfgdfgdfgdfgfdgg"


def func_to_detect_empty_or_no_understended_messege(str):
	if str:
		print("bot: " + str)
		print("")
	else:
		array_of_answers = ["I think you mean something else",
							"I'm a robot, and I’m talking more clearly",
							"I realized that I didn’t understand anything"]
		str = random.choice(array_of_answers)
		print("bot: " + str)
		print("")
	return str


def prelued():
	st = LancasterStemmer()
	last_string = ['', '', '']
	last_tag = ['', '', '']
	extation = True
	ERROR_TRESHOLD = 0.01

	data = pickle.load(open("training_data", "rb"))
	words = data['words']
	classes = data['classes']
	train_x_y = [data['train_x'], data['train_y']]

	with open('intents.json') as json_data:
		intents = json.load(json_data)

	net = tflearn.input_data(shape=[None, len(train_x_y[0][0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_x_y[1][0]), activation='softmax')
	net = tflearn.regression(net)
	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

	model.load('./model.tflearn')

	return words, classes, train_x_y, intents, model, last_tag, last_string, ERROR_TRESHOLD, st, extation


def check_mess_by_control_points(sentence_words, string):
	status = "0"
	try:
		if sentence_words[0] + sentence_words[1] == 'imean':
			str = "ok, i wrote down"
			status = "imean"
		elif sentence_words[0] + sentence_words[1] + sentence_words[3] == "whatdothink":
			str = "Now, i can't think, i can calculate, heh"
			status = "whatdothink"
		else:
			str = ''
	except:
		str = ''
	return str, status


def cleanup_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_new_words = [st.stem(word.lower()) for word in sentence_words]
	return sentence_new_words


def numerate_string_for_tenzor(sentence, words, show_details=False):
	sentence_words = cleanup_sentence(sentence)
	bag = [0] * len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i] = 1

	return (np.array(bag))


def classify(sentence, model):
	results = model.predict([numerate_string_for_tenzor(sentence, words)])[0]
	results = [[i, r] for i, r in enumerate(results) if r > ERROR_TRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append((classes[r[0]], r[1]))
	return return_list


def response(sentence, model, userID="123", show_details=True):
	results = classify(sentence, model)
	if results:
		while results:
			for i in intents['intents']:
				if i['tag'] == results[0][0]:
					return (random.choice(i['responses'])), (i['tag'])
			results.pop(0)



words, classes, train_x_y, intents, model, last_tag, last_string, ERROR_TRESHOLD, st, extation = prelued()


def composit(string):

	p = numerate_string_for_tenzor(string, words)
	sentence_words = cleanup_sentence(string)

	str, status = check_mess_by_control_points(sentence_words, string)
	answer = str
	if status == "imean":
		#print(" ".join(sentence_words[2:]) +"=="+ last_1_string +", tag:" + last_1_tag)
		tag = "imean"
	elif status == "0":
		str, tag = response(string, model)
		print("sf", tag)
		answer = func_to_detect_empty_or_no_understended_messege(str)
	elif status == "whatdothink":
		tag = "whatdothink"
	print("###" + string)
	print("@@@" + answer)
	return answer


telebot.apihelper.proxy = {'https': 'https://31.193.196.3:3128'}
bot = telebot.TeleBot('1095075688:AAHEOWn_WxId8b_5UBXg5u-jXon1kb4tV2w')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'hi')


@bot.message_handler(content_types='text')
def send_text(message):
	try:
		answerBot = composit(message.text)

		if len(answerBot) >= 25 and " " not in answerBot:
			bot.send_sticker(message.chat.id, answerBot)

		else:
			bot.send_message(message.chat.id, answerBot)
	except:
			pass


bot.polling()
