import json
import pickle
import nltk
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
model = load_model('mymodel.h5')
intents = json.load(open('intents.json', 'r', encoding='utf-8'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for result in results:
        return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
    return return_list


def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']
        
    # Handle different tags and generate appropriate responses
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = np.random.choice(i['responses'])
            break
    else:
        result = "I'm sorry, I don't understand."
        
    return result

     
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']
    return_list = predict_class(user_message, model)
    bot_response = get_response(return_list, intents)
    return bot_response


if __name__ == '__main__':
    app.run(debug=True)