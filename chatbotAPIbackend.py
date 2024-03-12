from flask import Flask, request, jsonify
from flask_cors import CORS
import json, random, pickle, wikipedia, re, time
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed data and model
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('IntellichatModel.h5')

# Function to clean up a sentence by tokenizing and lemmatizing its words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert a sentence into bag of words representation
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))

# Function to predict the intent class of a given sentence
def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True) 
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list

# Function to get a response based on predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

# Function to extract subject from a question
def extract_subject(question):
    # Split the question into words
    punctuation_marks = ['.', ',', '!', '?', ':', ';', "'", '"', '(', ')', '[', ']', '-', '—', '...', '/', '\\', '&', '*', '%', '$', '#', '@', '+', '-', '=', '<', '>', '_', '|', '~', '^']
    # Removing punctuation marks
    for punctuation_mark in punctuation_marks:
        if punctuation_mark in question:
            question = question.replace(punctuation_mark, '')
    
    subject = ''
    words = question.split(' ')
    list_size = len(words)

    for i in range(list_size):
        if i > 1 and i != list_size:
            subject += words[i]+' '
        elif i == list_size:
            subject += words[i]
    return subject

# Function to clean text by removing characters within parentheses
def clean_text(text):
    cleaned_text = re.sub(r'\([^()]*\)', '', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to search Wikipedia for information based on a question
def search_wikipedia(question, num_sentences=2):
    try:
        subject = extract_subject(question)
        wiki_result = wikipedia.summary(subject, auto_suggest=False, sentences=num_sentences)
        return clean_text(wiki_result)
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find information about {subject}."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found. Try being more specific: {', '.join(e.options)}"
    except Exception as e:
        return "Error, Something went wrong!"

# Function to get a response from the chatbot
def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

chat_responses = [
    "How old are you?", "What's your age?", "How are you?", "What's the weather like?",
    "How's the weather today?", "Who are you?", "What are you?", "What's your purpose?",
    "Who created you?", "What technology are you built with?", "How do you work?",
    "What's your underlying technology?", "What programming language are you written in?"
]

@app.route('/chat', methods=['POST'])
def chat():
    chat_responses = [
    "How old are you?", "What's your age?", "How are you?", "What's the weather like?",
    "How's the weather today?", "Who are you?", "What are you?", "What's your purpose?",
    "Who created you?", "What technology are you built with?", "How do you work?",
    "What's your underlying technology?", "What programming language are you written in?"]
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"response": "No message provided"}), 400

    if any(str(a) + symbol + str(b) in user_input for a in [1,2,3,4,5,6,7,8,9,0] for b in [1,2,3,4,5,6,7,8,9,0] for symbol in ['*','/','-','+','**']):
        index = []
        for i in range(len(user_input)):
            if user_input[i].isnumeric():
                index.append(i)
        try:
            ans = str(eval(user_input[index[0]:index[-1]+1]))
            return jsonify({"response": f"Answer is {ans}"}), 200
        except Exception as e:
            return jsonify({"response": f"Error! {e}"}), 500

    # Wikipedia search for questions starting with question words
    elif user_input.lower().startswith(('who', 'what', 'where', 'which', 'when', 'how')) and user_input not in chat_responses:
        wikipedia_result = search_wikipedia(user_input)
        return jsonify({"response": wikipedia_result}), 200

    # Chatbot response for other inputs
    else:
        bot_response = chatbot_response(user_input)
        return jsonify({"response": bot_response}), 200

if __name__ == '__main__':
    app.run(debug=True)





"""from flask import Flask, request, jsonify
from flask_cors import CORS
import json, random, pickle, wikipedia, re, time
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Importing necessary libraries and modules
grn = '\033[32m'  # Green color for user input
blu = '\033[34m'  # Blue color for bot responses

# Load intents data from JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed data and model
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('IntellichatModel.h5')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up a sentence by tokenizing and lemmatizing its words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert a sentence into bag of words representation
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))

# Function to predict the intent class of a given sentence
def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True) 
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list

# Function to get a response based on predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

# Function to extract subject from a question
def extract_subject(question):
    # Split the question into words
    punctuation_marks = ['.', ',', '!', '?', ':', ';', "'", '"', '(', ')', '[', ']', '-', '—', '...', '/', '\\', '&', '*', '%', '$', '#', '@', '+', '-', '=', '<', '>', '_', '|', '~', '^']
    # Removing punctuation marks
    for punctuation_mark in punctuation_marks:
        if punctuation_mark in question:
            question = question.replace(punctuation_mark, '')
    
    subject = ''
    words = question.split(' ')
    list_size = len(words)

    for i in range(list_size):
        if i > 1 and i != list_size:
            subject += words[i]+' '
        elif i == list_size:
            subject += words[i]
    return subject

# Function to clean text by removing characters within parentheses
def clean_text(text):
    cleaned_text = re.sub(r'\([^()]*\)', '', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to search Wikipedia for information based on a question
def search_wikipedia(question, num_sentences=2):
    try:
        subject = extract_subject(question)
        wiki_result = wikipedia.summary(subject, auto_suggest=False, sentences=num_sentences)
        return clean_text(wiki_result)
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find information about {subject}."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found. Try being more specific: {', '.join(e.options)}"
    except Exception as e:
        return "Error, Something went wrong!"

# Function to get a response from the chatbot
def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"response": "No message provided"}), 400

    # Chatbot response for other inputs
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)"""