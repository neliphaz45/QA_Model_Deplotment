# import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
# import pickle

# import tfidf_question_answer_deploy


app = Flask(__name__)
model = joblib.load(open('question_answer.pkl', 'rb'))

  

print(model)
@app.route('/')      # my default homepage
def home():
    return render_template('index.html')     # where your input needs to be

@app.route('/predict', method=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    # Here we are getting one input(question) from the user
    input_question = request.form.values()
    print(input_question)
    
    input_question = preprocessInputData(input_question)  # Cleaning the typed text

    answer = model.get_answer_percontext([input_question])   # the model accept the list inputs
    return render_template('index.html', prediction_text = 'The answer to this question is: {}'.format(answer))

def preprocessInputData(input_question):

    #pre processing steps like lower case, stemming and lemmatization
    input_question = input_question.lower()
    stop = stopwords.words('english')

    input_question = " ".join(x for x in input_question.split() if x not in stop)
    st = PorterStemmer()

    input_question = " ".join ([st.stem(word) for word in input_question.split()])
    input_question = " ".join ([Word(word).lemmatize() for word in input_question.split()])

    return input_question






# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     """For direct API calls through request"""
#     data = request.get_json(force=True)
#     predicted_answer = model.get_answer_percontext([data.values()])
#     return jsonify(predicted_answer)



if __name__ == "__main__":
    app.run(debug=True)

