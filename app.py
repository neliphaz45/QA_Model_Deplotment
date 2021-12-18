# import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
import numpy as np
import pandas as pd

from tfidf_question_answer_online import QuestionAnswer


app = Flask(__name__)
# model = joblib.load(open('question_answer.pkl', 'rb'))


# Getting data
# """Training the data"""
# # Importing the csv data
# train=pd.read_csv("question_answer.csv")


# qst_list=train['questions'].tolist()
# ans_list=train['answer_texts'].tolist()
# # qst_list

# # unpacking list of lists into one list for questions
# flat_list_qst = []
# for sublist in qst_list:
#     for item in sublist:
#         flat_list_qst.append(item)

# # unpacking list of lists into one list for answers
# flat_list_ans = []
# for sublist in ans_list:
#     for item in sublist:
#         flat_list_ans.append(item)



# model = QuestionAnswer(context= flat_list_qst, answers= flat_list_ans)
  
@app.route('/')      # my default homepage
def home():
    return render_template('index.html')     # where your input needs to be


@app.route('/predict/', methods=['GET','POST'])
def predict():

    if request.method == "POST":
        #get form data
        question = request.form.get('question')
        #call preprocessInputData and pass inputs
        preproc_question = preprocessInputData(question)
        
        # Need to retrain the model at every post
        """Training the data"""
        # Importing the csv data
        train=pd.read_csv("question_answer.csv")


        qst_list=train['questions'].values.tolist()
        ans_list=train['answer_texts'].values.tolist()
        # qst_list


        # Removing the special characters to easy training

        alldata = []
        def removesigns(list_data):
            for elemnt in list_data:
                new_string = elemnt.replace('[', "")
                new_string = new_string.replace(']', "")
                new_string = new_string.replace('\'', "")
                new_string = new_string.split(",")
                alldata.append(new_string)
            return alldata

        # function
        flat_list = []
        def flatten_list(data):
            # iterating over the data
            for element in data:
                # checking for list
                if type(element) == list:
                    # calling the same function with current element as new argument
                    flatten_list(element)
                else:
                    flat_list.append(element)
            return flat_list



        # removing special characters
        alldata = []
        flat_list = []
        flat_list_qst = removesigns(qst_list)
        flat_list_qst = flatten_list(flat_list_qst) # flattening the given list

        alldata = []
        flat_list = []
        flat_list_ans = removesigns(ans_list)
        flat_list_ans = flatten_list(flat_list_ans)


        print(flat_list_qst[10:20])
        print(flat_list_ans[10:20])


        model = QuestionAnswer(context= flat_list_qst, answers= flat_list_ans)

        # model = QuestionAnswer(context=["hello there, sup, holla", "are you doing ML class", "is it okay to present after a week"],
        #                          answers=["Hello", "Yes am doing ML class" ,"yes you have been approved to present next week"] )

        # score = np.array([0.5])[indices.astype(int)]
        answer =  model.get_answer_percontext([preproc_question])

        #pass prediction to template
        return render_template('index.html', prediction_text = 'The answer to this question: \n "{}" \n is: "{}"'.format(question, answer))
    pass

def preprocessInputData(input_question):
    #pre processing steps like lower case, stemming and lemmatization
    input_question = input_question.lower()
    stop = stopwords.words('english')
    input_question = " ".join(x for x in input_question.split() if x not in stop)
    st = PorterStemmer()
    input_question = " ".join ([st.stem(word) for word in input_question.split()])
    input_question = " ".join ([Word(word).lemmatize() for word in input_question.split()])
    return input_question

if __name__ == "__main__":
    app.run(debug=True)