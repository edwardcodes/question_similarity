#from requests import request
from database import MongoDB
from gen_ques_similarity import GeneratedQuestionSimiilarity
from translatesimilarity import TranslatedQnSimiilarity
from translation import Translation
import warnings
warnings.filterwarnings("ignore")
#from pydantic import BaseModel
#import uvicorn
#from fastapi import FastAPI
from flask import Flask,render_template,request

app = Flask(__name__)

# class Data(BaseModel):
#     question: str

@app.route('/')
def load_html():
    return render_template('index.html')
@app.route('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


@app.route('/predict',methods=['POST'])
def predict():
    data=request.form
    text=data['text'] 
     
    mongo_connect = MongoDB(LANGUAGE='ENGLISH')
    #english = mongo_connect.find_queries('ENGLISH')
    english_list = mongo_connect.json_to_list()
    #print(english_list[:3])

    #input_query = 'In which age Uzziah became king?'

    check = GeneratedQuestionSimiilarity(question=text,corpus=english_list)
    check.check_similarity()

    new_lang = Translation(query=text)
    print(new_lang.translate())
    # print(new_lang.translate())
    translated_query = ' '.join(new_lang.translate())
    print(translated_query)

    ## German Corpus
    german = MongoDB(LANGUAGE='GERMAN')

    german_list = german.json_to_list()
    #print(german_list[:3])

    check_german = TranslatedQnSimiilarity(query=translated_query,corpus=german_list)
    check_german.check_similarity()

    return {'success':True,'data':{'Question':text,'English Similarity':check.check_similarity(),\
        'translated_query':translated_query,'German Similarity':check_german.check_similarity()}
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)