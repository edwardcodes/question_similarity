import pymongo
import pandas as pd


## Connect Database
class MongoDB(object):

    def __init__(self,LANGUAGE = None):
        URI = "mongodb+srv://edward:Deepml123@mydb.s3awvun.mongodb.net/test"
        client = pymongo.MongoClient(URI)
        self.db = client["biblist"]
        self.collection = self.db["questions"]
        self.LANGUAGE = LANGUAGE

    def find_queries(self):
        return self.collection.find({'language': self.LANGUAGE},{'_id':0,'rawUniqueId':1,'text':1})
    
    def json_to_list(self):
        questions = []
        for q in self.find_queries():
            questions.append(q)

        ## creating dataframe
        df = pd.DataFrame(questions)
        df_list = df.text.tolist()
        return df_list

# mongo_connect = MongoDB(LANGUAGE='GERMAN')
# #english = mongo_connect.find_queries('ENGLISH')
# english_df = mongo_connect.json_list()
# print(english_df.head())
