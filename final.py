from database import MongoDB
from gen_ques_similarity import GeneratedQuestionSimiilarity
from translatesimilarity import TranslatedQnSimiilarity
from translation import Translation
import warnings
warnings.filterwarnings("ignore")


mongo_connect = MongoDB(LANGUAGE='ENGLISH')
#english = mongo_connect.find_queries('ENGLISH')
english_list = mongo_connect.json_to_list()
print(english_list[:3])

input_query = 'In which age Uzziah became king?'

check = GeneratedQuestionSimiilarity(question=input_query,corpus=english_list)
check.check_similarity()

new_lang = Translation(query=input_query)
print(new_lang.translate())
# print(new_lang.translate())
translated_query = ' '.join(new_lang.translate())
print(translated_query)

## German Corpus
german = MongoDB(LANGUAGE='GERMAN')

german_list = german.json_to_list()
print(german_list[:3])

check_german = TranslatedQnSimiilarity(query=translated_query,corpus=german_list)
check_german.check_similarity()

