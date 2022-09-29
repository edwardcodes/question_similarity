from sentence_transformers import SentenceTransformer, util
import torch

class GeneratedQuestionSimiilarity(object):

    def __init__(self, question,corpus):
        self.model = SentenceTransformer('./models/sentence-transformers_all-MiniLM-L6-v2/')
        #self.model = SentenceTransformer('all-MiniLM-L6-v2',cache_folder='./models/')
        self.question = question
        self.corpus = corpus

    def check_similarity(self):
        ques_embedding = self.model.encode(self.question, convert_to_tensor=True)
        corpus_embedding = self.model.encode(self.corpus,convert_to_tensor=True)

        # check whether the question's context is matching
        top_k = min(3,len(self.corpus))

        cosine_score = util.cos_sim(ques_embedding,corpus_embedding)[0]
        top_results = torch.topk(cosine_score,k=top_k)

        print(f'Input Query: {self.question}')
        #print('\n=======================================')
        print('\nTop most questions with same context:\n')
        

        for score, idx in zip(top_results[0], top_results[1]):
            if score >=0.70:
               return f'{self.corpus[idx]} with score: {score:.4f}'
