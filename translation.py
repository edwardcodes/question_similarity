import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class Translation(object):

    def __init__(self,query):
        MODEL_NAME = 'facebook/mbart-large-50-many-to-many-mmt'
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME,cache_dir='./models/')
        self.model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME,cache_dir='./models/')
        self.query = query


    def translate(self):
        self.tokenizer.src_lang = 'en_XX'
        encoded_eng_text = self.tokenizer(self.query, return_tensors='pt',max_length=512)
        generated_tokens = self.model.generate(**encoded_eng_text,forced_bos_token_id=self.tokenizer.lang_code_to_id['de_DE'])
        return self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
    
# def translate(query):
#     MODEL = 'facebook/mbart-large-50-many-to-many-mmt'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MBartForConditionalGeneration.from_pretrained(MODEL).to(device=device)
#     tokenizer = MBart50TokenizerFast.from_pretrained(MODEL,src_lang='en_XX',tgt_lang='de_DE')

#     encoded_eng_text = tokenizer(query, return_tensors='pt',max_length=512)
#     generated_tokens = model.generate(**encoded_eng_text,forced_bos_token_id=tokenizer.lang_code_to_id['de_DE'])
#     return tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)