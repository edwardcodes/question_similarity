from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translation(object):

    def __init__(self,query):
        MODEL_NAME = 'facebook/mbart-large-50-many-to-many-mmt'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir='./models/')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,cache_dir='./models/')
        self.query = query


    def translate(self):
        self.tokenizer.src_lang = 'en_XX'
        encoded_eng_text = self.tokenizer(self.query, return_tensors='pt',max_length=512)
        generated_tokens = self.model.generate(**encoded_eng_text,forced_bos_token_id=self.tokenizer.lang_code_to_id['de_DE'])
        return self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
    