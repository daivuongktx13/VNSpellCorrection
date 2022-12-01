from transformer import TransformerWithTR
from collator import *
from transformers import AutoTokenizer
import transformers
from models.tokenizer import TokenAligner
from dataset.vocab import Vocab

class ModelWrapper:

    def __init__(self, model, vocab: Vocab):
        self.model_name = model

        if model == "tfmwtr":
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base")
            self.tokenAligner = TokenAligner(self.tokenizer, vocab)
            self.bart = transformers.MBartForConditionalGeneration.from_pretrained("vinai/bartpho-word-base")
            self.model = TransformerWithTR(self.bart, self.tokenizer.pad_token_id)
            self.collator = DataCollatorForCharacterTransformer(self.tokenAligner)
            self.model.resize_token_embeddings(self.tokenAligner)
        else:
            raise(Exception(f"Model {model} isn't implemented!"))
        
