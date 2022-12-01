import sys
sys.path.append("..")
from dataset.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class TokenAligner():
    def __init__(self, tokenizer: AutoTokenizer, vocab: Vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab
    
    """
    params:
        text  ----  str
    """
    def _char_tokenize(self, text):
        characters = list(text)
        tokens = [ token + "@@" if i < len(characters) - 1 and characters[i + 1] != " " else token for i, token in enumerate(characters)]
        token_ids = self.tokenizer.encode(tokens, return_tensors = "pt").squeeze(0)
        return tokens, token_ids
    
    def char_tokenize(self, batch_texts):
        doc = dict()
        doc['tokens'] = []
        doc['token_ids'] = []
        for text in batch_texts:
            tokens, token_ids = self._char_tokenize(text)
            doc['tokens'].append(tokens)
            doc['token_ids'].append(token_ids)
        return doc

    def tokenize_for_transformer_with_tokenization(self, batch_noised_text, batch_label_texts = None):
        batch_srcs = self.char_tokenize(batch_noised_text)['token_ids']
        batch_srcs = pad_sequence(batch_srcs , 
            batch_first=True, padding_value=self.tokenizer.pad_token_id)
            
        if batch_label_texts != None:
            batch_lengths = [len(self.tokenizer.tokenize(text)) for text in batch_label_texts]
            batch_tgts = self.tokenizer.batch_encode_plus(batch_label_texts, max_length = 512, 
                    truncation = True, padding=True, return_tensors="pt")['input_ids']
            return batch_srcs, batch_tgts, batch_lengths
        
        return batch_srcs