import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, AutoTokenizer
from params import DEVICE
from models.tokenizer import TokenAligner
from dataset.vocab import Vocab

class TransformerWithTR(nn.Module):
    def __init__(self, bart_model, padding_index) -> None:
        super().__init__()
        self.bart: MBartForConditionalGeneration= bart_model
        self.pad_token_id = padding_index


    def forward(self, src_ids, attn_masks, labels = None):
        labels[labels == self.pad_token_id] = -100
        src_ids = src_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        attn_masks = attn_masks.to(DEVICE)
        out = dict()

        output = self.bart(input_ids = src_ids, attention_mask = attn_masks,
            labels = labels)
        logits = output['logits']
        out['loss'] = output['loss']
        out['logits'] = logits
        probs = F.softmax(logits, dim = -1)
        preds = torch.argmax(probs, dim = -1)
        out['preds'] = preds.cpu().detach().numpy()
        return out

    def resize_token_embeddings(self, tokenAligner: TokenAligner):
        vocab: Vocab = tokenAligner.vocab
        tokenizer: AutoTokenizer = tokenAligner.tokenizer
        char_vocab = []
        for i, key in enumerate(vocab.chartoken2idx.keys()):
            if i < 4:
                continue
            char_vocab.append(key)
            char_vocab.append(key + "@@")
        tokenizer.add_tokens(char_vocab)
        self.bart.resize_token_embeddings(len(tokenizer.get_vocab()))
        print("Resized token embeddings!")
        return
    
    def inference(self, src_ids, num_beams = 2, tokenAligner: TokenAligner = None):
        assert tokenAligner != None
        src_ids = src_ids.to(DEVICE)
        output = self.bart.generate(src_ids, num_beams=num_beams, max_new_tokens = 256)
        predict_text = tokenAligner.tokenizer.batch_decode(output, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces = False)
        return predict_text
        