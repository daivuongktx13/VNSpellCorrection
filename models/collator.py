
from abc import abstractmethod
from models.tokenizer import TokenAligner

class PTCollator():

    def __init__(self, tokenAligner: TokenAligner):
        self.tokenAligner = tokenAligner

    def collate(self, dataloader_batch, type = "train") -> dict:
        if type == "train":
            return self.collate_train(dataloader_batch)
        elif type == "test":
            return self.collate_test(dataloader_batch)
        elif type == "correct":
            return self.collate_correct(dataloader_batch)
        
    @abstractmethod
    def collate_train(self, dataloader_batch):
        
        pass

    @abstractmethod
    def collate_test(self, dataloader_batch):
        pass

    
    @abstractmethod
    def collate_correct(self, dataloader_batch):
        pass

class DataCollatorForCharacterTransformer(PTCollator):

    def __init__(self, tokenAligner: TokenAligner):
        super().__init__(tokenAligner)

    def collate_train(self, dataloader_batch):
        noised, labels = [], []
        for sample in dataloader_batch:
            noised.append(sample[0])
            labels.append(sample[1])

        batch_srcs, batch_tgts, batch_lengths, batch_attention_masks = self.tokenAligner.tokenize_for_transformer_with_tokenization(noised, labels)
        data = dict()
        data['batch_src'] = batch_srcs
        data['batch_tgt'] = batch_tgts
        data['attn_masks'] = batch_attention_masks
        data['lengths'] = batch_lengths
        return data
        
    def collate_test(self, dataloader_batch):
        noised, labels = [], []
        for sample in dataloader_batch:
            noised.append(sample[0])
            labels.append(sample[1])

        batch_srcs, batch_attention_masks = self.tokenAligner.tokenize_for_transformer_with_tokenization(noised, None)
        data = dict()
        data['batch_src'] = batch_srcs
        data['noised_texts'] = noised
        data['label_texts'] = labels
        data['attn_masks'] = batch_attention_masks
        return data

    def collate_correct(self, dataloader_batch):
        noised, labels = [], []
        for sample in dataloader_batch:
            noised.append(sample[0])
            labels.append(sample[1])

        batch_srcs, batch_attention_masks= self.tokenAligner.tokenize_for_transformer_with_tokenization(noised)

        data = dict()
        data['batch_src'] = batch_srcs
        data['noised_texts'] = noised
        data['label_texts'] = labels
        data['attn_masks'] = batch_attention_masks
        return data    
        
        