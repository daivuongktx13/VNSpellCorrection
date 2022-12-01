
from abc import abstractmethod
from models.tokenizer import TokenAligner
from dataset.noise import SynthesizeData

class PTCollator():

    def __init__(self, tokenAligner: TokenAligner):
        self.tokenAligner = tokenAligner
        self.noiser = SynthesizeData(tokenAligner.vocab)

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
        labels = []
        noised = []
        for sample in dataloader_batch:
            label = sample[0]
            noise = self.noiser.add_normal_noise(label)
            labels.append(label)
            noised.append(noise)

        batch_srcs, batch_tgts, batch_lengths = self.tokenAligner.tokenize_for_transformer_with_tokenization(noised, labels)
        data = dict()
        data['batch_src'] = batch_srcs
        data['batch_tgt'] = batch_tgts
        data['lengths'] = batch_lengths
        return data
        
    def collate_test(self, dataloader_batch):
        noised, labels = [], []
        for sample in dataloader_batch:
            noised.append(sample[0])
            labels.append(sample[1])
        batch_srcs, batch_tgts, batch_lengths = self.tokenAligner.tokenize_for_transformer_with_tokenization(noised, labels)
        data = dict()
        data['batch_src'] = batch_srcs
        data['batch_tgt'] = batch_tgts
        data['lengths'] = batch_lengths
        data['noised_texts'] = noised
        data['label_texts'] = labels
        return data

    def collate_correct(self, dataloader_batch):
        noised, labels = [], []
        for sample in dataloader_batch:
            noised.append(sample[0])
            labels.append(sample[1])

        batch_srcs= self.tokenAligner.tokenize_for_transformer_with_tokenization(noised)

        data = dict()
        data['batch_src'] = batch_srcs
        data['noised_texts'] = noised
        data['label_texts'] = labels
        return data    
        
        