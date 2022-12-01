import os
from dataset.util import load_dataset, train_validation_split
from dataset.vocab import Vocab

if __name__ == '__main__':
    import argparse

    description = '''
        train.py:

        Usage: python train.py --model tfmwtr --start-epoch n --data_path ./data --dataset binhvq

        Params:
            --start-epoch   n
                    n = 0: training from beginning
                    n > 0: continue training from the nth epoch
            --model
                    tfmwtr - Transformer with Tokenization Repair
            --data_path:    default to ./data
            --dataset:      default to 'binhvq'
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default='tfmwtr')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='binhvq')
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_path, f'{args.dataset}')
    vocab_path = os.path.join(dataset_path, f'{args.dataset}.vocab.pkl')

    vocab = Vocab()
    vocab.load_vocab_dict(vocab_path)

    checkpoint_dir = os.path.join(args.data_path, f'checkpoints/{args.model}')
    incorrect_file = f'{args.dataset}.train.noise'
    correct_file = f'{args.dataset}.train'
    length_file = f'{args.dataset}.length.train'

    data = load_dataset(base_path=dataset_path, corr_file=correct_file, incorr_file=incorrect_file,
                        length_file = length_file)

    train_data, valid_data = train_validation_split(data, 0.99, seed=11690)

    from dataset.autocorrect_dataset import SpellCorrectDataset
    from models.trainer import Trainer
    from models.model import ModelWrapper

    train_dataset = SpellCorrectDataset(dataset=train_data)
    valid_dataset = SpellCorrectDataset(dataset=valid_data)

    model_wrapper = ModelWrapper(args.model, vocab)

    trainer = Trainer(model_wrapper, train_dataset, valid_dataset)

    trainer.load_checkpoint(checkpoint_dir, args.dataset, args.start_epoch)

    trainer.train()
