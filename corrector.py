import os
from params import *
from dataset.vocab import Vocab
from dataset.util import load_dataset, load_vsec_dataset

if __name__ == "__main__":
    import argparse

    description = '''
        Corrector:

        Usage: python corrector.py --model tfmwtr --data_path ./data --dataset binhvq

        Params:
            --model
                    tfmwtr - Transformer with Tokenization Repair
            --data_path:    default to ./data
            --dataset:      default to 'binhvq'
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default='tfmwtr')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='binhvq')
    parser.add_argument('--test_dataset', type=str, default='binhvq')
    parser.add_argument("--beams", type=int, default=2)
    parser.add_argument("--fraction", type=float, default= 1.0)
    parser.add_argument("--metrics", type=str, default="teacher-forcing")
    parser.add_argument('--text', type=str, default='Bình mnh ơi day ch ưa, café xáng vớitôi dược không?')
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_path, f'{args.test_dataset}')

    weight_ext = 'pth'

    checkpoint_dir = os.path.join(args.data_path, f'checkpoints/{args.model}/{args.dataset}')

    weight_path = os.path.join(checkpoint_dir, f'{args.dataset}.weights.{weight_ext}')
    vocab_path = os.path.join(args.data_path, f'binhvq/binhvq.vocab.pkl')

    correct_file = f'{args.test_dataset}.test'
    incorrect_file = f'{args.test_dataset}.test.noise'
    length_file = f'{args.dataset}.length.test'

    if args.test_dataset != "vsec":
        test_data = load_dataset(base_path=dataset_path, corr_file=correct_file, incorr_file=incorrect_file,
                              length_file=length_file)
    else:
        test_data = load_vsec_dataset(base_path=dataset_path, corr_file=correct_file, incorr_file=incorrect_file)

    length_of_data = len(test_data)
    test_data = test_data[0 : int(args.fraction * length_of_data) ]

    vocab = Vocab()
    vocab.load_vocab_dict(vocab_path)

    from dataset.autocorrect_dataset import SpellCorrectDataset
    from models.corrector import Corrector
    from models.model import ModelWrapper
    from models.util import load_weights

    test_dataset = SpellCorrectDataset(dataset=test_data)

    model_wrapper = ModelWrapper(args.model, vocab)

    corrector = Corrector(model_wrapper)

    load_weights(corrector.model, weight_path)

    corrector.evaluate(test_dataset, metrics = args.metrics, beams = args.beams)