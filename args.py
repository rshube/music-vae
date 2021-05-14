import argparse
import torch

class ArgsWrapper:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = None
        self.parser = argparse.ArgumentParser(
            description='Run training for Music AutoEncoder',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.parser.add_argument(
            '--model',
            type=str,
            choices=[
                'SimpleAE',
                'SimpleVAE',
                'ComplexAE',
                'SimpleConvAE',
                'ComplexConvAE',
                'ImageConvAE'
            ],
            default='SimpleAE',
            help='the auto encoder model to train'
        )
        self.parser.add_argument(
            '--train_epochs',
            type=int,
            default=10,
            help='the number of epochs to complete during training'
        )
        self.parser.add_argument(
            '--lr',
            type=float,
            default=1e-3,
            help='the learning rate'
        )
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            help='the batch size'
        )
        # self.parser.add_argument(
        #     '--verbose',
        #     action='store_true',
        #     help='print training loss'
        # )
        # self.parser.add_argument(
        #     '--print_every',
        #     type=int,
        #     default=10,
        #     help='how often to print training output during verbose training'
        # )

    def parse_args(self):
        self.args = self.parser.parse_args()
        for k,v in self.args.__dict__.items():
            setattr(self, k, v)

        return self

    def print_args(self):
        print("\n========== HYPERPARAMETERS ==========")
        for k,v in self.args.__dict__.items():
            print(f'{k}: {v}')
        print("\n")



if __name__ == '__main__':
    args = ArgsWrapper()
    args.parse_args()
    args.init_logger()
    args.print_args()
    