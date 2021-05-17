import os
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from utils import ArgsWrapper, DATA_PATH
from training import (
    OptimizeFourierModel,
    OptimizeModel,
    TestAllFourierModel,
    TestAllModel
)
from models import (
    SimpleAutoEncoder,
    SimpleVarAutoEncoder,
    ComplexAutoEncoder,
    SimpleConvAutoEncoder,
    ComplexConvAutoEncoder,
    ImageConvAutoEncoder
)


def get_model(model):
    model = args.model
    fourier = False
    if model == 'SimpleAutoEncoder':
        return SimpleAutoEncoder(), fourier
    elif model == 'SimpleVarAutoEncoder':
        return SimpleVarAutoEncoder(), fourier
    elif model == 'ComplexAutoEncoder':
        return ComplexAutoEncoder(), fourier
    elif model == 'SimpleConvAutoEncoder':
        return SimpleConvAutoEncoder(), fourier
    elif model == 'ComplexConvAutoEncoder':
        return ComplexConvAutoEncoder(), fourier
    elif model == 'ImageConvAutoEncoder':
        fourier = True
        return ImageConvAutoEncoder(), fourier
    else:
        # argparser should catch this before, but just in case
        raise Exception(f'No model type matching {model}!')


def make_training_plot(losses, model_name, save_dir):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1),losses)
    ax.set_title(f'Loss as a function of epoch for {model_name}')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('MSE Loss')
    plt.savefig(os.path.join(save_dir, 'training_plot.svg'))
    plt.show()


def make_evaluation_plot(labels, results, model_name, save_dir):
    fig, ax = plt.subplots()
    ax.bar(range(len(results)),results)
    ax.set_title(f'Loss across data sets for {model_name}')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Data Set')
    ax.set_ylabel('MSE Loss')
    plt.savefig(os.path.join(save_dir, 'evaluation_plot.svg'))
    plt.show()

def main(args):
    model, fourier = get_model(args.model)
    model = model.to(args.device)

    # Train the models
    if not fourier:
        losses = OptimizeModel(args, model, 'lofi-track-1-clip-', 720)
    else:
        losses = OptimizeFourierModel(args, model, 'lofi-track-1-clip-', 5)

    # test the models
    if not fourier:
        labels, results = TestAllModel(model)
    else:
        labels, results = TestAllFourierModel(model)

    # save plots and model
    save_dir = os.path.join(os.getcwd(), 'results', args.model, datetime.now().strftime("%m_%d-%H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)
    make_training_plot(losses, args.model, save_dir)
    make_evaluation_plot(labels, results, args.model, save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))


if __name__ == '__main__':
    print(DATA_PATH)
    args = ArgsWrapper()
    args.parse_args()
    args.print_args()
    main(args)

    print("Done")