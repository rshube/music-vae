from utils import ArgsWrapper

def get_model(model):
    model = args.model
    if model == 'SimpleAE':
        return SimpleLinearAutoencoder()
    elif model == 'SimpleVAE':
        return SimpleLinearVariationalAutoencoder()
    elif model == 'ComplexAE':
        return ComplexLinearAutoencoder()
    elif model == 'SimpleConvAE':
        return SimpleConvolutionalEncoder()
    elif model == 'ComplexConvAE':
        return ComplexConvolutionalEncoder()
    elif model == 'ImageConvAE':
        return ImageConvolutionalEncoder()
    else:
        # argparser should catch this before, but just in case
        raise Exception(f'No model type matching {model}!')


def main(args):
    model = get_model(args.model).to(args.device)


if __name__ == '__main__':
    args = ArgsWrapper()
    args.parse_args()
    args.print_args()
    main(args)