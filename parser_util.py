import argparse


def get_train_parser():
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--model',
                        '-m',
                        help='Model to train VAE',
                        choices=['fc', 'cnn'],
                        required=True)
    parser.add_argument('--dataset',
                        '-d',
                        help='Choose dataset',
                        choices=['mnist', 'celebA'],
                        required=True)
    parser.add_argument('--batch-size', '-b', help='Batch size', type=int, default=512)
    parser.add_argument('--num-epochs', '-n', help='Number of epochs', type=int, default=10)
    parser.add_argument('--latent-dim',
                        '-l',
                        help='Dimension of latent space (default: 128)',
                        type=int,
                        default=128)
    parser.add_argument('--lr', help='Learning rate (default: 1e-4)', type=float, default=1e-4)
    parser.add_argument('--log-interval',
                        type=int,
                        default=50,
                        help='How many batches to wait before logging')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=True,
                        help='Save trained model')
    parser.add_argument('--save-path', help='Path to save trained model')
    parser.add_argument('--learning-curve-csv', help='csv file path to save learning curve')
    parser.add_argument('--scheduler-step',
                        help='Decay learning rate every this step',
                        type=int,
                        default=100)
    parser.add_argument('--scheduler-gamma',
                        help='Learning rate decay (default: 0.5)',
                        type=float,
                        default=0.5)
    return parser


def get_plot_parser():
    parser = argparse.ArgumentParser(description="Visualize Trained VAE")
    parser.add_argument('--model', '-m', help='Trained model', choices=['fc', 'cnn'], required=True)
    parser.add_argument('--dataset',
                        '-d',
                        help='Choose trained dataset',
                        choices=['mnist', 'celebA'],
                        required=True)
    parser.add_argument('--saved-path', help='Path to trained model', required=True)
    parser.add_argument('--latent-dim',
                        '-l',
                        help='Dimension of trained VAE latent space',
                        type=int,
                        required=True)

    return parser
