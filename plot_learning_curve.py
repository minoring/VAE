import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main(args):
    sns.set(style='darkgrid', font_scale=1.5)
    sns.set_palette("hls")
    sns.set(rc={'figure.figsize': (10, 7)})
    df = pd.read_csv(args.learning_curve_csv)
    g = sns.lineplot(x="Epochs", y=f"{args.loss}", hue="Split", data=df)
    legend = plt.legend(loc='upper left')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-curve-csv',
                        help='Path to learning curve csv file',
                        required=True)
    parser.add_argument('--loss',
                        help='Loss to plot',
                        choices=['Loss', 'Reconstruction_Loss', 'KL_Loss'])
    args = parser.parse_args()
    main(args)
