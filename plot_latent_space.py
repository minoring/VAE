import torch
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader

from model import FC, CNN
from parser_util import get_plot_parser
from data.utils import get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device on: ', device)

parser = get_plot_parser()
args = parser.parse_args()


def main():
    test_dataset = get_dataset(args.dataset, train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    img, _ = test_dataset[0]
    img_shape = img.shape
    if args.model == 'fc':
        input_size = torch.flatten(img).shape[0]
        model = FC(input_size=input_size, z_size=args.latent_dim).to(device)
    elif args.model == 'cnn':
        model = CNN(img_shape=img_shape, z_size=args.latent_dim).to(device)

    model.load_state_dict(torch.load(args.saved_path, map_location=device))
    model.eval()

    z_preds = []
    y_labels = []
    with torch.no_grad():
        for xs, ys in test_loader:
            xs = xs.to(device)
            _, mu, logvar = model(xs)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            zs = mu + eps * std
            z_preds.extend(zs.clone().cpu().numpy())
            y_labels.extend(ys.clone().cpu().numpy())
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, init='pca')
    tsne_results = tsne.fit_transform(z_preds)

    z1 = tsne_results[:, 0]
    z2 = tsne_results[:, 1]

    df = pd.DataFrame(list(zip(z1, z2, y_labels)), columns=['z1', 'z2', 'y'])
    sns_plot = sns.scatterplot(x='z1',
                               y='z2',
                               hue='y',
                               palette=sns.color_palette('hls', 10),
                               data=df,
                               legend='full')

    sns_plot.figure.savefig('latent.png')


if __name__ == '__main__':
    main()
