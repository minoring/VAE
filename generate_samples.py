import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import FC, CNN
from parser_util import get_plot_parser
from data.utils import get_dataset, inv_normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device on: ', device)

parser = get_plot_parser()
args = parser.parse_args()

NUM_SAMPLES = 64
NROW = 8


def main():
    test_dataset = get_dataset(args.dataset, train=False)
    img, _ = test_dataset[0]
    img_shape = img.shape
    if args.model == 'fc':
        input_size = torch.flatten(img).shape[0]
        model = FC(input_size=input_size, z_size=args.latent_dim).to(device)
    elif args.model == 'cnn':
        model = CNN(img_shape=img_shape, z_size=args.latent_dim).to(device)

    model.load_state_dict(torch.load(args.saved_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, args.latent_dim)
        z = z.to(device)

        preds = model.decode(z)

        preds = preds.detach().cpu()
        preds = preds.view(-1, *img_shape)
        preds = inv_normalize(preds, args.dataset)
        save_image(preds, args.dataset + f'_{args.model}' + '_random_sample.png', nrow=NROW)


if __name__ == '__main__':
    main()
