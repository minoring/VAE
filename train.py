import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import FC, CNN
from parser_util import get_train_parser
from data.utils import get_dataset, inv_normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device on: ', device)

parser = get_train_parser()
args = parser.parse_args()


def criterion(xs, x_preds, mu, logvar):
    recon_loss = F.smooth_l1_loss(x_preds, xs, reduction='sum')

    # See Appendix B from VAE paper:
    # Solution of KL divergence, Gaussian case.
    # https://arxiv.org/abs/1312.6114
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kl_loss


def train(epoch, model, img_shape, optimizer, train_loader):
    model.train()

    for batch_idx, (xs, _) in enumerate(train_loader):
        xs = xs.to(device)

        x_preds, mu, logvar = model(xs)
        x_preds = x_preds.view(-1, *img_shape)
        recon_loss, kl_loss = criterion(xs, x_preds, mu, logvar)
        loss = recon_loss + kl_loss

        if epoch == 1 and batch_idx == 0:
            print(
                'Initial loss: {:.6f} Initial reconstruction loss: {:.6f} Initial kl loss: {:.6f}'.
                format(loss.item(), recon_loss.item(), kl_loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # To make loss invariant to batch size, compute mean.
        mean_loss = loss.item() / len(xs)
        mean_recon_loss = recon_loss.item() / len(xs)
        mean_kl_loss = kl_loss.item() / len(xs)

        if batch_idx == 0:
            running_loss = mean_loss
            running_recon_loss = mean_recon_loss
            running_kl_loss = mean_kl_loss
        else:
            running_loss = 0.05 * mean_loss + (1 - 0.05) * running_loss
            running_recon_loss = 0.05 * mean_recon_loss + (1 - 0.05) * running_recon_loss
            running_kl_loss = 0.05 * mean_kl_loss + (1 - 0.05) * running_kl_loss

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {}[{}/{} ({:.0f})%]\tLoss: {:.8f}\tReconstruction Loss: {:.8f}\tKL Divergence Loss: {:.8f}'
                .format(epoch, batch_idx * len(xs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), running_loss, running_recon_loss,
                        running_kl_loss))

    return running_loss, running_recon_loss, running_kl_loss


def test(epoch, model, img_shape, test_loader):
    model.eval()

    sum_recon_loss = 0
    sum_kl_loss = 0
    sum_loss = 0

    with torch.no_grad():
        for batch_idx, (xs, _) in enumerate(test_loader):
            xs = xs.to(device)
            x_preds, mu, logvar = model(xs)
            x_preds = x_preds.view(-1, *img_shape)
            recon_loss, kl_loss = criterion(xs, x_preds, mu, logvar)
            loss = recon_loss + kl_loss

            sum_recon_loss += recon_loss.item()
            sum_kl_loss += kl_loss.item()
            sum_loss += loss.item()

            if batch_idx == 0:
                nrow = 8
                xs = xs.detach().cpu()
                x_preds = x_preds.detach().cpu()
                comparision = torch.cat([xs[:nrow], x_preds[:nrow]])
                comparision = inv_normalize(comparision, args.dataset)
                save_image(comparision,
                           'results/' + args.dataset + f'/{args.model}' + '/reconstruction_' +
                           'epoch_' + str(epoch) + '.png',
                           nrow=nrow)

    avg_loss = sum_loss / len(test_loader.dataset)
    avg_recon_loss = sum_recon_loss / len(test_loader.dataset)
    avg_kl_loss = sum_kl_loss / len(test_loader.dataset)
    print("======> Test ")
    print("Epoch: {} Test loss: {:.8f}\tTest reconstruction loss: {:.8f}\tTest kl loss: {:.8f}".
          format(epoch, avg_loss, avg_recon_loss, avg_kl_loss))
    return avg_loss, avg_recon_loss, avg_kl_loss


def main():
    train_dataset = get_dataset(args.dataset, train=True)
    test_dataset = get_dataset(args.dataset, train=False)

    cuda_kwargs = {}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              **cuda_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **cuda_kwargs)

    img, _ = train_dataset[0]
    img_shape = img.shape
    if args.model == 'fc':
        input_size = torch.flatten(img).shape[0]
        model = FC(input_size=input_size, z_size=args.latent_dim).to(device)
    elif args.model == 'cnn':
        model = CNN(img_shape=img_shape, z_size=args.latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.scheduler_step,
                                          gamma=args.scheduler_gamma)

    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    test_losses = []
    test_recon_losses = []
    test_kl_losses = []

    print("======> Start training")
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_recon_loss, train_kl_loss = train(epoch, model, img_shape, optimizer,
                                                            train_loader)
        test_loss, test_recon_loss, test_kl_loss = test(epoch, model, img_shape, test_loader)

        train_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        test_losses.append(test_loss)
        test_recon_losses.append(test_recon_loss)
        test_kl_losses.append(test_kl_loss)
        scheduler.step()

    if args.save_model:
        model_save_path = args.model + '_' + args.dataset + ".pt" if args.save_path is None else args.save_path

        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    train_split = ['train'] * args.num_epochs
    test_split = ['test'] * args.num_epochs
    epochs = list(range(1, args.num_epochs + 1))
    columns = ['Split', 'Epochs', 'Loss', 'Reconstruction_Loss', 'KL_Loss']
    train_df = pd.DataFrame(list(
        zip(train_split, epochs, train_losses, train_recon_losses, train_kl_losses)),
                            columns=columns)
    test_df = pd.DataFrame(list(
        zip(test_split, epochs, test_losses, test_recon_losses, test_kl_losses)),
                           columns=columns)

    learning_curve_csv = args.learning_curve_csv if args.learning_curve_csv is not None else f'{args.model}_' + args.dataset + '.csv'
    learning_curve_df = pd.concat([train_df, test_df], ignore_index=True)
    learning_curve_df.to_csv(learning_curve_csv, mode='w')
    print(f"Learning curve saved at: {learning_curve_csv}")


if __name__ == '__main__':
    main()
