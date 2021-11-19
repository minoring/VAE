import os

from torchvision import datasets, transforms

from data.celeba import CelebA


def get_dataset(dataset, train):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # To use CNN, resize it to 32x32.
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        dataset = datasets.MNIST('data/mnist', download=True, train=train, transform=transform)
    elif dataset == 'celebA':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(148),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = CelebA(root_dir='data/celebA', train=train, transform=transform)
    return dataset


def inv_normalize(img, dataset):
    # Inverse normalization
    # x = z * sigma + mean
    #   = (z + mean / sigma) * sigma
    #   = (z -(-mean / sigma)) / (1 / sigma)
    if dataset == 'mnist':
        inv = transforms.Normalize((-0.1307 / 0.3081, ), (1 / 0.3081, ))
    elif dataset == 'celebA':
        inv = transforms.Normalize((-1, -1, -1), (2, 2, 2))

    return inv(img)
