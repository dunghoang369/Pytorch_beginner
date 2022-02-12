from ImageTransform import ImageTransform
from config import *
from dataset import MyDataset
from utils import make_datapath_list, train_model, params_to_update, load_model


def main():
    train_path_list = make_datapath_list()
    val_path_list = make_datapath_list('val')

    # dataset
    train_dataset = MyDataset(train_path_list, transform=ImageTransform(resize, mean, std))
    val_dataset = MyDataset(val_path_list, transform=ImageTransform(resize, mean, std), phase='val')

    # dataloader
    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size)

    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    # network
    use_pretrained = True
    net = models.vgg16(use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    print(net)

    # loss
    criterior = nn.CrossEntropyLoss()

    # optimizer
    params1, params2, params3 = params_to_update(net)

    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3},
    ], momentum=0.9)

    # train model
    train_model(net, dataloader_dict, criterior, optimizer, epochs)


if __name__ == "__main__":
    # main()
    use_pretrained = True
    net = models.vgg16(use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    load_model(net, save_path)


