from lib import *
from config import *

def make_datapath_list(phase='train'):
    target_path = os.path.join('./hymenoptera_data/' + phase + '/**/*.jpg')
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list


def train_model(net, dataloader_dict, criterior, optimizer, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # move network to device(cpu/gpu)
        net.to(device)
        torch.backends.cudnn.benchmark = True

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move inputs, labels to GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradient of optimizer to be zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, predicts = torch.max(outputs, axis=1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(predicts == labels)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print(f"Training set: Average loss: {epoch_loss:.2f} Accuracy: {epoch_accuracy:.2f}")

    torch.save(net.state_dict(), save_path)


def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_params_name_1 = ['features']
    update_params_name_2 = ['classifier.0.weight', 'classififier.0.bias', 'classifier.3.weight', 'classififier.3.bias']
    update_params_name_3 = ['classifier.6.weight', 'classififier.6.bias']

    for name, param in net.named_parameters():
        if name in update_params_name_1:
            param.requires_grad = True # thay đổi weight
            params_to_update_1.append(param)
        elif name in update_params_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_params_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        else:
            param.requires_grad = False # không update weight nữa

    return params_to_update_1, params_to_update_2, params_to_update_3


def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights) # load weight vào khung layers, network chỉ là khung thui chứ chưa có giá trị
    # print(net)
    #
    # for name, param in net.named_parameters():
    #     print(name, param)
    # load_weights = torch.load(model_path, map_location=("cuda:0", "cpu"))
    # net.load_state_dict(load_weights)
    return net