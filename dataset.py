from lib import *

class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == 'train':
            label = img_path[25:29]
        elif self.phase == 'val':
            label = img_path[23:27]

        if label == 'ants':
            label = 0
        if label == 'bees':
            label = 1

        return img_transformed, label