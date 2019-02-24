from torch.utils.data.dataset import Dataset


class MNISTDataset(Dataset):

    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

        assert len(data) == len(label), 'Length of data and label should be same'
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        single_data = self.data[index]
        single_label = self.label[index]
        if self.transform is not None:
            single_data = self.transform(single_data)

        return single_data, single_label
