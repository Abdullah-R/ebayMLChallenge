import torch

class DataSeq(torch.utils.data.Dataset):

    def __init__(self, text, labels, model_name = "bert-base-german-cased"):

        self.labels = labels
        self.texts = text

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        batch_data = {key: value[idx] for key, value in self.texts.items()}
        return batch_data

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_texts = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_texts, batch_labels