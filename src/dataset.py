from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, input_chunks, target_chunks):
        self.input_chunks = input_chunks
        self.target_chunks = target_chunks

    def __len__(self):
        return len(self.input_chunks)

    def __getitem__(self, index):
        return self.input_chunks[index], self.target_chunks[index]