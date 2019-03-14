from sarcasmData import SarcasmData
import torch.utils.data as data

class ConcatDataset(data.Dataset):
    def __init__(self, originals, responses, labels):
        self.dataset = SarcasmData(originals, responses, labels)

    def __getitem__(self, index):
        txt_c = self.dataset.originals_idxs[index]
        txt_r = self.dataset.responses_idxs[index]
        label = self.dataset.labels[index]
        txt = (txt_c, txt_r)
        return txt, label

    def __len__(self):
        if len(self.dataset.originals_idxs) != len(self.dataset.responses_idxs):
            return 0
        return len(self.dataset.originals_idxs)