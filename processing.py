from sarcasmData import SarcasmData
import torch.utils.data as data

class DatasetProcessing(data.Dataset):
    def __init__(self, originals, responses, labels, tag):
        self.dataset = SarcasmData(originals, responses, labels)
        self.tag = tag


    def __getitem__(self, index):
        if self.tag == "context":
            txt = self.dataset.originals_idxs[index]
        else:
            txt = self.dataset.responses_idxs[index]
        
        label = self.dataset.labels[index]
        return txt, label

    def __len__(self):
        if self.tag == "context":
            return len(self.dataset.originals_idxs)
        else:
            return len(self.dataset.responses_idxs)