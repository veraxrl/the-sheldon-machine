import torch.utils.data as data
import torch


class SarcasmData(data.Dataset):

    def __init__(self, originals, responses, labels):
        super(SarcasmData, self).__init__()

        self.originals_idxs = torch.ByteTensor(originals)
        self.responses_idxs = torch.ByteTensor(responses)
        self.labels = torch.ByteTensor(labels)
