import torch
from lifestream.ptls.frames.bert import MlmDataset


class GptDataset(MlmDataset):
    pass


class GptIterableDataset(GptDataset, torch.utils.data.IterableDataset):
    pass
