from typing import Dict

import torch


class PaddedBatch:
    """Contains a padded batch of sequences with different lengths.

    Parameters:
        payload:
            container with data. Format supported:
            - dict with features. This is the input data for overall network pipeline.
                Kees are the feature names, values are (B, T) shape tensors.
                Long type for categorical features, embedding lookup table indexes expected
                Float type for numerical features.
            - trx embedding tensor. This is the intermediate data for overall network pipeline.
                shape (B, T, H)
            - feature tensor. Used in some cases
                shape (B, T)
        length:
            Tensor of shape (B,) with lengths of sequences.
            All sequences in `payload` has length T, but only L first are used.
            Unused positions padded with zeros

    Example:
        >>> data = PaddedBatch(
        >>>     payload=torch.tensor([
        >>>         [1, 2, 0, 0],
        >>>         [3, 4, 5, 6],
        >>>         [7, 8, 9, 0],
        >>>     ]),
        >>>     length=torch.tensor([2, 4, 3]),
        >>> )
        >>>
        >>> # check shape
        >>> torch.testing.assert_close(data.payload.size(), (3, 4))
        >>>
        >>> # get first transaction
        >>> torch.testing.assert_close(data.payload[:, 0], torch.tensor([1, 3, 7]))
        >>>
        >>> # get last transaction
        >>> torch.testing.assert_close(data.payload[torch.arange(3), data.seq_lens - 1], torch.tensor([2, 6, 9]))
        >>>
        >>> # get all transaction flatten
        >>> torch.testing.assert_close(data.payload[data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    """
    def __init__(self, payload: Dict[str, torch.Tensor], length: torch.LongTensor):
        self._payload = payload
        self._length = length

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length

    @property
    def device(self):
        return self._length.device

    def __len__(self):
        return len(self._length)

    def to(self, device, non_blocking=False):
        length = self._length.to(device=device, non_blocking=non_blocking)
        payload = {
            k: v.to(device=device, non_blocking=non_blocking) for k, v in self._payload.items()
        }
        return PaddedBatch(payload, length)

    @property
    def seq_len_mask(self):
        """mask with B*T size for valid tokens in `payload`
        """
        if type(self._payload) is dict:
            B, T = next(iter(self._payload.values())).size()
        else:
            B, T = self._payload.size()[:2]
        return (torch.arange(T, device=self._length.device).unsqueeze(0).expand(B, T) < \
                self._length.unsqueeze(1)).long()