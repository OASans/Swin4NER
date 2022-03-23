import torch
import torch.nn as nn
from pytorchcrf import CRF


# CRF
class Crf(nn.Module):
    def __init__(self, config):
        super(Crf, self).__init__()
        # self.params = {'crf': [], 'other': []}
        self.num_tags = config.num_tags

        self.crf = CRF(config.num_tags, batch_first=True)
        # self.params['crf'].extend([p for p in self.crf.parameters()])

        self.emission_linear = nn.Linear(config.hidden_size, config.num_tags)
        # self.params['other'].extend([p for p in self.emission_linear.parameters()])

    # def get_params(self):
    #     return self.params

    def decode(self, emission, mask):
        """
        emission: B T L F
        """
        emission_shape = emission.shape
        result = self.crf.decode(emission, mask)
        result = result.squeeze(dim=0)
        result = result.tolist()
        return result

    def cal_emission(self, text_vec):
        emission = self.emission_linear(text_vec)
        return emission

    def cal_loss(self, preds, y_true, mask):
        emission = preds['emission']
        _loss = -self.crf(emission, y_true, mask, reduction='token_mean')
        return _loss

    def forward(self, text_vec, mask, en_pred=True):
        emission = self.cal_emission(text_vec)
        if en_pred:
            pred = self.decode(emission, mask)
        else:
            pred = None
        return {'emission': emission,
                'pred': pred}