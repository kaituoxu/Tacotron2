import torch.nn as nn


class FeaturePredictNetLoss(nn.Module):
    def __init__(self):
        super(FeaturePredictNetLoss, self).__init__()

    def forward(self, input, target):
        """
        Args:
            input: (feat_predict, feat_residual_predict, stop_tokens_predict, attention_weights)
            target: (feat_target, stop_tokens_target)
        Detail:
            stop_tokens_predict: stop_token value before sigmoid, do sigmoid in BCEWithLogitsLoss for numerical stabel
        Returns:
            loss: loss averaged over batch size and sequence length
        """
        feat_predict, feat_residual_predict, stop_tokens_predict, _ = input
        feat_target, stop_tokens_target = target
        stop_tokens_predict = stop_tokens_predict.view(-1, 1)
        stop_tokens_target = stop_tokens_target.view(-1, 1)

        feat_loss = nn.MSELoss()(feat_predict, feat_target) + \
                    nn.MSELoss()(feat_residual_predict, feat_target)
        stop_loss = nn.BCEWithLogitsLoss()(stop_tokens_predict, stop_tokens_target)
        loss = feat_loss + stop_loss
        return loss
