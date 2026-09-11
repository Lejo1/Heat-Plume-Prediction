import torch.nn as nn
from torch import max, abs, zeros, sum
from skimage.metrics import structural_similarity as ssim
    
    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1., second_loss:nn.Module = nn.L1Loss()):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha}) with {self.secondary_loss_function}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)

        return self.alpha * self.mse(predictions, labels) + (1. - self.alpha) * eval_second


class E2ELoss(nn.Module):
    """
    Loss for the end-to-end LGCNN: L(T) + lambda_v * L(v), with L = MSE, MAE or Huber (`base`).
    Predictions and labels carry 3 channels [T, vx, vy] (normalized units, comparable scales).
    lambda_v = 0 recovers the pure temperature loss.
    Huber keeps PyTorch's delta=1.0: normalized errors are ~1e-2, so it stays in its quadratic
    branch, where it equals 0.5 * MSE.
    """
    BASES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}

    def __init__(self, lambda_v: float = 0.5, base: str = "mse"):
        super(E2ELoss, self).__init__()
        if base.lower() not in self.BASES:
            raise ValueError(f"E2ELoss base must be one of {list(self.BASES)}, got {base!r}")
        self.fn = self.BASES[base.lower()]()
        self.lambda_v = lambda_v
        self.name = rf"E2ELoss ({base.upper()}, lambda_v={lambda_v})"

    def forward(self, predictions, labels):
        loss_T = self.fn(predictions[:, 0:1], labels[:, 0:1])
        if self.lambda_v == 0:
            return loss_T
        return loss_T + self.lambda_v * self.fn(predictions[:, 1:3], labels[:, 1:3])
    

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.min = 0
        self.max = 1

    def forward(self, predictions, labels):
        num_channels = predictions.shape[1]
        ssim_total = 0
        for dp in range(predictions.shape[0]):
            for channel in range(num_channels):
                ssim_val = ssim(predictions[dp, channel].detach().cpu().numpy(), labels[dp, channel].detach().cpu().numpy(), data_range=self.max - self.min)
                ssim_total += ssim_val
        return ssim_total / num_channels
    

class LinfLoss(nn.Module):
    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, output, target):
        return max(abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    """

    def __init__(self, pat_thresholds: list):
        super(PATLoss, self).__init__()
        self.pat_thresholds = pat_thresholds

    def forward(self, output, label):
        if len(output.shape) == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)
        pat = zeros((output.shape[0], len(self.pat_thresholds)), device=output.device)
        for idx in range(output.shape[1]):
            pat[:, idx] = sum(abs(output[:, idx] - label[:, idx]) > self.pat_thresholds[idx], dim=(1, 2)) / (output.shape[2] * output.shape[3])
        return pat * 100