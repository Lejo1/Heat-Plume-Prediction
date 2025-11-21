import torch.nn as nn
from torch import manual_seed, Tensor, log, max, abs, zeros, sum
from skimage.metrics import structural_similarity as ssim

class KLD_log():
    def __init__(self):
        self.kld = nn.KLDivLoss()

    def __call__(self, prediction: Tensor, label: Tensor):
        return self.kld(log(prediction), label)
    
class CombiLoss(nn.Module):
# from Delft

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
    
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.min = 0
        self.max = 1

    def forward(self, predictions, labels):
        # assert predictions.max() <= self.max, f"Prediction values exceed max value with {predictions.max()}"
        # assert predictions.min() >= self.min, f"Prediction values are below min value with {predictions.min()}"
        # assert labels.max() <= self.max, f"Label values exceed max value with {labels.max()}"
        # assert labels.min() >= self.min, f"Label values are below min value with {labels.min()}"
        
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

class IoULoss(nn.Module):
    # best value is 1.0, worst is 0.0
    def __init__(self):
        super(IoULoss, self).__init__()
        # on binary mask of 0.9 threshold (values between 0 and 1)
        # TODO currently normalized data -> go to real data?
        self.threshold = 0.5 # TODO find reasonable threshold
        self.epsilon = 1e-6 # to avoid division by zero

    def forward(self, output, label):
        output = output > self.threshold
        label = label > self.threshold
        if len(output.shape) == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)
        intersection = (output & label).float().sum((2, 3))
        union = (output | label).float().sum((2, 3))
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return iou.mean() # averaged over batch and channels
    
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
    

# class ConservationOfMass(nn.Module):
#     ## CM (conservation of energy)-ish
# # TODO does this make sense for velocities???
# normalized_additional_energy1 =( np.sum(output[0]) - np.sum(label[0]))/np.sum(label[0])
# normalized_additional_energy2 = (np.sum(output[1]) - np.sum(label[1]))/np.sum(label[1])
# print("conservation vx", normalized_additional_energy1, "vy", normalized_additional_energy2, "best is 0, worst is inf")