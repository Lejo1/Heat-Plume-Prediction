import torch
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


class TrajectoryLoss:
    """Streamline-position loss for the e2e LGCNN: traces every heat pump's centre streamline in
    CNN1's predicted velocity and in the simulated velocity and returns the mean equal-time distance
    [cells] - per line averaged over the two lines' common in-domain length, then over all lines.

    The temperature loss reaches a streamline only within ~4 sigma of where it is drawn; this one pulls
    a line back however far it has drifted, and its gradient runs through the whole RK4 trajectory
    (full adjoint, independent of detach_trajectory / v_blur, which only concern the drawing route).
    Both lines are traced in CNN2's output region - the region predictions and labels are cropped to -
    from the heat pumps the model's last forward found there (model.last_hp).
    Plain class, not an nn.Module, so the model is not registered as a submodule of the loss.
    """
    def __init__(self, model):
        self.model = model

    def _lines(self, v_phys, hp):
        from step2_streamlines.streamlines_helpers import trace_streamlines
        m = self.model
        return trace_streamlines(hp, v_phys[0], v_phys[1], tuple(v_phys.shape[1:]), randomK_data=m.randomK_data,
                                 t_steps=m.t_steps, use_compile=m.use_compile)

    def __call__(self, predictions, labels):
        m = self.model
        terms = []
        for b, hp in enumerate(m.last_hp):
            if hp.shape[0] == 0:
                continue
            v_pred = predictions[b, 1:3] * m.v_delta + m.v_min
            with torch.no_grad():
                sim = self._lines(labels[b, 1:3] * m.v_delta + m.v_min, hp)
            for (px, py, _), (sx, sy, _) in zip(self._lines(v_pred, hp), sim):
                k = min(len(px), len(sx))
                if k >= 2:
                    terms.append(((px[:k] - sx[:k]) ** 2 + (py[:k] - sy[:k]) ** 2 + 1e-6).sqrt().mean())
        if not terms:  # no heat pump in the output region: zero, but still part of the graph
            return predictions[:, 1:3].sum() * 0.0
        return torch.stack(terms).mean()


class E2ELoss(nn.Module):
    """
    Loss for the end-to-end LGCNN: L(T) + lambda_v * L_v(v), with L = MSE, MAE or Huber (`base`)
    and L_v the same choice for the velocity term (`base_v`; None = same as `base`).
    Predictions and labels carry 3 channels [T, vx, vy] (normalized units, comparable scales).
    lambda_v = 0 recovers the pure temperature loss.
    lambda_traj > 0 with a TrajectoryLoss `traj` adds lambda_traj * (mean streamline distance [cells]).
    Huber keeps PyTorch's delta=1.0: normalized errors are ~1e-2, so it stays in its quadratic
    branch, where it equals 0.5 * MSE.
    """
    BASES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}

    def __init__(self, lambda_v: float = 0.5, base: str = "mse", base_v: str = None,
                 lambda_traj: float = 0.0, traj: "TrajectoryLoss" = None):
        super(E2ELoss, self).__init__()
        base_v = base if base_v is None else base_v
        for label, b in (("base", base), ("base_v", base_v)):
            if b.lower() not in self.BASES:
                raise ValueError(f"E2ELoss {label} must be one of {list(self.BASES)}, got {b!r}")
        self.fn = self.BASES[base.lower()]()
        self.fn_v = self.BASES[base_v.lower()]()
        self.lambda_v = lambda_v
        if lambda_traj and traj is None:
            raise ValueError("E2ELoss: lambda_traj > 0 needs a TrajectoryLoss (traj=...)")
        self.lambda_traj, self.traj = lambda_traj, traj
        v_part = "" if base_v.lower() == base.lower() else f", v: {base_v.upper()}"
        traj_part = f", lambda_traj={lambda_traj}" if lambda_traj else ""
        self.name = rf"E2ELoss ({base.upper()}{v_part}, lambda_v={lambda_v}{traj_part})"

    def forward(self, predictions, labels):
        loss = self.fn(predictions[:, 0:1], labels[:, 0:1])
        if self.lambda_v != 0:
            loss = loss + self.lambda_v * self.fn_v(predictions[:, 1:3], labels[:, 1:3])
        if self.lambda_traj:
            loss = loss + self.lambda_traj * self.traj(predictions, labels)
        return loss
    

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