import torch
import torch.nn as nn

class SSIM(nn.Module):
    """Structural Similarity Index Metric

    https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.1939
    """

    def __init__(self):
        super().__init__()

        self.pad = nn.ReflectionPad2d(1)

        self.x_mean = nn.AvgPool2d(3, 1)
        self.y_mean = nn.AvgPool2d(3, 1)

        self.x_square_mean = nn.AvgPool2d(3, 1)
        self.y_square_mean = nn.AvgPool2d(3, 1)
        self.xy_mean = nn.AvgPool2d(3, 1)

        # k1 = self.cfg.LOSSES.SSIM.K1
        # k2 = self.cfg.LOSSES.SSIM.K2
        # l = self.cfg.LOSSES.SSIM.L

        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def __call__(self, x, y):
        # symmetrically pad array for border artifact reduction
        x, y = self.pad(x), self.pad(y)

        mean_x, mean_y = self.x_mean(x), self.y_mean(y)

        var_x = self.x_square_mean(x**2) - mean_x**2
        var_y = self.y_square_mean(y**2) - mean_y**2
        cov_xy = self.xy_mean(x * y) - mean_x * mean_y

        ssim = 2 * mean_x * mean_y + self.c1
        ssim *= 2 * cov_xy + self.c2
        ssim /= mean_x**2 + mean_y**2 + self.c1
        ssim /= var_x + var_y + self.c2

        # enforce boundedness to expel floating point gremlins
        return torch.clamp(ssim, min=0.0, max=1.0)


class ReprojectionLoss(nn.Module):
    """Photometric reconstruction error. The Monodepth2 paper introduces two
    extra terms on top of the raw reconstruction error to improve performance:
    - pixelwise-minimum: take minimum loss across both reconstructed images to
                         remove contributions from pixels that are not visible
                         in one of the two scenes
    - stationary pixel mask: ignore contributions from pixels that do not change
                             ~too~ much across frames
    """

    def __init__(self):
        super().__init__()
        self.l1 = lambda x, y: torch.abs(x - y)
        self.ssim = SSIM()
        self.alpha = 0.85

    def photometric(self, gt, pred):
        l1 = self.l1(gt, pred).mean(1, True)
        ssim = (1 - self.ssim(gt, pred).mean(1, True)) / 2
        return (1 - self.alpha) * l1 + self.alpha * ssim

    def __call__(self, inp, out):
        total_losses = []
        # iterate over scales
        for scale in range(4):
            reconstruction_losses = []
            motion_losses = []
            # iterate over input images
            for frame_id in [-1, 1]:
                gt = inp[("color", 0, 0)]
                pred = out[("color", frame_id, scale)]
                # loss due to reconstruction error
                reconstruction_loss = self.photometric(gt, pred)
                reconstruction_losses.append(reconstruction_loss)
                # loss due to pixel motion alone
                motion_loss = self.photometric(gt, inp[("color", frame_id, 0)])
                # for breaking ties
                motion_loss += 1e-5 * torch.randn(motion_loss.shape, device=motion_loss.device)
                motion_losses.append(motion_loss)

            # pixelwise-minimum to prevent occluded pixels from contributing to loss
            # mask out stationary pixels
            reconstruction_loss = torch.cat(reconstruction_losses, dim=1)
            motion_loss = torch.cat(motion_losses, dim=1)
            scale_loss, idxs = torch.min(torch.cat([reconstruction_loss, motion_loss], dim=1), dim=1)
            out[f"identity_selection/{scale}"] = (idxs > motion_loss.shape[1] - 1).float()
            total_losses.append(scale_loss.mean())

        return sum(total_losses).mean()


class SmoothnessLoss(nn.Module):
    """Edge-aware smoothness by Godard et al. Penalizes sharp
    transitions in depth while allowing for edges

    https://arxiv.org/pdf/1609.03677.pdf
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-7
        self.smoothness = 1e-3

    def partials(self, t):
        del_x_t = torch.abs(t[..., :-1] - t[..., 1:])
        del_y_t = torch.abs(t[..., :-1, :] - t[..., 1:, :])
        return del_x_t, del_y_t

    def __call__(self, inp, out):
        losses = []
        # color = inp[1]["image"]
        # color = inp[]
        for scale in range(4):
            color = inp[("color", 0, scale)]
            # disparity = out["reprojected"][("upsampled_disparity", scale)]
            disparity = out[("disp", scale)]
            # mean-normalize disparity to avoid penalizing large depth predictions
            disparity_norm = disparity / (disparity.mean(2, True).mean(3, True) + self.eps)

            grads_disp = self.partials(disparity_norm)
            grads_color = [torch.mean(t, 1, keepdim=True) for t in self.partials(color)]
            loss = sum(
                [
                    (g_d * torch.exp(-g_c)).mean()
                    for g_d, g_c in zip(grads_disp, grads_color)
                ]
            )
            loss *= self.smoothness / (2 ** scale)
            losses.append(loss)
        return sum(losses).mean()

class JackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reprojection = ReprojectionLoss()
        self.smoothness = SmoothnessLoss()

    def forward(self, inp, out):
        return self.reprojection(inp, out) + 1e-3 * self.smoothness(inp, out)