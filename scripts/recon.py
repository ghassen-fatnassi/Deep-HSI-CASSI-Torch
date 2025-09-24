import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict

# replace these imports with your actual ConvAE implementation
from ConvAE import Encoder, Decoder, ConvAutoencoder

class Config:
    N_VALID_SPECTRALS = 31
    WEIGHT_DECAY_LAMBDA = 1e-8

class Modulation:
    SHIFT_VALUES = [24, 19, 14, 9, 4, 0, -4, -7, -11, -14, -17,
                    -20, -23, -25, -28, -30, -32, -34, -36, -38, -40, -42,
                    -43, -45, -46, -48, -49, -50, -51, -53, -54, -56, -57]

    @staticmethod
    def generate_random_mask(h: int, w: int, scale: float = 1.0) -> np.ndarray:
        mask = np.random.randint(0, 2, size=(int(h/scale), int(w/scale))).astype(np.float32)
        if scale != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def generate_shifted_mask_cube(mask2d: np.ndarray, chs: int = 31,
                                   shift_list: Optional[List[int]] = None) -> np.ndarray:
        if shift_list is None:
            shift_list = Modulation.SHIFT_VALUES[:chs]
        h, w = mask2d.shape
        cube = np.zeros((h, w, chs), dtype=mask2d.dtype)
        for ch in range(chs):
            shift_val = float(shift_list[ch % len(shift_list)])
            M = np.float32([[1, 0, shift_val], [0, 1, 0]])
            warped = cv2.warpAffine(mask2d, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            cube[:, :, ch] = warped
        return cube

    @staticmethod
    def generate_coded_image_from_cube(hs_cube: np.ndarray, mask_cube: np.ndarray) -> np.ndarray:
        masked = hs_cube * mask_cube
        proj = np.sum(masked, axis=2)
        proj_norm = proj / float(mask_cube.shape[2])
        return proj_norm

class ADMMHelpers:
    @staticmethod
    def gradient_operator(x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> returns (B, C, H, W, 2) [dv, dh] same spatial dims
        B, C, H, W = x.shape
        dv = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)
        dh = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)
        dv[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        dh[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        stacked = torch.stack([dv, dh], dim=-1)
        return stacked

    @staticmethod
    def soft_threshold(v: torch.Tensor, lambda_val: float, rho: float) -> torch.Tensor:
        thresh = lambda_val / rho
        return torch.sign(v) * torch.clamp(torch.abs(v) - thresh, min=0.0)

class HyperspectralReconstruction(nn.Module):
    def __init__(self, autoencoder: nn.Module, img_h: int, img_w: int, img_chs: int = 31, n_features_code: int = 64):
        super().__init__()
        self.autoencoder = autoencoder
        self.img_h = img_h
        self.img_w = img_w
        self.img_chs = img_chs
        self.n_features_code = n_features_code

        # make xk trainable
        xk_init = torch.zeros(1, n_features_code, img_h, img_w)
        self.xk = nn.Parameter(xk_init)

        zk_shape = (1, img_chs, img_h, img_w, 2)
        self.register_buffer('zk', torch.zeros(zk_shape))
        self.register_buffer('uk', torch.zeros(zk_shape))

    def forward_projection(self, img_recon: torch.Tensor, mask3d: torch.Tensor) -> torch.Tensor:
        # img_recon: (B,C,H,W), mask3d: (B,C,H,W)
        img_masked = img_recon * mask3d
        img_prj = torch.sum(img_masked, dim=1, keepdim=True) / float(self.img_chs)
        return img_prj

    def compute_losses(self, img_snapshot_coded: torch.Tensor, mask3d: torch.Tensor,
                       rho: float, lambda_alpha: float) -> Dict:
        img_recon = self.autoencoder.decoder(self.xk)  # expect decoder accepts (B, n_features, H, W) or (n_features...) adapt if needed
        if img_recon.dim() == 3:
            img_recon = img_recon.unsqueeze(0)
        img_prj = self.forward_projection(img_recon, mask3d)

        loss_data = 0.5 * torch.mean((img_prj - img_snapshot_coded) ** 2)

        G_xk = ADMMHelpers.gradient_operator(img_recon)
        diff_admm = G_xk - self.zk + self.uk
        loss_admm = 0.5 * rho * torch.mean(diff_admm ** 2)

        loss_alpha_fidelity = torch.tensor(0., device=img_recon.device)

        alpha_from_encoder = self.autoencoder.encoder(img_recon)
        if alpha_from_encoder.dim() == 3:
            alpha_from_encoder = alpha_from_encoder.unsqueeze(0)
        diff_alpha = self.xk - alpha_from_encoder
        loss_alpha_fidelity = lambda_alpha * 0.5 * torch.mean(diff_alpha ** 2)

        total_loss = loss_data + loss_admm + loss_alpha_fidelity

        return {
            'total': total_loss,
            'data': loss_data,
            'admm': loss_admm,
            'alpha_fidelity': loss_alpha_fidelity,
            'img_recon': img_recon,
            'img_prj': img_prj,
            'G_xk': G_xk
        }

    def admm_update(self, img_recon: torch.Tensor, lambda_val: float, rho: float):
        G_xk = ADMMHelpers.gradient_operator(img_recon)
        with torch.no_grad():
            self.zk.copy_(ADMMHelpers.soft_threshold(G_xk + self.uk, lambda_val, rho))
            self.uk.copy_(self.uk + G_xk - self.zk)

    def reset_admm_variables(self, fill=1e-4):
        with torch.no_grad():
            self.xk.data.fill_(0.0)
            self.zk.fill_(fill)
            self.uk.fill_(fill)

def load_autoencoder_from_checkpoint(path: str, device: str, in_channels: int = 31, R: int = 64) -> ConvAutoencoder:
    checkpoint = torch.load(path, map_location=device)
    # build encoder/decoder using assumed signatures; adjust params if your ConvAE differs
    encoder = Encoder(in_channels=in_channels, R=R, d=11, use_batchnorm=True)
    decoder = Decoder(out_channels=in_channels, R=R, d=11, use_batchnorm=False, output_activation='sigmoid')
    model = ConvAutoencoder(encoder, decoder).to(device)
    # try:
    #     # checkpoint might be state_dict or full model
    #     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['state_dict'])
    #     elif isinstance(checkpoint, dict):
    #         model.load_state_dict(checkpoint)
    #     else:
    #         model = checkpoint.to(device)
    # except Exception as e:
    #     print("Warning: couldn't load checkpoint directly into model:", e)
    #     # try looser loading
    #     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items() if 'module.' in k or True}, strict=False)
    model.eval()
    return model

def reconstruct_snapshot_tensors(coded_snapshot: torch.Tensor,
                                 mask2d: np.ndarray,
                                 autoencoder_path: str,
                                 gt_hs: Optional[torch.Tensor] = None,
                                 img_n_chs: int = 31,
                                 param_rho: float = 7.5e-2,
                                 param_sparsity: float = 1e-2,
                                 param_lambda_alpha_fidelity: float = 1e-1,
                                 param_learning_rate: float = 5e-2,
                                 n_iters_ADMM: int = 5,
                                 n_iters_ADAM: int = 200,
                                 ENABLE_ALPHA_FIDELITY: bool = True,
                                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    # coded_snapshot: torch tensor shape (H,W) or (1,H,W) or (B,1,H,W)
    # gt_hs: optional torch tensor shape (C,H,W)
    if coded_snapshot.dim() == 2:
        coded_snapshot = coded_snapshot.unsqueeze(0).unsqueeze(0)
    elif coded_snapshot.dim() == 3 and coded_snapshot.shape[0] != 1:
        coded_snapshot = coded_snapshot.unsqueeze(0)  # (1,1,H,W)
    coded_snapshot = coded_snapshot.to(device).float()

    # generate mask cube via numpy/cv2 then convert to torch
    mask_cube_np = Modulation.generate_shifted_mask_cube(mask2d, chs=img_n_chs)
    mask3d = torch.from_numpy(mask_cube_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)

    if gt_hs is not None:
        if not torch.is_tensor(gt_hs):
            raise ValueError("gt_hs must be a torch tensor of shape (C,H,W)")
        gt_tensor = gt_hs.unsqueeze(0).to(device).float()
    else:
        gt_tensor = None

    H, W = mask2d.shape
    R = 64  # latent channels
    autoencoder = load_autoencoder_from_checkpoint(autoencoder_path, device, in_channels=img_n_chs, R=R)

    recon_model = HyperspectralReconstruction(autoencoder, img_h=H, img_w=W, img_chs=img_n_chs, n_features_code=R).to(device)
    recon_model.reset_admm_variables()

    optimizer = optim.Adam([recon_model.xk], lr=param_learning_rate, weight_decay=Config.WEIGHT_DECAY_LAMBDA)
    param_lambda = param_sparsity * param_rho

    for i_admm in range(n_iters_ADMM):
        print(f'ADMM iter {i_admm+1}/{n_iters_ADMM}')
        for i_inner in range(n_iters_ADAM):
            optimizer.zero_grad()
            losses = recon_model.compute_losses(coded_snapshot, mask3d, param_rho, param_lambda_alpha_fidelity)
            losses['total'].backward()
            optimizer.step()

            if (i_inner % 50) == 0:
                total = losses['total'].item()
                data_l = losses['data'].item()
                admm_l = losses['admm'].item()
                alpha_l = losses['alpha_fidelity'].item()
                print(f'  inner {i_inner}: total={total:.6f} data={data_l:.6f} admm={admm_l:.6f} alpha={alpha_l:.6f}')
                if gt_tensor is not None:
                    with torch.no_grad():
                        recon = losses['img_recon']
                        mse = torch.mean((recon - gt_tensor) ** 2)
                        psnr = -10.0 * torch.log10(mse + 1e-12)
                        print(f'    PSNR {psnr.item():.2f} dB')

        with torch.no_grad():
            img_recon = recon_model.autoencoder.decoder(recon_model.xk)
            if img_recon.dim() == 3:
                img_recon = img_recon.unsqueeze(0)
            recon_model.admm_update(img_recon, param_lambda, param_rho)

    with torch.no_grad():
        img_recon_final = recon_model.autoencoder.decoder(recon_model.xk)
        if img_recon_final.dim() == 3:
            img_recon_final = img_recon_final.unsqueeze(0)
        img_recon_final = torch.clamp(img_recon_final, 0.0, 1.0)
        img_recon_np = img_recon_final.squeeze(0).permute(1, 2, 0).cpu().numpy()

    wavelengths = np.arange(400, 400 + img_n_chs * 10, 10).astype(np.float32)
    return img_recon_np, wavelengths

# Example usage (you provide tensors/checkpoint)
if __name__ == "__main__":
    # user should replace these with real tensors/paths
    H = 96; W = 96; C = 31
    # ground truth tensor (C,H,W)
    gt_hs_tensor = torch.load("test/balloons_ms_patch109.pt")  # user-provided tensor
    # 2D mask generation (numpy)
    mask2d = Modulation.generate_random_mask(h=H, w=W, scale=1.0)
    # generate mask3d & coded image purely with numpy for demo
    mask_cube = Modulation.generate_shifted_mask_cube(mask2d, chs=C)
    gt_np = gt_hs_tensor.permute(1,2,0).numpy()
    coded_np = Modulation.generate_coded_image_from_cube(gt_np, mask_cube)
    coded_torch = torch.from_numpy(coded_np).float()

    # path to checkpoint
    checkpoint_path = 'checkpoints/best_autoencoder.pth'
    recon, wls = reconstruct_snapshot_tensors(coded_torch, mask2d, checkpoint_path, gt_hs=gt_hs_tensor,
                                               img_n_chs=C, n_iters_ADMM=2, n_iters_ADAM=100,
                                               device='cpu')
    print("recon shape:", recon.shape, "wls len:", len(wls))
