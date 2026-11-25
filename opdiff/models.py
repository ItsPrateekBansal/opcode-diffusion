"""GAN, WGAN-GP and 1D U-Net diffusion for 104-dim embeddings (Bao et al., Tables 5-6).

Notes on matching the paper:
- Keras BatchNorm momentum 0.8 equals PyTorch momentum 0.2.
- Settings the paper does not state (latent size, batch size, epochs, critic
  steps, gradient-penalty weight) use the values from my reproduction report.
- Diffusion: the paper trains and samples only up to T/2 and starts sampling
  from real samples noised to T/2. Here both are parameters:
  `train_tmax` (largest timestep seen in training) and `start_t` (where
  sampling starts). start_t = T means pure noise (standard DDPM).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    bs = min(bs, n)
    for s in range(0, n - bs + 1, bs):  # drop the last partial batch
        yield idx[s:s + bs]


def _as_tensor(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


# ======================================================================= GAN
class GanGenerator(nn.Module):
    def __init__(self, z_dim=100, out_dim=104):
        super().__init__()
        layers, d = [], z_dim
        for h in (200, 400, 800):
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h, momentum=0.2), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class GanDiscriminator(nn.Module):
    def __init__(self, in_dim=104):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 400), nn.LeakyReLU(0.2),
                                 nn.Linear(400, 200), nn.LeakyReLU(0.2),
                                 nn.Linear(200, 1))  # sigmoid is inside the loss

    def forward(self, x):
        return self.net(x)


def train_gan(X, cfg, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z_dim, bs = cfg.get("z_dim", 100), cfg.get("batch", 64)
    G, D = GanGenerator(z_dim, X.shape[1]).to(device), GanDiscriminator(X.shape[1]).to(device)
    lr = cfg.get("lr", 3e-4)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    data = _as_tensor(X, device)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(cfg.get("epochs", 200)):
        for b in _batches(len(X), bs, rng):
            real = data[b]
            z = torch.randn(len(b), z_dim, device=device)
            fake = G(z)
            lossD = bce(D(real), torch.ones(len(b), 1, device=device)) + \
                bce(D(fake.detach()), torch.zeros(len(b), 1, device=device))
            optD.zero_grad(); lossD.backward(); optD.step()
            lossG = bce(D(fake), torch.ones(len(b), 1, device=device))
            optG.zero_grad(); lossG.backward(); optG.step()
    return G


@torch.no_grad()
def sample_gan(G, n, device, z_dim=100):
    G.eval()
    return G(torch.randn(n, z_dim, device=device)).cpu().numpy()


# =================================================================== WGAN-GP
class WganGenerator(nn.Module):
    """z -> linear to 64 x 26 -> three Conv1D blocks (64, 32, 16 filters, k=3,
    same padding) with x2 upsampling (26 -> 52 -> 104) -> dense -> tanh."""

    def __init__(self, z_dim=100, out_dim=104):
        super().__init__()
        assert out_dim % 4 == 0
        self.L = out_dim // 4
        self.fc_in = nn.Linear(z_dim, 64 * self.L)
        def block(ci, co):
            return [nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co, momentum=0.2), nn.Tanh()]
        self.conv = nn.Sequential(
            nn.BatchNorm1d(64, momentum=0.2), nn.Tanh(),
            *block(64, 64), nn.Upsample(scale_factor=2),
            *block(64, 32), nn.Upsample(scale_factor=2),
            *block(32, 16))
        self.fc_out = nn.Linear(16 * out_dim, out_dim)

    def forward(self, z):
        h = self.fc_in(z).view(-1, 64, self.L)
        h = self.conv(h)
        return torch.tanh(self.fc_out(h.flatten(1)))


class WganCritic(nn.Module):
    """Three Conv1D layers (64, 128, 256 filters, k=3, same), LeakyReLU, no BatchNorm."""

    def __init__(self, in_dim=104):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, padding=1), nn.LeakyReLU(0.2))
        self.out = nn.Linear(256 * in_dim, 1)

    def forward(self, x):  # x: (b, 104); reshaped to Conv1D input here
        return self.out(self.net(x.unsqueeze(1)).flatten(1))


def _gradient_penalty(C, real, fake):
    eps = torch.rand(real.size(0), 1, device=real.device)  # (b, 1): data is (b, 104)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad = torch.autograd.grad(C(mix).sum(), mix, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_wgangp(X, cfg, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z_dim, bs = cfg.get("z_dim", 100), cfg.get("batch", 64)
    n_critic, lam = cfg.get("n_critic", 5), cfg.get("gp_lambda", 10.0)
    G, C = WganGenerator(z_dim, X.shape[1]).to(device), WganCritic(X.shape[1]).to(device)
    lr = cfg.get("lr", 1e-4)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.9))
    data = _as_tensor(X, device)
    step = 0
    for _ in range(cfg.get("epochs", 200)):
        for b in _batches(len(X), bs, rng):
            real = data[b]
            fake = G(torch.randn(len(b), z_dim, device=device)).detach()
            lossC = C(fake).mean() - C(real).mean() + lam * _gradient_penalty(C, real, fake)
            optC.zero_grad(); lossC.backward(); optC.step()
            step += 1
            if step % n_critic == 0:
                lossG = -C(G(torch.randn(len(b), z_dim, device=device))).mean()
                optG.zero_grad(); lossG.backward(); optG.step()
    return G


sample_wgangp = sample_gan


# ================================================================= diffusion
class TimeEmbedding(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, ci, co, tdim, dropout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ci), nn.Conv1d(ci, co, 3, padding=1)
        self.t = nn.Linear(tdim, co)
        self.n2, self.c2 = nn.GroupNorm(8, co), nn.Conv1d(co, co, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x))) + self.t(F.silu(temb))[:, :, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm, self.attn = nn.GroupNorm(8, c), nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        return x + self.attn(h, h, h, need_weights=False)[0].transpose(1, 2)


class UNet1D(nn.Module):
    """4 encoder / 4 decoder levels, base 32, multipliers (1, 2, 4, 8),
    dropout 0.01, SiLU, attention at the bottleneck (Table 6)."""

    def __init__(self, base=32, mults=(1, 2, 4, 8), dropout=0.01, temb=64):
        super().__init__()
        tdim = base * 8
        self.time = TimeEmbedding(temb, tdim)
        self.inp = nn.Conv1d(1, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = base
        for ch in chs:
            self.down.append(ResBlock(c, ch, tdim, dropout))
            self.pool.append(nn.Conv1d(ch, ch, 4, stride=2, padding=1))
            c = ch
        self.mid1, self.mid_attn, self.mid2 = (ResBlock(c, c, tdim, dropout), Attention(c),
                                               ResBlock(c, c, tdim, dropout))
        self.up, self.unpool = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs):
            self.unpool.append(nn.ConvTranspose1d(c, c, 4, stride=2, padding=1))
            self.up.append(ResBlock(c + ch, ch, tdim, dropout))
            c = ch
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.time(t)
        h, skips = self.inp(x), []
        for blk, pool in zip(self.down, self.pool):
            h = blk(h, temb)
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for blk, unpool in zip(self.up, self.unpool):
            h = blk(torch.cat([unpool(h), skips.pop()], dim=1), temb)
        return self.out(h)


class Diffusion:
    """Linear beta schedule, T = 1000. Inputs padded from 104 to 128 so the
    length can be halved four times (128 -> 64 -> 32 -> 16 -> 8)."""

    def __init__(self, T=1000, beta1=1e-4, betaT=0.02, pad_to=128, device="cpu"):
        self.T, self.pad_to, self.device = T, pad_to, device
        self.beta = torch.linspace(beta1, betaT, T, device=device)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)

    def pad(self, x):
        return F.pad(x, (0, self.pad_to - x.shape[-1]))

    def q_sample(self, x0, t, noise):  # t in 1..T
        ab = self.abar[t - 1].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def train(self, X, cfg, seed):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        tmax, bs = cfg.get("train_tmax", self.T // 2), cfg.get("batch", 128)
        net = UNet1D(dropout=cfg.get("dropout", 0.01)).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 5e-5), betas=(0.9, 0.999))
        data = self.pad(_as_tensor(X, self.device).unsqueeze(1))
        net.train()
        for _ in range(cfg.get("epochs", 200)):
            for b in _batches(len(X), bs, rng):
                x0 = data[b]
                t = torch.randint(1, tmax + 1, (len(b),), device=self.device)
                noise = torch.randn_like(x0)
                loss = F.mse_loss(net(self.q_sample(x0, t, noise), t), noise)
                opt.zero_grad(); loss.backward(); opt.step()
        return net

    @torch.no_grad()
    def sample(self, net, X_seed, n, start_t, seed, batch=4096):
        """Algorithm 1: start from real samples noised to start_t, then denoise.
        start_t >= T starts from pure Gaussian noise instead."""
        net.eval()
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng = np.random.default_rng(seed)
        seed_idx = rng.integers(0, len(X_seed), size=n)
        out = []
        for s in range(0, n, batch):
            idx = seed_idx[s:s + batch]
            x0 = self.pad(_as_tensor(X_seed[idx], self.device).unsqueeze(1))
            noise = torch.randn(x0.shape, generator=g, device=self.device)
            if start_t >= self.T:
                x, t0 = noise, self.T
            else:
                x, t0 = self.q_sample(x0, torch.full((len(idx),), start_t, device=self.device), noise), start_t
            for t in range(t0, 0, -1):
                tt = torch.full((len(idx),), t, device=self.device)
                eps = net(x, tt)
                a, ab, b = self.alpha[t - 1], self.abar[t - 1], self.beta[t - 1]
                x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                if t > 1:
                    x = x + b.sqrt() * torch.randn(x.shape, generator=g, device=self.device)
            out.append(x[:, 0, :X_seed.shape[1]].clamp(-1, 1).cpu().numpy())
        return np.concatenate(out), seed_idx
