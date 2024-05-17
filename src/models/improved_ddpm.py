import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import random


from tqdm import tqdm


class IDDPMTrainer:
    def __init__(self, iddpm, device, lr=1e-4, Lambda=1e-3):
        self.n_steps = iddpm.n_steps
        self.Lambda = Lambda
        self.best_loss = float("inf")
        self.device = device
        self.model = iddpm
        self.t_vals = np.arange(0, self.n_steps, 1)
        self.t_dist = torch.distributions.uniform.Uniform(
            float(1) - float(0.499), float(self.n_steps) + float(0.499)
        )
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-4)
        self.mse_loss = nn.MSELoss(reduction="none").to(self.device)
        self.losses = np.zeros((self.n_steps, 10))
        self.losses_ct = np.zeros(self.n_steps, dtype=int)

    def update_losses(self, loss_vlb, t):
        for t_val, loss in zip(t, loss_vlb):
            if self.losses_ct[t_val] == 10:
                self.losses[t_val] = np.concatenate((self.losses[t_val][1:], [loss]))
            else:
                self.losses[t_val, self.losses_ct[t_val]] = loss
                self.losses_ct[t_val] += 1

    def loss_simple(self, eps, eps_theta):
        return ((eps_theta - eps) ** 2).flatten(1, -1).mean(-1)

    # Formula derived from: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    def loss_vlb_gauss(self, mean_real, mean_fake, var_real, var_fake):
        """KL divergence between two gaussians."""
        std_real = torch.sqrt(var_real)
        std_fake = torch.sqrt(var_fake)
        kl_div = (
            (
                torch.log(std_fake / std_real)
                + ((var_real) + (mean_real - mean_fake) ** 2) / (2 * (var_fake))
                - torch.tensor(1 / 2)
            )
            .flatten(1, -1)
            .mean(-1)
        )
        return kl_div

    def calc_losses(self, eps, eps_theta, var_theta, x, x_with_noise, t):

        mean_t_pred = self.model.noise_to_mean(eps_theta, x_with_noise, t, True)
        var_t_pred = self.model.vs_to_variance(var_theta, t)

        beta_t = self.model.scheduler.beta_t[t]
        a_bar_t = self.model.scheduler.a_bar_t[t]
        a_bar_t1 = self.model.scheduler.a_bar_t1[t]
        beta_tilde_t = self.model.scheduler.beta_tilde_t[t]
        sqrt_a_bar_t1 = self.model.scheduler.sqrt_a_bar_t1[t]
        sqrt_a_t = self.model.scheduler.sqrt_a_t[t]

        mean_t = ((sqrt_a_bar_t1 * beta_t) / (1 - a_bar_t)) * x + (
            (sqrt_a_t * (1 - a_bar_t1)) / (1 - a_bar_t)
        ) * x_with_noise

        loss_simple = self.loss_simple(eps, eps_theta)
        loss_vlb = (
            self.loss_vlb_gauss(mean_t, mean_t_pred.detach(), beta_tilde_t, var_t_pred)
            * self.Lambda
        )
        loss_hybrid = loss_simple + loss_vlb

        with torch.no_grad():
            t = t.detach().cpu().numpy()
            loss = loss_vlb.detach().cpu()
            self.update_losses(loss, t)

            if np.sum(self.losses_ct) == self.losses.size - 20:
                p_t = np.sqrt((self.losses**2).mean(-1))
                p_t = p_t / p_t.sum()
                loss = loss / torch.tensor(p_t[t], device=self.device)

        return loss_hybrid.mean(), loss_simple.mean(), loss_vlb.mean()

    def train(self, loader, n_epochs, model_store_path=None):
        self.model.train()
        n = len(loader.dataset)

        self.losses_comb = np.array([])
        self.losses_mean = np.array([])
        self.losses_var = np.array([])
        self.steps_list = np.array([])

        losses_comb_s = torch.tensor(0.0, requires_grad=False)
        losses_mean_s = torch.tensor(0.0, requires_grad=False)
        losses_var_s = torch.tensor(0.0, requires_grad=False)

        for epoch in range(n_epochs):
            for _, batch in enumerate(
                tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")
            ):
                x = batch[0]
                if np.sum(self.losses_ct) == self.losses.size - 20:
                    # Weights for each value of t
                    p_t = np.sqrt((self.losses**2).mean(-1))
                    p_t = p_t / p_t.sum()
                    t = torch.tensor(
                        np.random.choice(self.t_vals, size=x.shape[0], p=p_t),
                        device=self.device,
                    )
                else:
                    t = self.t_dist.sample((x.shape[0],)).to(self.device)
                    t = torch.round(t).to(torch.long)

                with torch.no_grad():
                    eps = torch.randn_like(x).to(device)
                    x_with_noise = self.model(x, t, eps)

                print(x_with_noise.shape, t.shape)
                eps_theta, var_theta = self.model.backward(x_with_noise, t)
                loss, loss_mean, loss_var = self.calc_losses(
                    eps, eps_theta, var_theta, x, x_with_noise, t
                )
                loss = loss * x.shape[0] / n
                loss_mean = loss_mean * x.shape[0] / n
                loss_var = loss_var * x.shape[0] / n

                # Backprop the loss, but save the intermediate gradients
                loss.backward()

                losses_comb_s += loss.cpu().detach()
                losses_mean_s += loss_mean.cpu().detach()
                losses_var_s += loss_var.cpu().detach()

                self.optim.step()
                self.optim.zero_grad()

                self.losses_comb = np.append(self.losses_comb, losses_comb_s.item())
                self.losses_mean = np.append(self.losses_mean, losses_mean_s.item())
                self.losses_var = np.append(self.losses_var, losses_var_s.item())

                losses_comb_s *= 0
                losses_mean_s *= 0
                losses_var_s *= 0

            print(
                f"Loss at epoch {epoch + 1}: "
                + f"Combined: {round(self.losses_comb[-10:].mean(), 4)}    "
                f"Mean: {round(self.losses_mean[-10:].mean(), 4)}    "
                f"Variance: {round(self.losses_var[-10:].mean(), 6)}\n\n"
            )

            if model_store_path is not None and self.best_loss > loss:
                self.best_loss = loss
                torch.save(self.model.state_dict(), model_store_path)
                print("Best model ever (stored)")
