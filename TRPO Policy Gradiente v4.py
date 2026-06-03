# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:44:39 2026

@author: Leonimer
"""

"""
=============================================================================
TRPO — Trust Region Policy Optimization  +  Métricas de Treinamento
Implementação didática em PyTorch
Referência: Schulman et al., 2015 (https://arxiv.org/abs/1502.05477)
=============================================================================

Visão geral do algoritmo:
--------------------------
TRPO é um método de Policy Gradient que garante melhorias monotônicas na
política ao restringir o tamanho do passo de atualização usando a divergência
KL como medida de distância entre políticas.

Ideia central:
    maximizar  L(θ)  (surrogate objective / ganho esperado)
    sujeito a  KL(π_old || π_new) ≤ δ  (trust region constraint)

Etapas:
  1. Coleta de trajetórias com a política atual (π_old)
  2. Estimativa das vantagens (Advantage) via GAE
  3. Cálculo do gradiente natural (conjugate gradient)
  4. Line search para garantir a restrição de KL
  5. Atualização da política dentro da trust region
  6. Atualização da função de valor (crítico) por regressão

Métricas coletadas automaticamente durante o treino:
  - Recompensa por episódio (+ média móvel e banda de variância)
  - KL divergência por update (com linha do limite δ)
  - Step time decomposto: CG+FVP / Line search / Crítico
  - Histograma de step times totais (percentis p50/p90/p99)
  - Perda do crítico (MSE em escala log)
  - Iterações do backtracking line search (aceitas vs. falhas)

=============================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from dataclasses import dataclass, field
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional


# =============================================================================
# 1. LOGGER DE MÉTRICAS
# =============================================================================

@dataclass
class TRPOLogger:
    """
    Coleta métricas em tempo real durante o treinamento TRPO.

    Métricas por episódio
    ---------------------
    episode_rewards : recompensa acumulada de cada episódio

    Métricas por update de política
    --------------------------------
    kl_divs       : KL(π_old ‖ π_new) após o line search aceito
    time_cg       : tempo (ms) gasto no Conjugate Gradient + FVP
    time_ls       : tempo (ms) gasto no backtracking line search
    time_critic   : tempo (ms) gasto nas épocas de atualização do crítico
    value_losses  : MSE final do crítico em cada update
    ls_iters      : iteração do line search em que o passo foi aceito (0 = falhou)
    ls_success    : True se o line search encontrou um passo válido
    mean_advantages: média das vantagens normalizadas do batch
    """
    # Por episódio
    episode_rewards  : List[float] = field(default_factory=list)

    # Por update
    kl_divs          : List[float] = field(default_factory=list)
    time_cg          : List[float] = field(default_factory=list)   # ms
    time_ls          : List[float] = field(default_factory=list)   # ms
    time_critic      : List[float] = field(default_factory=list)   # ms
    value_losses     : List[float] = field(default_factory=list)
    ls_iters         : List[int]   = field(default_factory=list)
    ls_success       : List[bool]  = field(default_factory=list)
    mean_advantages  : List[float] = field(default_factory=list)

    def log_episode(self, reward: float):
        self.episode_rewards.append(reward)

    def log_update(
        self,
        kl         : float,
        time_cg    : float,
        time_ls    : float,
        time_critic: float,
        value_loss : float,
        ls_iters   : int,
        ls_success : bool,
        mean_adv   : float = 0.0,
    ):
        self.kl_divs.append(kl)
        self.time_cg.append(time_cg)
        self.time_ls.append(time_ls)
        self.time_critic.append(time_critic)
        self.value_losses.append(value_loss)
        self.ls_iters.append(ls_iters)
        self.ls_success.append(ls_success)
        self.mean_advantages.append(mean_adv)

    @property
    def time_total(self) -> List[float]:
        """Tempo total de cada update = CG + line search + crítico."""
        return [c + l + r for c, l, r in zip(self.time_cg, self.time_ls, self.time_critic)]


# =============================================================================
# 2. REDES NEURAIS — ATOR E CRÍTICO
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Rede do Ator (Policy Network) — π(a|s; θ)

    Para ambientes discretos : retorna logits → distribuição Categorical
    Para ambientes contínuos : retorna média e log_std → distribuição Normal
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64, continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        # Backbone compartilhado
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if continuous:
            self.mean_head = nn.Linear(hidden_dim, act_dim)
            # log_std independente do estado — parâmetro global
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.logits_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        """Retorna a distribuição π(·|obs)."""
        features = self.net(obs)
        if self.continuous:
            mean = self.mean_head(features)
            std  = self.log_std.exp().expand_as(mean)
            return Normal(mean, std)
        else:
            return Categorical(logits=self.logits_head(features))

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-probabilidade das ações tomadas."""
        return self.forward(obs).log_prob(actions)

    def get_kl(self, obs: torch.Tensor, old_dist_params: dict) -> torch.Tensor:
        """
        KL(π_old ‖ π_new) média sobre o batch de estados.
        Usa torch.distributions.kl_divergence internamente.
        """
        new_dist = self.forward(obs)
        if self.continuous:
            old_dist = Normal(old_dist_params["mean"], old_dist_params["std"])
        else:
            old_dist = Categorical(logits=old_dist_params["logits"])
        return torch.distributions.kl_divergence(old_dist, new_dist).mean()


class ValueNetwork(nn.Module):
    """
    Rede do Crítico (Value Network) — V(s; φ)

    Estima o retorno esperado a partir de s.
    Usada para calcular a vantagem: A(s,a) = Q(s,a) - V(s)
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# =============================================================================
# 3. BUFFER DE TRAJETÓRIAS
# =============================================================================

class TrajectoryBuffer:
    """
    Armazena transições (s, a, r, done, log_π, V) de um batch de coleta.
    Limpo após cada update.
    """
    def __init__(self):
        self.obs      : List[np.ndarray] = []
        self.actions  : List[np.ndarray] = []
        self.rewards  : List[float]      = []
        self.dones    : List[bool]       = []
        self.log_probs: List[float]      = []
        self.values   : List[float]      = []

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs);        self.actions.append(action)
        self.rewards.append(reward); self.dones.append(done)
        self.log_probs.append(log_prob); self.values.append(value)

    def get_tensors(self, device: str = "cpu"):
        obs       = torch.FloatTensor(np.array(self.obs)).to(device)
        actions   = torch.FloatTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values    = torch.FloatTensor(np.array(self.values)).to(device)
        return obs, actions, log_probs, values

    def clear(self):
        self.__init__()


# =============================================================================
# 4. GAE — GENERALIZED ADVANTAGE ESTIMATION
# =============================================================================

def compute_gae(
    rewards   : List[float],
    values    : torch.Tensor,
    dones     : List[bool],
    gamma     : float = 0.99,
    lam       : float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE (Schulman et al., 2016):

        A^GAE_t = Σ_{l≥0} (γλ)^l · δ_{t+l}
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)   (TD residual)

    Returns:
        advantages : vantagens normalizadas (média 0, std 1)
        returns    : retornos descontados = A + V  (targets do crítico)
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1].item()
        mask   = 1.0 - float(dones[t])
        delta  = rewards[t] + gamma * next_v * mask - values[t].item()
        gae    = delta + gamma * lam * mask * gae
        advantages[t] = gae

    advantages = torch.FloatTensor(advantages)
    returns    = advantages + values.cpu()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


# =============================================================================
# 5. GRADIENTE NATURAL — NÚCLEO DO TRPO
# =============================================================================

def flat_grad(y: torch.Tensor, params, create_graph: bool = False) -> torch.Tensor:
    """
    Concatena gradientes ∂y/∂p para todos os parâmetros p em um vetor 1D.
    """
    grads = torch.autograd.grad(y, params, create_graph=create_graph, allow_unused=True)
    return torch.cat([
        g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)
    ])


def fisher_vector_product(
    policy    : PolicyNetwork,
    obs       : torch.Tensor,
    old_params: dict,
    vector    : torch.Tensor,
    damping   : float = 0.1,
) -> torch.Tensor:
    """
    Produto implícito Fv = (∇²_θ KL) · v via truque de Pearlmutter.

    Custo: O(n) em parâmetros (dois passes de backprop) em vez de O(n²).
    Regularização diagonal (+λI) evita F singular.
    """
    kl      = policy.get_kl(obs, old_params)
    grad_kl = flat_grad(kl, list(policy.parameters()), create_graph=True)
    kl_v    = (grad_kl * vector).sum()
    fvp     = flat_grad(kl_v, list(policy.parameters()), create_graph=False)
    return fvp + damping * vector


def conjugate_gradient(
    fvp_fn,
    b      : torch.Tensor,
    n_iters: int   = 10,
    tol    : float = 1e-10,
) -> torch.Tensor:
    """
    Resolve Fx = b (gradiente natural x* = F⁻¹g) sem montar F.

    Algoritmo Hestenes-Stiefel:
        α = rᵀr / pᵀFp
        x ← x + αp
        r ← r − αFp
        β = r_new·r_new / r_old·r_old
        p ← r + βp
    """
    x = torch.zeros_like(b)
    r = b.clone(); p = b.clone()
    rdotr = r.dot(r)

    for _ in range(n_iters):
        Fp        = fvp_fn(p)
        alpha     = rdotr / (p.dot(Fp) + 1e-8)
        x         = x + alpha * p
        r         = r - alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < tol:
            break
        beta  = new_rdotr / (rdotr + 1e-8)
        p     = r + beta * p
        rdotr = new_rdotr

    return x


# =============================================================================
# 6. AGENTE TRPO
# =============================================================================

class TRPOAgent:
    """
    Agente TRPO completo com coleta integrada de métricas.

    Parâmetros chave
    ----------------
    delta           : raio da trust region  KL(π_old ‖ π_new) ≤ δ
    backtrack_coeff : fator de redução do passo no line search  (c < 1)
    backtrack_iters : máximo de tentativas no backtracking
    damping         : regularização diagonal da Fisher  (F + λI)
    logger          : TRPOLogger externo (criado em train() e passado aqui)
    """

    def __init__(
        self,
        obs_dim         : int,
        act_dim         : int,
        logger          : TRPOLogger,
        continuous      : bool  = False,
        hidden_dim      : int   = 64,
        gamma           : float = 0.99,
        lam             : float = 0.95,
        delta           : float = 0.01,
        damping         : float = 0.1,
        cg_iters        : int   = 10,
        backtrack_coeff : float = 0.8,
        backtrack_iters : int   = 10,
        value_lr        : float = 1e-3,
        value_epochs    : int   = 5,
        device          : str   = "cpu",
    ):
        self.gamma           = gamma
        self.lam             = lam
        self.delta           = delta
        self.damping         = damping
        self.cg_iters        = cg_iters
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.value_epochs    = value_epochs
        self.continuous      = continuous
        self.device          = device
        self.logger          = logger   # ← referência ao logger externo

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_dim, continuous).to(device)
        self.value  = ValueNetwork(obs_dim, hidden_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.buffer = TrajectoryBuffer()

    # ── 6a. Seleção de ação ──────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """Amostra ação, retorna (action, log_prob, value)."""
        obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        dist     = self.policy(obs_t)
        action   = dist.sample()
        log_prob = dist.log_prob(action).squeeze(0).sum().item()
        value    = self.value(obs_t).squeeze(0).item()
        return action.squeeze(0).cpu().numpy(), log_prob, value

    # ── 6b. Snapshot da política antiga ─────────────────────────────────────

    @torch.no_grad()
    def get_old_dist_params(self, obs: torch.Tensor) -> dict:
        """Captura os parâmetros de π_old (antes do update)."""
        dist = self.policy(obs)
        if self.continuous:
            return {"mean": dist.loc.detach(), "std": dist.scale.detach()}
        return {"logits": dist.logits.detach()}

    # ── 6c. Surrogate objective ──────────────────────────────────────────────

    def surrogate_objective(
        self,
        obs          : torch.Tensor,
        actions      : torch.Tensor,
        advantages   : torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        L(θ) = E_t [ r_t(θ) · A_t ]   onde  r_t = π_θ / π_old

        Equivalente ao ganho esperado estimado com importância ponderada.
        """
        new_log_probs = self.policy.get_log_prob(obs, actions)
        if self.continuous and actions.dim() > 1:
            new_log_probs = new_log_probs.sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()

    # ── 6d. Manipulação de parâmetros ────────────────────────────────────────

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in self.policy.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view_as(p))
            idx += n

    # ── 6e. Atualização da política (TRPO) ───────────────────────────────────

    def update_policy(
        self,
        obs          : torch.Tensor,
        actions      : torch.Tensor,
        advantages   : torch.Tensor,
        old_log_probs: torch.Tensor,
        old_params   : dict,
    ) -> Tuple[float, int, bool]:
        """
        4 etapas do update TRPO:

        1. g  = ∇_θ L(θ_old)                   (gradiente da objective)
        2. x* = F⁻¹g  via Conjugate Gradient    (gradiente natural)
        3. α  = √(2δ / x*ᵀFx*)                 (escala do passo máximo)
        4. Backtracking line search até KL ≤ δ e L melhora

        Returns: (kl_final, ls_iter_aceita, sucesso)
        """
        # ── Etapa 1: gradiente da objective ──
        t0 = time.perf_counter()
        obj           = self.surrogate_objective(obs, actions, advantages, old_log_probs)
        policy_params = list(self.policy.parameters())
        g             = flat_grad(obj, policy_params)

        # ── Etapa 2: conjugate gradient ──
        def fvp(v):
            return fisher_vector_product(self.policy, obs, old_params, v, self.damping)

        x        = conjugate_gradient(fvp, g.detach(), n_iters=self.cg_iters)
        t_cg_ms  = (time.perf_counter() - t0) * 1000  # ms

        # ── Etapa 3: tamanho máximo do passo ──
        # Expansão de Taylor de 2ª ordem da KL: KL ≈ ½ sᵀFs
        # Para s = α·x: α = √(2δ / xᵀFx)
        xFx       = x.dot(fvp(x))
        max_step  = torch.sqrt(2 * self.delta / (xFx + 1e-8))
        full_step = max_step * x

        # ── Etapa 4: backtracking line search ──
        t1              = time.perf_counter()
        old_params_flat = self._get_flat_params().detach()
        old_obj         = obj.item()
        kl_final        = 0.0
        ls_iter         = 0
        success         = False

        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff ** i          # c^i decrescente
            self._set_flat_params(old_params_flat + step_size * full_step)

            kl      = self.policy.get_kl(obs, old_params)
            new_obj = self.surrogate_objective(obs, actions, advantages, old_log_probs).item()

            if kl.item() <= self.delta and new_obj >= old_obj:
                kl_final = kl.item()
                ls_iter  = i
                success  = True
                break

        if not success:
            # Nenhum passo válido — restaura parâmetros
            self._set_flat_params(old_params_flat)
            kl_final = self.policy.get_kl(obs, old_params).item()

        t_ls_ms = (time.perf_counter() - t1) * 1000  # ms
        return kl_final, t_cg_ms, t_ls_ms, ls_iter, success

    # ── 6f. Atualização do crítico ───────────────────────────────────────────

    def update_value(self, obs: torch.Tensor, returns: torch.Tensor) -> Tuple[float, float]:
        """
        Minimiza MSE:  L(φ) = E_t [(V_φ(s_t) − R_t)²]
        Usa Adam com `value_epochs` épocas por update.

        Returns: (value_loss_final, tempo_ms)
        """
        t0 = time.perf_counter()
        for _ in range(self.value_epochs):
            pred   = self.value(obs)
            vloss  = F.mse_loss(pred, returns)
            self.value_optimizer.zero_grad()
            vloss.backward()
            self.value_optimizer.step()
        t_cr_ms = (time.perf_counter() - t0) * 1000
        return vloss.item(), t_cr_ms

    # ── 6g. Update completo ──────────────────────────────────────────────────

    def update(self):
        """
        Ciclo completo de um update TRPO:
          1. Computa GAE sobre o buffer
          2. Atualiza a política (gradiente natural + line search)
          3. Atualiza o crítico (Adam)
          4. Registra todas as métricas no logger
          5. Limpa o buffer
        """
        obs, actions, old_log_probs, values = self.buffer.get_tensors(self.device)

        # GAE
        advantages, returns = compute_gae(
            rewards    = self.buffer.rewards,
            values     = values,
            dones      = self.buffer.dones,
            gamma      = self.gamma,
            lam        = self.lam,
            last_value = 0.0,
        )
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        # Snapshot π_old
        old_params = self.get_old_dist_params(obs)

        # Atualização da política (com temporização interna)
        kl, t_cg, t_ls, ls_iter, ls_ok = self.update_policy(
            obs, actions, advantages, old_log_probs, old_params
        )

        # Atualização do crítico (com temporização interna)
        vloss, t_cr = self.update_value(obs, returns)

        # ── Registra métricas no logger ──────────────────────────────────
        self.logger.log_update(
            kl          = kl,
            time_cg     = t_cg,
            time_ls     = t_ls,
            time_critic = t_cr,
            value_loss  = vloss,
            ls_iters    = ls_iter,
            ls_success  = ls_ok,
            mean_adv    = advantages.mean().item(),
        )

        self.buffer.clear()


# =============================================================================
# 7. VISUALIZAÇÃO DE MÉTRICAS
# =============================================================================

def _moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    """Média móvel causal com janela w."""
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        sl = arr[max(0, i - w + 1): i + 1]
        out[i] = sl.mean()
    return out

def _moving_std(arr: np.ndarray, w: int) -> np.ndarray:
    """Desvio padrão móvel causal com janela w."""
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        sl = arr[max(0, i - w + 1): i + 1]
        out[i] = sl.std()
    return out

def _style(ax, title="", xlabel="", ylabel=""):
    """Estilo visual consistente para todos os subplots."""
    ax.set_facecolor("#f7f7f7")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555", labelsize=8)
    ax.yaxis.label.set_color("#555"); ax.xaxis.label.set_color("#555")
    if title:   ax.set_title(title, fontsize=10, fontweight="bold", color="#111", pad=6)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color="#ddd", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", color="#eee", linewidth=0.4)


def _plot_rewards(ax, logger: TRPOLogger, window: int = 50):
    """Recompensa por episódio + média móvel + banda ±1σ."""
    r   = np.array(logger.episode_rewards, dtype=float)
    eps = np.arange(1, len(r) + 1)
    ma  = _moving_avg(r, window)
    std = _moving_std(r, window)

    ax.plot(eps, r,  color="#3266ad", alpha=0.25, lw=0.8, label="por episódio")
    ax.fill_between(eps, ma - std, ma + std, color="#3266ad", alpha=0.13, label=f"±1σ (w={window})")
    ax.plot(eps, ma, color="#e07b3a", lw=2.0, label=f"média móvel ({window} ep)")

    _style(ax, title="Recompensa por episódio", xlabel="episódio", ylabel="recompensa")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")

    # Anotação do valor final
    ax.annotate(
        f"final: {ma[-1]:.1f}",
        xy=(eps[-1], ma[-1]), xytext=(-50, 12), textcoords="offset points",
        fontsize=8, color="#e07b3a",
        arrowprops=dict(arrowstyle="->", color="#e07b3a", lw=0.8),
    )


def _plot_kl(ax, logger: TRPOLogger, delta: float):
    """KL divergência por update com linha do limite δ."""
    kls = np.array(logger.kl_divs, dtype=float)
    upd = np.arange(1, len(kls) + 1)

    ax.fill_between(upd, 0, kls, color="#1d9e75", alpha=0.2)
    ax.plot(upd, kls, color="#1d9e75", lw=1.2, label="KL divergência")
    ax.axhline(delta, color="#e24b4a", lw=1.2, ls="--", label=f"δ = {delta}")

    over = kls > delta
    if over.any():
        ax.scatter(upd[over], kls[over], color="#e24b4a", s=18, zorder=5, label="violação KL")

    _style(ax, title="KL divergência por update", xlabel="update", ylabel="KL(π_old ‖ π_new)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, framealpha=0.9)


def _plot_step_times(ax, logger: TRPOLogger):
    """Barras empilhadas: CG+FVP / line search / crítico."""
    upd = np.arange(1, len(logger.time_cg) + 1)
    w   = min(5, len(upd))                          # janela de suavização
    tc  = _moving_avg(np.array(logger.time_cg),     w)
    tl  = _moving_avg(np.array(logger.time_ls),     w)
    tr  = _moving_avg(np.array(logger.time_critic),  w)

    ax.bar(upd, tc,         color="#534ab7", label="CG + FVP",     width=0.85)
    ax.bar(upd, tl, bottom=tc,      color="#85b7eb", label="Line search",  width=0.85)
    ax.bar(upd, tr, bottom=tc + tl, color="#3b6d11", label="Crítico (Adam)", width=0.85)
    ax.plot(upd, tc + tl + tr, color="#111", lw=0.8, alpha=0.45, ls="--", label="total")

    _style(ax, title="Step time por update (suavizado)", xlabel="update", ylabel="ms")
    ax.legend(fontsize=8, framealpha=0.9)


def _plot_time_hist(ax, logger: TRPOLogger):
    """Histograma de step times totais com percentis."""
    totals = np.array(logger.time_total, dtype=float)
    ax.hist(totals, bins=min(25, len(totals)), color="#534ab7", alpha=0.8,
            edgecolor="white", lw=0.5)

    for pct, col, ls in [(50, "#111", "-"), (90, "#e07b3a", "--"), (99, "#e24b4a", ":")]:
        v = np.percentile(totals, pct)
        ax.axvline(v, color=col, lw=1.2, ls=ls, label=f"p{pct}: {v:.0f} ms")

    _style(ax, title="Distribuição de step times", xlabel="ms por update", ylabel="frequência")
    ax.legend(fontsize=8, framealpha=0.9)


def _plot_value_loss(ax, logger: TRPOLogger):
    """Perda do crítico em escala logarítmica."""
    losses = np.array(logger.value_losses, dtype=float)
    upd    = np.arange(1, len(losses) + 1)
    ma     = _moving_avg(losses, min(10, len(losses)))

    ax.semilogy(upd, losses, color="#ba7517", lw=1.0, alpha=0.45, label="value loss")
    ax.semilogy(upd, ma,     color="#ba7517", lw=2.0,             label="média móvel (10)")
    ax.fill_between(upd, 1e-3, losses, color="#ba7517", alpha=0.08)

    _style(ax, title="Perda do crítico (log-MSE)", xlabel="update", ylabel="MSE (log)")
    ax.legend(fontsize=8, framealpha=0.9)


def _plot_line_search(ax, logger: TRPOLogger):
    """Iterações do line search: verde = aceito, vermelho = rollback."""
    iters   = np.array(logger.ls_iters,   dtype=int)
    success = np.array(logger.ls_success, dtype=bool)
    upd     = np.arange(1, len(iters) + 1)
    colors  = np.where(success, "#5dcaa5", "#f09595")

    ax.bar(upd, iters, color=colors, width=0.85)
    rate = success.mean() * 100
    ax.text(0.98, 0.95, f"taxa de aceitação: {rate:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

    _style(ax, title="Iterações do line search (backtracking)", xlabel="update", ylabel="iteração aceita")
    ax.set_ylim(0, max(iters.max() + 1, 4) if len(iters) else 4)
    ax.legend(handles=[
        Patch(facecolor="#5dcaa5", label="aceito"),
        Patch(facecolor="#f09595", label="falhou → rollback"),
    ], fontsize=8, framealpha=0.9)


def plot_metrics(
    logger    : TRPOLogger,
    env_name  : str   = "TRPO",
    delta     : float = 0.01,
    ma_window : int   = 50,
    save_path : Optional[str] = "trpo_metrics.png",
    dpi       : int   = 150,
) -> plt.Figure:
    """
    Gera o painel completo de métricas TRPO e salva em PNG.

    Layout (3 linhas × 3 colunas):
    ┌─────────────────────────────────────────┐
    │        Recompensa por episódio          │  ← linha 1 inteira
    ├──────────────┬──────────────┬───────────┤
    │ KL diverg.   │ Step times   │ Histograma│  ← linha 2
    ├──────────────┼──────────────┴───────────┤
    │ Value loss   │ Line search              │  ← linha 3
    └──────────────┴──────────────────────────┘

    Args:
        logger    : TRPOLogger preenchido durante treinamento
        env_name  : nome do ambiente (aparece no título)
        delta     : trust region radius (linha no gráfico de KL)
        ma_window : janela da média móvel das recompensas
        save_path : caminho de saída do PNG (None = só exibe)
        dpi       : resolução da figura salva
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"TRPO — Métricas de Treinamento  |  {env_name}",
        fontsize=14, fontweight="bold", color="#111", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    ax_rew  = fig.add_subplot(gs[0, :])     # linha 1 inteira
    ax_kl   = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[1, 2])
    ax_vl   = fig.add_subplot(gs[2, 0])
    ax_ls   = fig.add_subplot(gs[2, 1:])   # colunas 1-2

    _plot_rewards(ax_rew,  logger, window=ma_window)
    _plot_kl(ax_kl,        logger, delta=delta)
    _plot_step_times(ax_time, logger)
    _plot_time_hist(ax_hist,  logger)
    _plot_value_loss(ax_vl,   logger)
    _plot_line_search(ax_ls,  logger)

    # Sumário textual no rodapé
    totals = np.array(logger.time_total)
    rews   = np.array(logger.episode_rewards)
    summary = (
        f"episódios: {len(rews)}  |  updates: {len(logger.kl_divs)}  |  "
        f"recompensa final (MA): {_moving_avg(rews, ma_window)[-1]:.1f}  |  "
        f"step time médio: {totals.mean():.1f} ms  |  "
        f"line search OK: {np.mean(logger.ls_success)*100:.1f}%"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=8.5, color="#555")

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[métricas] Figura salva em: {save_path}")

    return fig


# =============================================================================
# 8. LOOP DE TREINAMENTO
# =============================================================================

def train(
    env_name        : str   = "CartPole-v1",
    n_episodes      : int   = 300,
    steps_per_update: int   = 2048,
    log_interval    : int   = 20,
    seed            : int   = 42,
    metrics_path    : str   = "trpo_metrics.png",
    delta           : float = 0.01,
    gamma           : float = 0.99,
    lam             : float = 0.95,
) -> TRPOLogger:
    """
    Loop principal de treinamento TRPO.

    Fluxo:
    ------
    Enquanto episódios < n_episodes:
        1. Coleta steps_per_update transições com π_atual
        2. Executa TRPOAgent.update()  (GAE → CG → line search → crítico)
        3. Logger registra todas as métricas automaticamente

    Ao final: gera e salva o painel de métricas em `metrics_path`.

    Returns:
        logger : TRPOLogger com todo o histórico de métricas
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env        = gym.make(env_name)
    obs_dim    = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim    = env.action_space.shape[0] if continuous else env.action_space.n

    print(f"Ambiente  : {env_name}")
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  contínuo={continuous}")
    print("=" * 60)

    # Logger criado aqui e injetado no agente
    logger = TRPOLogger()

    agent = TRPOAgent(
        obs_dim    = obs_dim,
        act_dim    = act_dim,
        logger     = logger,
        continuous = continuous,
        delta      = delta,
        gamma      = gamma,
        lam        = lam,
    )

    episode_count     = 0
    total_steps       = 0
    current_ep_reward = 0.0
    obs, _            = env.reset(seed=seed)
    a = 0
    steps_ = 0
    while episode_count < n_episodes:
        # ── Fase de coleta ─────────────────────────────────────────────────
        for _ in range(steps_per_update):
            action, log_prob, value = agent.select_action(obs)
            env_action = (
                np.clip(action, env.action_space.low, env.action_space.high)
                if continuous else int(action)
            )
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            agent.buffer.store(obs, action, reward, done, log_prob, value)
            obs = next_obs
            current_ep_reward += reward
            total_steps       += 1
            steps_ += 1
            if done:
                # ── Registra recompensa do episódio no logger ─────────────
                logger.log_episode(current_ep_reward)
                episode_count     += 1
                rewards_ = current_ep_reward
                current_ep_reward  = 0.0
                obs, _ = env.reset()

                if episode_count % log_interval == 0:
                    r_arr  = np.array(logger.episode_rewards)
                    mean_r = r_arr[-log_interval:].mean()
                    n_upd  = len(logger.kl_divs)
                    kl_str = f"KL={logger.kl_divs[-1]:.5f}" if logger.kl_divs else ""
                    ms_str = f"step={logger.time_total[-1]:.0f}ms" if logger.time_total else ""
                    print(
                        f"Ep {episode_count:4d} | Total steps {total_steps:7d} | "
                        f"Recomp. média: {mean_r:7.2f} | Updates: {n_upd:4d} | "
                        f"{kl_str}  {ms_str}"
                    )

                if episode_count > (n_episodes - 10) and a == 0:
                    a = 1
                    env.close()
                    env = gym.make(env_name, render_mode='human')
                    env.reset()
                    
                if a == 1:
                    print(f"Episode: {episode_count:4d} | Rewards: {rewards_:4.0f} | "
                          f"Steps: {steps_:4d}")
                    steps_ = 0
                    
        # ── Update TRPO (métricas registradas internamente) ─────────────────
        agent.update()

    env.close()

    # ── Gera painel de métricas ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Gerando painel de métricas...")
    plot_metrics(
        logger    = logger,
        env_name  = env_name,
        delta     = delta,
        save_path = metrics_path,
    )

    rews = np.array(logger.episode_rewards)
    print(f"Recompensa média (últimos 50 ep): {rews[-50:].mean():.2f}")
    print(f"Recompensa máxima               : {rews.max():.2f}")
    print(f"Step time médio                 : {np.mean(logger.time_total):.1f} ms")
    print(f"Taxa de aceitação (line search) : {np.mean(logger.ls_success)*100:.1f}%")
    return logger


# =============================================================================
# 9. EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    logger = train(
        # env_name         = "CartPole-v1",  # Troque por "Pendulum-v1" para contínuo
        env_name         = "Pendulum-v1",  # Troque por "Pendulum-v1" para contínuo
        # env_name         = "Acrobot-v1",  # Troque por "Pendulum-v1" para contínuo
        n_episodes       = 1000,
        steps_per_update = 2048,
        log_interval     = 20,
        seed             = 42,
        metrics_path     = "trpo_metrics.png",
    )