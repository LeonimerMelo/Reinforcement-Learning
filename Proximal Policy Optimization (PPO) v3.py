"""
=============================================================
  Proximal Policy Optimization (PPO) com PyTorch + CUDA
=============================================================

Algoritmo : PPO-Clip (Schulman et al., 2017)
            https://arxiv.org/abs/1707.06347

Ambientes suportados:
  • CartPole-v1    — discreto, fácil referência
  • Acrobot-v1     — discreto, controle de pêndulo duplo
  • Pendulum-v1    — contínuo  (ação ∈ [-2, 2])
  • Hopper-v4      — contínuo  (requer MuJoCo)

Conceitos cobertos:
  1.  Actor-Critic com rede compartilhada
  2.  Suporte a espaços de ação DISCRETO e CONTÍNUO
  3.  Rollout Buffer com coleta vetorizada
  4.  GAE — Generalized Advantage Estimation
  5.  PPO-Clip loss
  6.  Mini-batches + múltiplas épocas de atualização
  7.  CUDA automático
  8.  Gráficos de métricas com matplotlib

Instalação:
  pip install torch gymnasium matplotlib

  Para Hopper (MuJoCo):
  pip install gymnasium[mujoco]
  
# Instalar dependências
pip install torch gymnasium matplotlib

# Treinar (CartPole padrão)
python ppo_pytorch.py

# Outros ambientes
python ppo_pytorch.py --env Acrobot-v1
python ppo_pytorch.py --env Pendulum-v1
python ppo_pytorch.py --env Hopper-v4   # requer: pip install gymnasium[mujoco]

# Com avaliação visual ao final
python ppo_pytorch.py --env Pendulum-v1 --eval

# Reproduzibilidade
python ppo_pytorch.py --env CartPole-v1 --seed 0
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")          # renderiza sem display (salva em arquivo)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════
# 0.  CONFIGURAÇÃO POR AMBIENTE
#     Cada ambiente tem hiperparâmetros e critérios de sucesso
#     ligeiramente diferentes.
# ══════════════════════════════════════════════════════════════
ENV_CONFIGS = {
    "CartPole-v1": dict(
        total_timesteps = 300_000,
        n_steps         = 2048,
        n_envs          = 4,
        n_epochs        = 10,
        mini_batch_size = 64,
        clip_eps        = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        lr              = 3e-4,
        hidden          = 64,
        action_type     = "discrete",
        solve_threshold = 475.0,      # retorno médio para declarar "resolvido"
        solve_window    = 100,
    ),
    "Acrobot-v1": dict(
        total_timesteps = 500_000,
        n_steps         = 2048,
        n_envs          = 4,
        n_epochs        = 10,
        mini_batch_size = 64,
        clip_eps        = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        lr              = 3e-4,
        hidden          = 64,
        action_type     = "discrete",
        solve_threshold = -100.0,     # retorno médio ≥ -100 → resolvido
        solve_window    = 100,
    ),
    "Pendulum-v1": dict(
        total_timesteps = 500_000,
        n_steps         = 2048,
        n_envs          = 4,
        n_epochs        = 10,
        mini_batch_size = 64,
        clip_eps        = 0.2,
        ent_coef        = 0.0,        # Pendulum não precisa de entropia alta
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        lr              = 3e-4,
        hidden          = 256,        # rede maior para espaço contínuo
        action_type     = "continuous",
        solve_threshold = -200.0,     # retorno médio ≥ -200 → bom desempenho
        solve_window    = 100,
    ),
    "Hopper-v4": dict(
        total_timesteps = 1_000_000,
        n_steps         = 2048,
        n_envs          = 4,
        n_epochs        = 10,
        mini_batch_size = 64,
        clip_eps        = 0.2,
        ent_coef        = 0.0,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        lr              = 3e-4,
        hidden          = 256,
        action_type     = "continuous",
        solve_threshold = 2000.0,
        solve_window    = 100,
    ),
}


# ══════════════════════════════════════════════════════════════
# 1.  REDE ACTOR-CRITIC  (suporte discreto e contínuo)
# ══════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """
    Rede com tronco compartilhado e duas cabeças:
      • Actor  → política π(a|s)
      • Critic → valor V(s)

    Para ações DISCRETAS:
      - actor_head produz logits → distribuição Categorical

    Para ações CONTÍNUAS:
      - actor_mean  produz μ(s)  → média da Gaussiana
      - actor_log_std é parâmetro treinável global (independente do estado)
      - distribuição Normal(μ, σ) com clipping de log_std para estabilidade
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        action_type: str = "discrete",   # "discrete" ou "continuous"
    ):
        super().__init__()
        self.action_type = action_type

        # ── Tronco compartilhado ──────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # ── Cabeça do Critic ──────────────────────────────────────
        self.critic_head = nn.Linear(hidden, 1)

        # ── Cabeça do Actor ───────────────────────────────────────
        if action_type == "discrete":
            self.actor_head = nn.Linear(hidden, act_dim)

        elif action_type == "continuous":
            self.actor_mean    = nn.Linear(hidden, act_dim)
            # log_std como parâmetro independente do estado (mais estável)
            self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self._init_weights()

    def _init_weights(self):
        """Inicialização ortogonal recomendada para PPO."""
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

        if self.action_type == "discrete":
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.constant_(self.actor_head.bias, 0.0)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Retorna:
          dist  – distribuição (Categorical ou Normal)
          value – V(s), shape (B,)
        """
        feat  = self.shared(obs)
        value = self.critic_head(feat).squeeze(-1)

        if self.action_type == "discrete":
            dist = Categorical(logits=self.actor_head(feat))

        else:  # continuous
            mean = self.actor_mean(feat)
            # Clipa log_std para evitar explosão/colapso da distribuição
            log_std = self.actor_log_std.clamp(-20, 2)
            std  = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)

        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """
        Amostra ação da política atual sem rastrear gradientes.
        Para Normal multivariada, log_prob é a soma sobre dimensões.
        """
        dist, value = self.forward(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        if self.action_type == "continuous":
            log_prob = log_prob.sum(dim=-1)   # produto de independentes → soma de logs
        return action, log_prob, value


# ══════════════════════════════════════════════════════════════
# 2.  ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════
class RolloutBuffer:
    """
    Armazena N_STEPS × N_ENVS transições e calcula GAE ao final.

    GAE (Generalized Advantage Estimation):
      δ_t = r_t + γ·V(s_{t+1})·(1−d_t) − V(s_t)
      A_t = δ_t + (γ·λ)·(1−d_t)·A_{t+1}

    λ → 0 : alta viés, baixa variância (como TD)
    λ → 1 : baixa viés, alta variância (como Monte-Carlo)
    λ = 0.95 é o padrão PPO.
    """

    def __init__(
        self,
        n_steps:    int,
        n_envs:     int,
        obs_dim:    int,
        act_dim:    int,
        action_type: str,
        gamma:      float,
        gae_lambda: float,
        mini_batch_size: int,
    ):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.action_type = action_type
        T, E = n_steps, n_envs

        # Dtype de ação depende do tipo
        act_dtype = torch.long if action_type == "discrete" else torch.float32
        act_shape = (T, E) if action_type == "discrete" else (T, E, act_dim)

        self.obs       = torch.zeros(T, E, obs_dim)
        self.actions   = torch.zeros(act_shape, dtype=act_dtype)
        self.log_probs = torch.zeros(T, E)
        self.rewards   = torch.zeros(T, E)
        self.values    = torch.zeros(T, E)
        self.dones     = torch.zeros(T, E)
        self.ptr       = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor):
        advantages = torch.zeros_like(self.rewards)
        gae        = torch.zeros(self.n_envs)

        for t in reversed(range(self.n_steps)):
            nxt_val  = last_value      if t == self.n_steps - 1 else self.values[t + 1]
            nxt_done = last_done       if t == self.n_steps - 1 else self.dones[t + 1]
            delta    = self.rewards[t] + self.gamma * nxt_val * (1.0 - nxt_done) - self.values[t]
            gae      = delta + self.gamma * self.gae_lambda * (1.0 - nxt_done) * gae
            advantages[t] = gae

        self.returns    = advantages + self.values
        self.advantages = advantages

    def get_minibatches(self, device):
        """Achata buffer e gera mini-batches aleatórios."""
        B = self.n_steps * self.n_envs

        obs        = self.obs.view(B, -1).to(device)
        log_probs  = self.log_probs.view(B).to(device)
        returns    = self.returns.view(B).to(device)
        advantages = self.advantages.view(B).to(device)

        # Normalização das advantages por mini-batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.action_type == "discrete":
            actions = self.actions.view(B).to(device)
        else:
            actions = self.actions.view(B, -1).to(device)

        indices = torch.randperm(B)
        for start in range(0, B, self.mini_batch_size):
            idx = indices[start : start + self.mini_batch_size]
            yield obs[idx], actions[idx], log_probs[idx], returns[idx], advantages[idx]

    def reset(self):
        self.ptr = 0


# ══════════════════════════════════════════════════════════════
# 3.  AGENTE PPO
# ══════════════════════════════════════════════════════════════
class PPOAgent:
    """
    Encapsula rede, otimizador e lógica de atualização.

    Perda total:
      L = −L_CLIP + VF_COEF·L_VF − ENT_COEF·L_ENT

      L_CLIP = E[ min(r·A, clip(r, 1−ε, 1+ε)·A) ]
      L_VF   = MSE(V(s), retorno)
      L_ENT  = H[π(·|s)]
    """

    def __init__(self, obs_dim, act_dim, cfg: dict, device):
        self.cfg    = cfg
        self.device = device
        self.net    = ActorCritic(
            obs_dim, act_dim,
            hidden      = cfg["hidden"],
            action_type = cfg["action_type"],
        ).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=1e-5)

    def update(self, buffer: RolloutBuffer):
        cfg = self.cfg
        policy_losses, value_losses, entropy_losses, clip_fracs = [], [], [], []

        for _ in range(cfg["n_epochs"]):
            for obs, actions, old_log_probs, returns, advantages in buffer.get_minibatches(self.device):

                dist, values = self.net(obs)

                # log_prob da ação (soma sobre dims para contínuo)
                new_log_probs = dist.log_prob(actions)
                if cfg["action_type"] == "continuous":
                    new_log_probs = new_log_probs.sum(dim=-1)

                entropy = dist.entropy()
                if cfg["action_type"] == "continuous":
                    entropy = entropy.sum(dim=-1)
                entropy = entropy.mean()

                # ── Razão de importância ──────────────────────────
                ratio = (new_log_probs - old_log_probs).exp()

                # Fração de amostras que foram clipadas (diagnóstico)
                clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()
                clip_fracs.append(clip_frac.item())

                # ── PPO-Clip ──────────────────────────────────────
                surr1       = ratio * advantages
                surr2       = ratio.clamp(1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value Function ─────────────────────────────────
                value_loss  = nn.functional.mse_loss(values, returns)

                # ── Perda total ────────────────────────────────────
                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return (
            np.mean(policy_losses),
            np.mean(value_losses),
            np.mean(entropy_losses),
            np.mean(clip_fracs),
        )


# ══════════════════════════════════════════════════════════════
# 4.  MÉTRICAS  (coleta e plotting)
# ══════════════════════════════════════════════════════════════
class MetricsLogger:
    """
    Registra todas as métricas relevantes durante o treinamento e
    gera um painel de gráficos ao final (ou sob demanda).

    Métricas registradas por iteração:
      • mean_return       – média dos retornos (últimos N ep.)
      • min/max_return    – mínimo e máximo dos episódios recentes
      • policy_loss       – L_CLIP
      • value_loss        – L_VF
      • entropy           – H[π]
      • clip_fraction     – fração de razões clipadas (saúde do PPO)
      • steps_per_second  – throughput de coleta

    Exemplo de diagnóstico:
      clip_fraction → 0 : updates muito pequenos (LR alto demais?)
      clip_fraction → 1 : updates muito grandes (LR alto demais ou ε baixo?)
      entropy decrescente : política convergindo (esperado)
      value_loss crescente : critic não consegue acompanhar (problema)
    """

    def __init__(self):
        self.steps         = []
        self.mean_returns  = []
        self.min_returns   = []
        self.max_returns   = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropies     = []
        self.clip_fracs    = []
        self.sps_list      = []     # steps per second

    def log(self, step, recent_returns, pl, vl, ent, cf, sps):
        self.steps.append(step)
        if recent_returns:
            self.mean_returns.append(np.mean(recent_returns))
            self.min_returns.append(np.min(recent_returns))
            self.max_returns.append(np.max(recent_returns))
        else:
            self.mean_returns.append(np.nan)
            self.min_returns.append(np.nan)
            self.max_returns.append(np.nan)
        self.policy_losses.append(pl)
        self.value_losses.append(vl)
        self.entropies.append(ent)
        self.clip_fracs.append(cf)
        self.sps_list.append(sps)

    def _smooth(self, data, w=5):
        """Média móvel simples para suavizar curvas."""
        out = []
        q   = deque(maxlen=w)
        for v in data:
            if not np.isnan(v):
                q.append(v)
            out.append(np.mean(q) if q else np.nan)
        return np.array(out)

    def plot(self, env_name: str, save_path: str = "ppo_metrics.png"):
        """
        Gera painel 3×2 com as 6 métricas mais importantes:
          [0,0] Retorno médio (+ banda min/max + linha de sucesso)
          [0,1] Clip Fraction
          [1,0] Policy Loss (L_CLIP)
          [1,1] Value Loss (L_VF)
          [2,0] Entropia da política
          [2,1] Throughput (steps/segundo)
        """
        steps = np.array(self.steps)
        W = 7   # janela de suavização

        # ── Estética ──────────────────────────────────────────────
        BG      = "#0f1117"
        PANEL   = "#1a1d27"
        ACCENT  = "#4fc3f7"
        GREEN   = "#69f0ae"
        ORANGE  = "#ffa726"
        RED     = "#ef5350"
        GRAY    = "#546e7a"
        TEXT    = "#eceff1"
        GRID    = "#263238"

        plt.rcParams.update({
            "font.family"  : "monospace",
            "text.color"   : TEXT,
            "axes.facecolor"    : PANEL,
            "figure.facecolor"  : BG,
            "axes.edgecolor"    : GRID,
            "axes.labelcolor"   : TEXT,
            "xtick.color"       : GRAY,
            "ytick.color"       : GRAY,
            "grid.color"        : GRID,
            "grid.linestyle"    : "--",
            "grid.alpha"        : 0.5,
        })

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"PPO · {env_name} · métricas de treinamento",
            fontsize=16, fontweight="bold", color=TEXT, y=0.98,
        )

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)
        axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

        def style_ax(ax, title, xlabel, ylabel):
            ax.set_title(title, color=TEXT, fontsize=11, pad=8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True)

        # ── [0] Retorno médio ─────────────────────────────────────
        ax = axes[0]
        mean_r = np.array(self.mean_returns)
        min_r  = np.array(self.min_returns)
        max_r  = np.array(self.max_returns)
        smooth = self._smooth(mean_r, W)

        mask = ~np.isnan(mean_r)
        ax.fill_between(steps[mask], min_r[mask], max_r[mask],
                        alpha=0.15, color=ACCENT, label="min/max")
        ax.plot(steps[mask], mean_r[mask],
                color=ACCENT, alpha=0.3, linewidth=1)
        ax.plot(steps[mask], smooth[mask],
                color=GREEN, linewidth=2.2, label=f"média (smooth={W})")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        style_ax(ax, "Retorno por episódio", "Passos", "Retorno")

        # ── [1] Clip Fraction ─────────────────────────────────────
        ax = axes[1]
        cf = np.array(self.clip_fracs)
        ax.plot(steps, cf, color=ORANGE, alpha=0.4, linewidth=1)
        ax.plot(steps, self._smooth(cf, W), color=ORANGE, linewidth=2.2)
        ax.axhline(0.1, color=GRAY, linestyle=":", linewidth=1, label="ref 10%")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        style_ax(ax, "Clip Fraction  (saúde do clipping)", "Passos", "Fração clipada")

        # ── [2] Policy Loss ───────────────────────────────────────
        ax = axes[2]
        pl = np.array(self.policy_losses)
        ax.plot(steps, pl, color=RED, alpha=0.3, linewidth=1)
        ax.plot(steps, self._smooth(pl, W), color=RED, linewidth=2.2)
        style_ax(ax, "Policy Loss  (L_CLIP)", "Passos", "Loss")

        # ── [3] Value Loss ────────────────────────────────────────
        ax = axes[3]
        vl = np.array(self.value_losses)
        ax.plot(steps, vl, color="#ba68c8", alpha=0.3, linewidth=1)
        ax.plot(steps, self._smooth(vl, W), color="#ba68c8", linewidth=2.2)
        style_ax(ax, "Value Loss  (L_VF = MSE)", "Passos", "Loss")

        # ── [4] Entropia ──────────────────────────────────────────
        ax = axes[4]
        en = np.array(self.entropies)
        ax.plot(steps, en, color="#4dd0e1", alpha=0.3, linewidth=1)
        ax.plot(steps, self._smooth(en, W), color="#4dd0e1", linewidth=2.2)
        style_ax(ax, "Entropia  H[π]  (exploração)", "Passos", "Entropia (nats)")

        # ── [5] Throughput ────────────────────────────────────────
        ax = axes[5]
        sp = np.array(self.sps_list)
        ax.plot(steps, sp, color="#aed581", alpha=0.4, linewidth=1)
        ax.plot(steps, self._smooth(sp, W), color="#aed581", linewidth=2.2)
        style_ax(ax, "Throughput", "Passos", "steps / segundo")

        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        # plt.show()
        plt.close()
        print(f"\n📊  Gráfico salvo em: {os.path.abspath(save_path)}")


# ══════════════════════════════════════════════════════════════
# 5.  LOOP DE TREINAMENTO
# ══════════════════════════════════════════════════════════════
def train(env_name: str, seed: int = 42):
    cfg    = ENV_CONFIGS[env_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═'*60}")
    print(f"  PPO  ·  {env_name}  ·  device={device}")
    print(f"  action_type={cfg['action_type']}  "
          f"envs={cfg['n_envs']}  steps={cfg['n_steps']}")
    print(f"{'═'*60}\n")

    # ── Ambientes vetorizados ─────────────────────────────────
    # SyncVectorEnv roda ambientes em série no mesmo processo.
    # Para alta performance use AsyncVectorEnv.
    def make_env(seed_offset=0):
        def _f():
            e = gym.make(env_name)
            e.reset(seed=seed + seed_offset)
            return e
        return _f

    envs = gym.vector.SyncVectorEnv(
        [make_env(i) for i in range(cfg["n_envs"])]
    )

    obs_dim = envs.single_observation_space.shape[0]

    # act_dim: número de ações (discreto) ou dimensão da ação (contínuo)
    if cfg["action_type"] == "discrete":
        act_dim = envs.single_action_space.n
    else:
        act_dim = envs.single_action_space.shape[0]

    agent  = PPOAgent(obs_dim, act_dim, cfg, device)
    buffer = RolloutBuffer(
        n_steps      = cfg["n_steps"],
        n_envs       = cfg["n_envs"],
        obs_dim      = obs_dim,
        act_dim      = act_dim,
        action_type  = cfg["action_type"],
        gamma        = cfg["gamma"],
        gae_lambda   = cfg["gae_lambda"],
        mini_batch_size = cfg["mini_batch_size"],
    )

    logger  = MetricsLogger()
    win_buf = deque(maxlen=cfg["solve_window"])  # retornos recentes

    obs_np, _ = envs.reset(seed=seed)
    obs  = torch.tensor(obs_np, dtype=torch.float32)
    done = torch.zeros(cfg["n_envs"])

    total_steps    = 0
    iteration      = 0
    ep_return      = np.zeros(cfg["n_envs"])
    t_start        = time.time()
    solved         = False
    while total_steps < cfg["total_timesteps"]:

        # ── 5.1  Rollout ──────────────────────────────────────
        buffer.reset()
        t_rollout = time.time()

        for _ in range(cfg["n_steps"]):
            obs_gpu            = obs.to(device)
            action, lp, value  = agent.net.act(obs_gpu)
            action_np          = action.cpu().numpy()

            # Para ambientes contínuos precisa clipar a ação ao espaço válido
            if cfg["action_type"] == "continuous":
                act_space  = envs.single_action_space
                action_np  = np.clip(action_np, act_space.low, act_space.high)

            next_obs_np, rew_np, term_np, trunc_np, _ = envs.step(action_np)
            done_np = np.logical_or(term_np, trunc_np).astype(np.float32)
            ep_return += rew_np

            # Ação para buffer: contínua mantém shape (E, act_dim)
            if cfg["action_type"] == "continuous":
                act_buf = torch.tensor(action_np, dtype=torch.float32)
            else:
                act_buf = action.cpu()

            buffer.add(
                obs      = obs.cpu(),
                action   = act_buf,
                log_prob = lp.cpu(),
                reward   = torch.tensor(rew_np,  dtype=torch.float32),
                value    = value.cpu(),
                done     = torch.tensor(done_np, dtype=torch.float32),
            )

            for i, d in enumerate(done_np):
                if d:
                    win_buf.append(ep_return[i])
                    ep_return[i] = 0.0

            obs  = torch.tensor(next_obs_np, dtype=torch.float32)
            done = torch.tensor(done_np,     dtype=torch.float32)
            total_steps += cfg["n_envs"]

        # ── 5.2  GAE ──────────────────────────────────────────
        with torch.no_grad():
            _, last_val = agent.net(obs.to(device))
        buffer.compute_gae(last_val.cpu(), done)

        # ── 5.3  Atualização PPO ───────────────────────────────
        pl, vl, ent, cf = agent.update(buffer)

        # ── 5.4  Métricas e log ───────────────────────────────
        elapsed = time.time() - t_rollout
        sps     = (cfg["n_steps"] * cfg["n_envs"]) / max(elapsed, 1e-6)

        logger.log(
            step            = total_steps,
            recent_returns  = list(win_buf),
            pl=pl, vl=vl, ent=ent, cf=cf, sps=sps,
        )

        iteration += 1
        mean_ret  = np.mean(win_buf) if win_buf else float("nan")

        print(
            f"iter={iteration:4d} | steps={total_steps:8d} | "
            f"ret={mean_ret:8.2f} | "
            f"π={pl:+.4f} | V={vl:.4f} | "
            f"H={ent:.3f} | clip={cf:.3f} | "
            f"sps={sps:.0f}"
        )

        # ── Critério de sucesso ────────────────────────────────
        if (
            len(win_buf) >= cfg["solve_window"]
            and mean_ret >= cfg["solve_threshold"]
            and not solved
        ):
            solved = True
            total_t = time.time() - t_start
            print(f"\n✅  Ambiente resolvido em {total_steps:,} passos "
                  f"({total_t:.1f}s)!\n")

    envs.close()

    total_t = time.time() - t_start
    print(f"\n🏁  Treinamento concluído: {total_steps:,} passos em {total_t:.1f}s")

    # ── Gera gráficos ─────────────────────────────────────────
    fname = f"ppo_{env_name.replace('-', '_').lower()}_metrics.png"
    logger.plot(env_name, save_path=fname)

    return agent, logger


# ══════════════════════════════════════════════════════════════
# 6.  AVALIAÇÃO VISUAL (opcional)
# ══════════════════════════════════════════════════════════════
def evaluate(agent: PPOAgent, env_name: str, cfg: dict, n_episodes: int = 5):
    """Roda episódios com renderização visual. Requer pygame."""
    print(f"\n── Avaliação visual  ({n_episodes} episódios) ──")
    env = gym.make(env_name, render_mode="human")
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action, _, _ = agent.net.act(obs_t)
            if cfg["action_type"] == "continuous":
                act_np = action.squeeze(0).cpu().numpy()
                act_np = np.clip(act_np, env.action_space.low, env.action_space.high)
            else:
                act_np = action.item()
            obs, reward, term, trunc, _ = env.step(act_np)
            total += reward
            done   = term or trunc
        returns.append(total)
        print(f"  Episódio {ep+1}: retorno = {total:.2f}")

    env.close()
    print(f"  Média: {np.mean(returns):.2f}  ±  {np.std(returns):.2f}")


# ══════════════════════════════════════════════════════════════
# 7.  MAIN  (argparse para escolha de ambiente)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO com PyTorch + CUDA — escolha o ambiente"
    )
    parser.add_argument(
        "--env",
        type    = str,
        # default = "CartPole-v1",
        default = "Pendulum-v1",
        # default = "Acrobot-v1",
        # default = "Hopper-v4",
        choices = list(ENV_CONFIGS.keys()),
        help    = "Ambiente Gymnasium para treinar  (default: CartPole-v1)",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Semente aleatória  (default: 42)",
    )
    parser.add_argument(
        "--eval",
        action  = "store_true",
        help    = "Renderiza episódios após o treino (requer pygame/MuJoCo viewer)",
    )
    args = parser.parse_args()

    print(f"Ambientes disponíveis: {list(ENV_CONFIGS.keys())}")
    print(f"Ambiente selecionado : {args.env}")
    print(f"Semente              : {args.seed}")

    trained_agent, metrics = train(args.env, seed=args.seed)

    if args.eval:
        try:
            evaluate(trained_agent, args.env, ENV_CONFIGS[args.env])
        except Exception as exc:
            print(f"[AVISO] Render indisponível: {exc}")
