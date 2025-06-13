import os
import io
import sys
import time
import glob
import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from networks import ImpalaResNetExtractor, ResNetLSTMExtractor, ImpalaResNetSimpleExtractor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.logger import configure

# ── Configuración general ────────────────────────────────────────
gym.register_envs(ale_py)
ENV_ID             = "BreakoutNoFrameskip-v4"
TOTAL_TIMESTEPS    = 10_000
N_ENVS             = 10
SEED               = 0
FRAME_STACK        = 4
SAVE_FREQ          = 100
N_EVAL_EPISODES    = 50
eval_seeds         = [42, 123, 456, 789, 1024]
FINAL_VIDEO_LENGTH = 2000

DIR_IMAGES = ENV_ID + "-images"
DIR_MODELS = ENV_ID + "-models"
DIR_LOGS   = os.path.join(ENV_ID + "-logs", "comparativa_4models")
VIDEO_DIR  = "videos"
for d in (DIR_IMAGES, DIR_MODELS, DIR_LOGS, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)

# ── CUDA optimizaciones ───────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.95)

# ── Auxiliares ─────────────────────────────────────────────────────
def make_env(n_envs=1, seed=SEED):
    return VecFrameStack(make_atari_env(ENV_ID, n_envs=n_envs, seed=seed), n_stack=FRAME_STACK)

def get_architecture_summary(model):
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        summary(model.policy, input_size=(4,84,84), device="cuda")
    finally:
        sys.stdout = old
    return buf.getvalue()

def evaluate_model_multi_seed(model, seeds, n_eps):
    all_rewards, all_lengths, seed_means = [], [], []
    for seed in seeds:
        rewards, lengths = [], []
        env_eval = VecTransposeImage(make_env(1, seed=seed))
        for _ in range(n_eps):
            obs = env_eval.reset()
            done = False
            R, L = 0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, done, _ = env_eval.step(action)
                R += rew[0]; L += 1
            rewards.append(R)
            lengths.append(L)
        env_eval.close()
        all_rewards += rewards
        all_lengths += lengths
        seed_means.append(np.mean(rewards))
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward' : np.std(all_rewards),
        'mean_length': np.mean(all_lengths),
        'std_length' : np.std(all_lengths),
        'seed_std'   : np.std(seed_means)
    }

# ── Entrenamiento y evaluación ─────────────────────────────────────
def train_and_eval(name, policy_kwargs=None):
    log_dir = os.path.join(DIR_LOGS, f"{name.lower()}_run")
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["tensorboard"])
    env = make_env(N_ENVS)
    model = DQN(
        policy="CnnPolicy",
        env=env,
        tensorboard_log=log_dir,
        learning_rate=2.5e-4,
        buffer_size=70_000,
        learning_starts=10_000,
        batch_size=128,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=10_000,
        exploration_fraction=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        policy_kwargs=policy_kwargs or {"net_arch":[512]},
        seed=SEED,
        device="cuda",
        verbose=0
    )
    model.set_logger(logger)
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=f"{name.lower()}_run",
        reset_num_timesteps=True,
        progress_bar=True
    )
    dur = time.time() - t0
    model.save(os.path.join(DIR_MODELS, f"{name.lower()}_final"))
    stats = evaluate_model_multi_seed(model, eval_seeds, N_EVAL_EPISODES)
    # Registrar en TB
    model.logger.record("Eval/mean_reward", stats['mean_reward'])
    model.logger.record("Eval/std_reward", stats['std_reward'])
    model.logger.record("Eval/mean_length", stats['mean_length'])
    model.logger.record("Eval/std_length", stats['std_length'])
    model.logger.record("Eval/seed_std", stats['seed_std'])
    model.logger.record("Training_Time", dur)
    model.logger.record("Arch_Summary", get_architecture_summary(model))
    model.logger.dump(TOTAL_TIMESTEPS)
    torch.cuda.empty_cache()
    return model, stats, dur

if __name__ == "__main__":
    # Config modelos
    configs = {
        "cnn_standard": None,
        "impalaresnet": {
            "features_extractor_class": ImpalaResNetExtractor,
            "features_extractor_kwargs": {"features_dim":256}
        },
        "resnetlstm": {
            "features_extractor_class": ResNetLSTMExtractor,
            "features_extractor_kwargs": {"features_dim":512}
        },
        "impalaresnetsimple": {
            "features_extractor_class": ImpalaResNetSimpleExtractor,
            "features_extractor_kwargs": {"features_dim":256}
        }
    }
    results = {}
    models = {}
    for name, pk in configs.items():
        print(f"\n>> Entrenando {name}")
        model, stats, dur = train_and_eval(name, pk)
        models[name] = model
        results[name] = {**stats, 'duration': dur}

    # Comparación rápida
    print("\n===== Resumen Final =====")
    for k, v in results.items():
        print(f"{k}: Reward={v['mean_reward']:.2f}±{v['std_reward']:.2f}, Time={v['duration']:.1f}s")

    # Gráficos comparativos
    sns.set_theme(style="darkgrid")
    names = list(results.keys())
    rewards = [results[n]['mean_reward'] for n in names]
    errs    = [results[n]['std_reward']   for n in names]
    times   = [results[n]['duration']     for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax[0].bar(names, times)
    ax[0].bar_label(bars, [f"{t:.1f}s" for t in times], padding=3)
    ax[0].set_title("Training Time")
    bars2= ax[1].bar(names, rewards, yerr=errs, capsize=5)
    ax[1].bar_label(bars2, [f"{r:.1f}" for r in rewards], padding=3)
    ax[1].set_title("Mean Reward")
    plt.tight_layout()
    outp = os.path.join(DIR_IMAGES, f"Comparison_{ENV_ID}.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    # Añadir TB
    writer = SummaryWriter(log_dir=DIR_LOGS)
    writer.add_figure("Comparison", fig, TOTAL_TIMESTEPS)
    writer.close()
    plt.close(fig)

    # Vídeos finales
    print("\n🎬 Generando vídeos...")
    for name, mdl in models.items():
        for f in glob.glob(os.path.join(VIDEO_DIR, f"{name}*.mp4")): os.remove(f)
        env_v = VecFrameStack(make_atari_env(ENV_ID, n_envs=1, seed=SEED), n_stack=FRAME_STACK)
        rec = VecVideoRecorder(
            env_v, VIDEO_DIR,
            record_video_trigger=lambda x: x == 0,
            video_length=FINAL_VIDEO_LENGTH,
            name_prefix=name
        )
        obs, total = rec.reset(), 0
        for _ in range(FINAL_VIDEO_LENGTH):
            act, _ = mdl.predict(obs, deterministic=True)
            obs, rew, done, _ = rec.step(act)
            total += rew[0]
            if done[0]: obs = rec.reset()
        rec.close()
        file = glob.glob(os.path.join(VIDEO_DIR, f"{name}*.mp4"))[-1]
        print(f"🏆 Video {name}: {file} Reward={total:.1f}")

    print("\n✅ Proceso completo. Ejecuta: tensorboard --logdir", DIR_LOGS)
