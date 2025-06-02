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
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from networks import ResNetExtractor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# ── Configuración general ────────────────────────────────────────
gym.register_envs(ale_py)
ENV_ID              = "EnduroNoFrameskip-v4"
TOTAL_TIMESTEPS     = 10_000   # 10M pasos
N_ENVS              = 10
SEED                = 42
FRAME_STACK         = 4
SAVE_FREQ           = 100
EVAL_FREQ           = 100
N_EVAL_EPISODES     = 5
FINAL_VIDEO_LENGTH  = 2_500

DIR_IMAGES = ENV_ID + "-images"
DIR_MODELS = ENV_ID + "-models"
DIR_LOGS   = os.path.join(ENV_ID + "-logs", "comparativa")
VIDEO_DIR  = "videos"
for d in (DIR_IMAGES, DIR_MODELS, DIR_LOGS, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)

# ── CUDA optimizaciones ───────────────────────────────────────────
#torch.backends.cudnn.benchmark = True
#torch.cuda.empty_cache()
#torch.cuda.set_per_process_memory_fraction(0.95)

# ── Funciones auxiliares ────────────────────────────────────────────
def make_env(n_envs=1):
    return VecFrameStack(
        make_atari_env(ENV_ID, n_envs=n_envs, seed=SEED),
        n_stack=FRAME_STACK
    )

def get_architecture_summary(model):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        summary(model.policy, input_size=(4,84,84), device="cuda")
    finally:
        sys.stdout = old
    return buf.getvalue()

def visualize_and_save_feature_maps(tensor, title, out_dir):
    if tensor.dim() == 4:
        tensor = tensor[0]
    os.makedirs(out_dir, exist_ok=True)
    n_maps = tensor.shape[0]
    cols = 8
    rows = (n_maps + cols - 1) // cols
    fig = plt.figure(figsize=(15, rows * 2))
    for i in range(n_maps):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(tensor[i].detach().cpu().numpy(), cmap='viridis')
        ax.axis("off")
        ax.set_title(f"Map {i}")
        # Guardar cada feature map
        plt.imsave(os.path.join(out_dir, f"map_{i}.png"), tensor[i].detach().cpu().numpy(), cmap='viridis')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ── Entrenamiento y evaluación ──────────────────────────────────────
if __name__ == "__main__":
    # --- 1) DQN run ---------------------------------------------------
    log_dir_dqn = os.path.join(DIR_LOGS, "cnn_run")
    os.makedirs(log_dir_dqn, exist_ok=True)
    logger_dqn  = configure(log_dir_dqn, ["tensorboard"])

    env_dqn     = make_env(N_ENVS)
    cb_dqn      = CheckpointCallback(SAVE_FREQ, DIR_MODELS, name_prefix="cnn_atari")
    eval_cb_dqn = EvalCallback(
        VecTransposeImage(make_env(1)),
        best_model_save_path=DIR_MODELS,
        log_path=log_dir_dqn,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    model_dqn = DQN(
        policy="CnnPolicy",
        env=env_dqn,
        tensorboard_log=log_dir_dqn,
        learning_rate=2.5e-4,
        buffer_size=50000,
        learning_starts=10_000,
        batch_size=64,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=10_000,
        exploration_fraction=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        policy_kwargs={"net_arch": [512]},
        verbose=0,
        seed=SEED
    )
    model_dqn.set_logger(logger_dqn)

    t0 = time.time()
    model_dqn.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[cb_dqn, eval_cb_dqn],
        tb_log_name="cnn_run",
        reset_num_timesteps=True,
        progress_bar=True
    )
    dur_dqn = time.time() - t0
    model_dqn.save(os.path.join(DIR_MODELS, "dqn_final"))
    env_dqn.close()
    model_dqn.logger.record("Training_Time/DQN", dur_dqn)
    model_dqn.logger.record("Arch/DQN", get_architecture_summary(model_dqn))
    model_dqn.logger.dump(TOTAL_TIMESTEPS)
    torch.cuda.empty_cache()

    # --- 2) ResNet run ------------------------------------------------
    log_dir_res = os.path.join(DIR_LOGS, "resnet_run")
    os.makedirs(log_dir_res, exist_ok=True)
    logger_res  = configure(log_dir_res, ["tensorboard"])

    env_res     = make_env(N_ENVS)
    cb_res      = CheckpointCallback(SAVE_FREQ, DIR_MODELS, name_prefix="resnet_atari")
    eval_cb_res = EvalCallback(
        VecTransposeImage(make_env(1)),
        best_model_save_path=DIR_MODELS,
        log_path=log_dir_res,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    policy_kwargs = {
        "features_extractor_class": ResNetExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [512]
        }

    model_res = DQN(
        policy="CnnPolicy",
        env=env_res,
        tensorboard_log=log_dir_res,
        learning_rate=2.5e-4,
        buffer_size=50000,
        learning_starts=10_000,
        batch_size=64,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=10_000,
        exploration_fraction=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        policy_kwargs=policy_kwargs,
        device="cuda",
        verbose=0,
        seed=SEED
    )
    model_res.set_logger(logger_res)

    t0 = time.time()
    model_res.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[cb_res, eval_cb_res],
        tb_log_name="resnet_run",
        reset_num_timesteps=True,
        progress_bar=True
    )
    dur_res = time.time() - t0
    model_res.save(os.path.join(DIR_MODELS, "resnet_final"))
    env_res.close()
    model_res.logger.record("Training_Time/ResNet", dur_res)
    model_res.logger.record("Arch/ResNet", get_architecture_summary(model_res))
    model_res.logger.dump(TOTAL_TIMESTEPS)
    torch.cuda.empty_cache()

    # --- Gráfica comparativa de tiempos -------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    bars = ax.bar(["CNN", "ResNet"], [dur_dqn, dur_res])
    ax.bar_label(bars, labels=[f"{dur_dqn:.1f}s", f"{dur_res:.1f}s"], padding=3) # Tiempo en seg

    # Guardar grafica
    outfile = os.path.join(DIR_IMAGES, 'Comparativa'+ENV_ID+'.png')
    fig.savefig(outfile, dpi=300, bbox_inches="tight")

    SummaryWriter(log_dir=DIR_LOGS).add_figure("Comparativa", fig, TOTAL_TIMESTEPS)
    plt.close(fig)

    print(f"✅ Logs TB en: {DIR_LOGS}")
    print("Ejecuta: tensorboard --logdir", DIR_LOGS)
    print("Verás dos runs: dqn_run y resnet_run en la interfaz.")

    # --- Videos finales --------------------------------
    for f in glob.glob(os.path.join(VIDEO_DIR, "dqn_final*.mp4")):
        os.remove(f)
    env_vid = VecFrameStack(make_atari_env(ENV_ID, n_envs=1, seed=SEED), FRAME_STACK)
    vid_rec = VecVideoRecorder(
        env_vid, VIDEO_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=FINAL_VIDEO_LENGTH,
        name_prefix="dqn_final"
    )
    obs, total_reward = vid_rec.reset(), 0
    for _ in range(FINAL_VIDEO_LENGTH):
        action, _ = model_dqn.predict(obs, deterministic=True)
        obs, rew, done, _ = vid_rec.step(action)
        total_reward += rew[0]
        if done[0]:
            obs = vid_rec.reset()
    vid_rec.close()
    print(f"🏆 Video DQN: {glob.glob(os.path.join(VIDEO_DIR, 'dqn_final*.mp4'))[-1]}  Rew={total_reward:.1f}")

    for f in glob.glob(os.path.join(VIDEO_DIR, "resnet_final*.mp4")):
        os.remove(f)
    env_vid = VecFrameStack(make_atari_env(ENV_ID, n_envs=1, seed=SEED), FRAME_STACK)
    vid_rec = VecVideoRecorder(
        env_vid, VIDEO_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=FINAL_VIDEO_LENGTH,
        name_prefix="resnet_final"
    )
    obs, total_reward = vid_rec.reset(), 0
    for _ in range(FINAL_VIDEO_LENGTH):
        action, _ = model_res.predict(obs, deterministic=True)
        obs, rew, done, _ = vid_rec.step(action)
        total_reward += rew[0]
        if done[0]:
            obs = vid_rec.reset()
    vid_rec.close()
    print(f"🏆 Video ResNet: {glob.glob(os.path.join(VIDEO_DIR, 'resnet_final*.mp4'))[-1]}  Rew={total_reward:.1f}")

    try:
        if os.name == 'nt':
            os.startfile(os.path.abspath(VIDEO_DIR))
    except:
        pass

    print("\n✅ Proceso completado.")
