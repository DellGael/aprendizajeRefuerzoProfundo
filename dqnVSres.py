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
from networks import ImpalaResNetExtractor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# ── Configuración general ────────────────────────────────────────
gym.register_envs(ale_py)
ENV_ID              = "BreakoutNoFrameskip-v4"
TOTAL_TIMESTEPS     = 400000 # 10M pasos
N_ENVS              = 10
SEED                = 0
FRAME_STACK         = 4
SAVE_FREQ           = 100
N_EVAL_EPISODES     = 100  # Episodios por semilla
N_EVAL_SEEDS        = 5   # Número de semillas para evaluación
EVAL_SEEDS          = [42, 123, 456, 789, 1024]  # Semillas específicas
FINAL_VIDEO_LENGTH  = 5000

DIR_IMAGES = ENV_ID + "-images"
DIR_MODELS = ENV_ID + "-models"
DIR_LOGS   = os.path.join(ENV_ID + "-logs", "comparativa")
VIDEO_DIR  = "videos"
for d in (DIR_IMAGES, DIR_MODELS, DIR_LOGS, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)

# ── CUDA optimizaciones ───────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.95)

# ── Funciones auxiliares ────────────────────────────────────────────
def make_env(n_envs=1, seed=SEED):
    return VecFrameStack(
        make_atari_env(ENV_ID, n_envs=n_envs, seed=seed),
        n_stack=FRAME_STACK
    )

def get_architecture_summary(model):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        summary(model.policy, input_size=(4,84,84), device="cuda")
    finally:
        sys.stdout = old
    return buf.getvalue()

def evaluate_model_single_seed(model, seed, n_episodes=10, deterministic=True):
    """Evaluate a trained model with a single seed and return statistics"""
    eval_env = VecTransposeImage(make_env(1, seed=seed))
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                break
    
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    eval_env.close()
    
    return {
        'seed': seed,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }

def evaluate_model_multi_seed(model, seeds=EVAL_SEEDS, n_episodes=10, deterministic=True):
    """Evaluate a trained model across multiple seeds and return comprehensive statistics"""
    print(f"  Evaluating across {len(seeds)} seeds with {n_episodes} episodes each...")
    
    seed_results = []
    all_rewards = []
    all_lengths = []
    
    for i, seed in enumerate(seeds):
        print(f"    Seed {i+1}/{len(seeds)} (seed={seed}):")
        seed_result = evaluate_model_single_seed(model, seed, n_episodes, deterministic)
        seed_results.append(seed_result)
        
        # Collect all rewards and lengths
        all_rewards.extend(seed_result['episode_rewards'])
        all_lengths.extend(seed_result['episode_lengths'])
        
        print(f"      Mean Reward: {seed_result['mean_reward']:.2f} ± {seed_result['std_reward']:.2f}")
        print(f"      Mean Length: {seed_result['mean_length']:.1f} ± {seed_result['std_length']:.1f}")
    
    # Calculate overall statistics
    overall_mean_reward = np.mean(all_rewards)
    overall_std_reward = np.std(all_rewards)
    overall_mean_length = np.mean(all_lengths)
    overall_std_length = np.std(all_lengths)
    
    # Calculate seed-wise means (for variance across seeds)
    seed_mean_rewards = [result['mean_reward'] for result in seed_results]
    seed_mean_lengths = [result['mean_length'] for result in seed_results]
    
    return {
        'seed_results': seed_results,
        'all_rewards': all_rewards,
        'all_lengths': all_lengths,
        'overall_mean_reward': overall_mean_reward,
        'overall_std_reward': overall_std_reward,
        'overall_mean_length': overall_mean_length,
        'overall_std_length': overall_std_length,
        'seed_mean_rewards': seed_mean_rewards,
        'seed_std_rewards': np.std(seed_mean_rewards),
        'seed_mean_lengths': seed_mean_lengths,
        'seed_std_lengths': np.std(seed_mean_lengths),
        'total_episodes': len(all_rewards)
    }

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
    # --- 1) DQN Training ---------------------------------------------------
    print("🚀 Starting CNN DQN Training...")
    log_dir_dqn = os.path.join(DIR_LOGS, "cnn_run")
    os.makedirs(log_dir_dqn, exist_ok=True)
    logger_dqn  = configure(log_dir_dqn, ["tensorboard"])

    env_dqn     = make_env(N_ENVS)
    cb_dqn      = CheckpointCallback(SAVE_FREQ, DIR_MODELS, name_prefix="cnn_atari")

    model_dqn = DQN(
        policy="CnnPolicy",
        env=env_dqn,
        tensorboard_log=log_dir_dqn,
        learning_rate=2.5e-4,
        buffer_size=70000,
        learning_starts=10_000,
        batch_size=128,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=10_000,
        exploration_fraction=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        policy_kwargs={"net_arch": [512]},
        verbose=0,
        seed=SEED,
        device="cuda"
    )
    model_dqn.set_logger(logger_dqn)

    t0 = time.time()
    model_dqn.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="cnn_run",
        reset_num_timesteps=True,
        progress_bar=True
    )
    dur_dqn = time.time() - t0
    model_dqn.save(os.path.join(DIR_MODELS, "dqn_final"))
    env_dqn.close()
    print(f"✅ CNN DQN Training completed in {dur_dqn:.1f}s")

    # --- 1.1) CNN DQN Multi-Seed Evaluation --------------------------------
    print("📊 Evaluating CNN DQN across multiple seeds...")
    eval_stats_dqn = evaluate_model_multi_seed(model_dqn, EVAL_SEEDS, N_EVAL_EPISODES)
    
    # Log evaluation results
    model_dqn.logger.record("Eval/overall_mean_reward", eval_stats_dqn['overall_mean_reward'])
    model_dqn.logger.record("Eval/overall_std_reward", eval_stats_dqn['overall_std_reward'])
    model_dqn.logger.record("Eval/overall_mean_length", eval_stats_dqn['overall_mean_length'])
    model_dqn.logger.record("Eval/seed_std_rewards", eval_stats_dqn['seed_std_rewards'])
    model_dqn.logger.record("Eval/seed_std_lengths", eval_stats_dqn['seed_std_lengths'])
    model_dqn.logger.record("Eval/total_episodes", eval_stats_dqn['total_episodes'])
    model_dqn.logger.record("Training_Time/DQN", dur_dqn)
    model_dqn.logger.record("Arch/DQN", get_architecture_summary(model_dqn))
    model_dqn.logger.dump(TOTAL_TIMESTEPS)
    
    print(f"  CNN DQN - Overall Mean Reward: {eval_stats_dqn['overall_mean_reward']:.2f} ± {eval_stats_dqn['overall_std_reward']:.2f}")
    print(f"  CNN DQN - Overall Mean Length: {eval_stats_dqn['overall_mean_length']:.1f} ± {eval_stats_dqn['overall_std_length']:.1f}")
    print(f"  CNN DQN - Seed Variability (Reward): {eval_stats_dqn['seed_std_rewards']:.2f}")
    print(f"  CNN DQN - Seed Variability (Length): {eval_stats_dqn['seed_std_lengths']:.1f}")
    
    torch.cuda.empty_cache()

    # --- 2) ResNet Training ------------------------------------------------
    print("\n🚀 Starting ResNet DQN Training...")
    log_dir_res = os.path.join(DIR_LOGS, "resnet_run")
    os.makedirs(log_dir_res, exist_ok=True)
    logger_res  = configure(log_dir_res, ["tensorboard"])

    env_res     = make_env(N_ENVS)
    cb_res      = CheckpointCallback(SAVE_FREQ, DIR_MODELS, name_prefix="resnet_atari")

    # Configuración de policy_kwargs mejorada
    policy_kwargs = {
            "features_extractor_class": ImpalaResNetExtractor,
            "features_extractor_kwargs": {"features_dim": 256}
            }


    model_res = DQN(
        policy="CnnPolicy",
        env=env_res,
        tensorboard_log=log_dir_res,
        learning_rate=2.5e-4,
        buffer_size=70000,
        learning_starts=10_000,
        batch_size=128,
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
        tb_log_name="resnet_run",
        reset_num_timesteps=True,
        progress_bar=True
    )
    dur_res = time.time() - t0
    model_res.save(os.path.join(DIR_MODELS, "resnet_final"))
    env_res.close()
    print(f"✅ ResNet DQN Training completed in {dur_res:.1f}s")

    # --- 2.1) ResNet DQN Multi-Seed Evaluation -----------------------------
    print("📊 Evaluating ResNet DQN across multiple seeds...")
    eval_stats_res = evaluate_model_multi_seed(model_res, EVAL_SEEDS, N_EVAL_EPISODES)
    
    # Log evaluation results
    model_res.logger.record("Eval/overall_mean_reward", eval_stats_res['overall_mean_reward'])
    model_res.logger.record("Eval/overall_std_reward", eval_stats_res['overall_std_reward'])
    model_res.logger.record("Eval/overall_mean_length", eval_stats_res['overall_mean_length'])
    model_res.logger.record("Eval/seed_std_rewards", eval_stats_res['seed_std_rewards'])
    model_res.logger.record("Eval/seed_std_lengths", eval_stats_res['seed_std_lengths'])
    model_res.logger.record("Eval/total_episodes", eval_stats_res['total_episodes'])
    model_res.logger.record("Training_Time/ResNet", dur_res)
    model_res.logger.record("Arch/ResNet", get_architecture_summary(model_res))
    model_res.logger.dump(TOTAL_TIMESTEPS)
    
    print(f"  ResNet DQN - Overall Mean Reward: {eval_stats_res['overall_mean_reward']:.2f} ± {eval_stats_res['overall_std_reward']:.2f}")
    print(f"  ResNet DQN - Overall Mean Length: {eval_stats_res['overall_mean_length']:.1f} ± {eval_stats_res['overall_std_length']:.1f}")
    print(f"  ResNet DQN - Seed Variability (Reward): {eval_stats_res['seed_std_rewards']:.2f}")
    print(f"  ResNet DQN - Seed Variability (Length): {eval_stats_res['seed_std_lengths']:.1f}")
    
    torch.cuda.empty_cache()

    # --- 3) Enhanced Comparison Graphics -----------------------------------
    print("\n📈 Creating enhanced comparison graphics...")
    sns.set_theme(style="darkgrid")
    
    # Create a more comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training time comparison
    bars1 = ax1.bar(["CNN", "ResNet"], [dur_dqn, dur_res], color=['skyblue', 'lightcoral'])
    ax1.bar_label(bars1, labels=[f"{dur_dqn:.1f}s", f"{dur_res:.1f}s"], padding=3)
    ax1.set_title("Training Time Comparison")
    ax1.set_ylabel("Time (seconds)")
    
    # Overall performance comparison
    means = [eval_stats_dqn['overall_mean_reward'], eval_stats_res['overall_mean_reward']]
    stds = [eval_stats_dqn['overall_std_reward'], eval_stats_res['overall_std_reward']]
    bars2 = ax2.bar(["CNN", "ResNet"], means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral'])
    ax2.bar_label(bars2, labels=[f"{means[0]:.1f}", f"{means[1]:.1f}"], padding=3)
    ax2.set_title(f"Overall Performance Comparison\n({N_EVAL_SEEDS} seeds × {N_EVAL_EPISODES} episodes = {eval_stats_dqn['total_episodes']} total episodes)")
    ax2.set_ylabel("Mean Reward")
    
    # Seed variability comparison (rewards)
    seed_vars_reward = [eval_stats_dqn['seed_std_rewards'], eval_stats_res['seed_std_rewards']]
    bars3 = ax3.bar(["CNN", "ResNet"], seed_vars_reward, color=['skyblue', 'lightcoral'])
    ax3.bar_label(bars3, labels=[f"{seed_vars_reward[0]:.2f}", f"{seed_vars_reward[1]:.2f}"], padding=3)
    ax3.set_title("Seed Variability (Reward)")
    ax3.set_ylabel("Std Dev across Seeds")
    
    # Episode length comparison
    length_means = [eval_stats_dqn['overall_mean_length'], eval_stats_res['overall_mean_length']]
    length_stds = [eval_stats_dqn['overall_std_length'], eval_stats_res['overall_std_length']]
    bars4 = ax4.bar(["CNN", "ResNet"], length_means, yerr=length_stds, capsize=5, color=['skyblue', 'lightcoral'])
    ax4.bar_label(bars4, labels=[f"{length_means[0]:.0f}", f"{length_means[1]:.0f}"], padding=3)
    ax4.set_title("Episode Length Comparison")
    ax4.set_ylabel("Mean Episode Length")
    
    plt.tight_layout()
    
    # Save comparison plot
    outfile = os.path.join(DIR_IMAGES, f'Comparativa_MultiSeed_{ENV_ID}.png')
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"💾 Enhanced comparison plot saved to: {outfile}")
    
    # Add to tensorboard
    writer = SummaryWriter(log_dir=DIR_LOGS)
    writer.add_figure("Enhanced_Comparison", fig, TOTAL_TIMESTEPS)
    writer.close()
    plt.close(fig)

    # --- 4) Detailed Performance Summary -----------------------------------
    print("\n" + "="*80)
    print("📋 DETAILED RESULTS SUMMARY (MULTI-SEED EVALUATION)")
    print("="*80)
    print(f"Training Steps: {TOTAL_TIMESTEPS:,}")
    print(f"Evaluation Seeds: {EVAL_SEEDS}")
    print(f"Episodes per Seed: {N_EVAL_EPISODES}")
    print(f"Total Evaluation Episodes: {eval_stats_dqn['total_episodes']}")
    print("-"*80)
    print(f"CNN DQN:")
    print(f"  Training Time: {dur_dqn:.1f}s")
    print(f"  Overall Mean Reward: {eval_stats_dqn['overall_mean_reward']:.2f} ± {eval_stats_dqn['overall_std_reward']:.2f}")
    print(f"  Overall Mean Length: {eval_stats_dqn['overall_mean_length']:.1f} ± {eval_stats_dqn['overall_std_length']:.1f}")
    print(f"  Seed Variability (Reward): {eval_stats_dqn['seed_std_rewards']:.2f}")
    print(f"  Seed Variability (Length): {eval_stats_dqn['seed_std_lengths']:.1f}")
    print(f"  Per-seed rewards: {[f'{r:.1f}' for r in eval_stats_dqn['seed_mean_rewards']]}")
    print("-"*80)
    print(f"ResNet DQN:")
    print(f"  Training Time: {dur_res:.1f}s")
    print(f"  Overall Mean Reward: {eval_stats_res['overall_mean_reward']:.2f} ± {eval_stats_res['overall_std_reward']:.2f}")
    print(f"  Overall Mean Length: {eval_stats_res['overall_mean_length']:.1f} ± {eval_stats_res['overall_std_length']:.1f}")
    print(f"  Seed Variability (Reward): {eval_stats_res['seed_std_rewards']:.2f}")
    print(f"  Seed Variability (Length): {eval_stats_res['seed_std_lengths']:.1f}")
    print(f"  Per-seed rewards: {[f'{r:.1f}' for r in eval_stats_res['seed_mean_rewards']]}")
    print("-"*80)
    
    # Determine winner with statistical considerations
    reward_diff = eval_stats_res['overall_mean_reward'] - eval_stats_dqn['overall_mean_reward']
    if abs(reward_diff) > max(eval_stats_res['overall_std_reward'], eval_stats_dqn['overall_std_reward']) / 2:
        if reward_diff > 0:
            winner = "ResNet DQN"
            improvement = reward_diff
        else:
            winner = "CNN DQN"
            improvement = abs(reward_diff)
        print(f"🏆 Clear Winner: {winner} (+{improvement:.2f} reward difference)")
    else:
        print(f"🤝 Performance is similar (difference: {abs(reward_diff):.2f})")
        if eval_stats_res['seed_std_rewards'] < eval_stats_dqn['seed_std_rewards']:
            print(f"📊 ResNet DQN shows more consistent performance across seeds")
        elif eval_stats_dqn['seed_std_rewards'] < eval_stats_res['seed_std_rewards']:
            print(f"📊 CNN DQN shows more consistent performance across seeds")
    
    print("="*80)

    print(f"\n✅ Logs TB en: {DIR_LOGS}")
    print("Ejecuta: tensorboard --logdir", DIR_LOGS)
    print("Verás dos runs: cnn_run y resnet_run en la interfaz.")

    # --- 5) Final Videos ---------------------------------------------------
    print("\n🎬 Generating final demonstration videos...")
    
    # DQN Video
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
    print(f"🏆 Video DQN: {glob.glob(os.path.join(VIDEO_DIR, 'dqn_final*.mp4'))[-1]}  Reward={total_reward:.1f}")

    # ResNet Video
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
    print(f"🏆 Video ResNet: {glob.glob(os.path.join(VIDEO_DIR, 'resnet_final*.mp4'))[-1]}  Reward={total_reward:.1f}")

    try:
        if os.name == 'nt':
            os.startfile(os.path.abspath(VIDEO_DIR))
    except:
        pass

    print("\n✅ Proceso completado.")
