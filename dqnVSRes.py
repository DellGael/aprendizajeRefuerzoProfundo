# Arquitectura Residual
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        n_ch = obs_space.shape[0]
        self.body = nn.Sequential(
            ResidualBlock(n_ch, 32, stride=4),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(obs_space.sample()[None]).float()
            n_flatten = self.body(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    def forward(self, obs):
        return self.linear(self.body(obs))

# ── Entrenamiento y evaluación ──────────────────────────────────────
if _name_ == "_main_":
    # --- 1) DQN run ---------------------------------------------------
    log_dir_dqn = os.path.join(DIR_LOGS, "dqn_run")
    os.makedirs(log_dir_dqn, exist_ok=True)
    logger_dqn  = configure(log_dir_dqn, ["tensorboard"])

    env_dqn     = make_env(N_ENVS)
    cb_dqn      = CheckpointCallback(SAVE_FREQ, DIR_MODELS, name_prefix="dqn_atari")
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
        policy_kwargs={"net_arch": [512, 512]},
        device="cuda",
        verbose=0,
        seed=SEED
    )
    model_dqn.set_logger(logger_dqn)

    t0 = time.time()
    model_dqn.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[cb_dqn, eval_cb_dqn],
        tb_log_name="dqn_run",
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
        "net_arch": [512, 512]
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
    fig, ax = plt.subplots()
    ax.bar(["DQN", "ResNet"], [dur_dqn, dur_res])
    SummaryWriter(log_dir=DIR_LOGS).add_figure("Comparativa", fig, TOTAL_TIMESTEPS)
    plt.close(fig)


#Extraccion de los videos del agente
for f in glob.glob(os.path.join(VIDEO_DIR, "dqn_final*.mp4")):
    os.remove(f)
env_vid = VecFrameStack(
    make_atari_env(ENV_ID, n_envs=1, seed=SEED),
    FRAME_STACK
)
vid_rec = VecVideoRecorder(
    env_vid,
    VIDEO_DIR,
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
env_vid = VecFrameStack(
    make_atari_env(ENV_ID, n_envs=1, seed=SEED),
    FRAME_STACK
)
vid_rec = VecVideoRecorder(
    env_vid,
    VIDEO_DIR,
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
