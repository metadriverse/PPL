import argparse
import os
import uuid
from pathlib import Path

from ppl.experiments.metadrive.experttakeover_env import ExpertTakeoverEnv
from ppl.ppl import PPL
from ppl.sb3.common.callbacks import CallbackList, CheckpointCallback
from ppl.sb3.common.monitor import Monitor
from ppl.sb3.common.vec_env import SubprocVecEnv
from ppl.sb3.common.wandb_callback import WandbCallback
from ppl.sb3.haco import HACOReplayBuffer
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.shared_control_monitor import SharedControlMonitor
from ppl.utils.utils import get_time_str
import pathlib
FOLDER_PATH = pathlib.Path(__file__).parent.parent
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="ppl_metadrive", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--save_freq", default=150, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--num_predicted_steps", default=20, type=int)   # The predictor anticipates the next H steps from the current state
    parser.add_argument("--preference_horizon", default=3, type=int)   # Add the first L predicted steps to the preference buffer
    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")   # Debug mode
    parser.add_argument("--bc_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", default=0.1, type=float)
    
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = "PPL"
    if args.only_bc_loss=="True":
        experiment_batch_name = "PPL_BCLossOnly"
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = FOLDER_PATH.parent.parent
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(
            num_predicted_steps=args.num_predicted_steps,
            preference_horizon=args.preference_horizon,
        ),

        # Algorithm config
        algo=dict(
            only_bc_loss=args.only_bc_loss,
            bc_loss_weight=args.bc_loss_weight,
            beta = args.beta,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=10,  # The number of steps before
            batch_size=args.batch_size,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )
    if args.toy_env:
        config["env_config"].update(
            num_scenarios=1,
            traffic_density=0.0,
            map="COT",
            use_render=True
        )
        
    # ===== Setup the training environment =====
    train_env = ExpertTakeoverEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            start_seed=1000,
        )
        from ppl.experiments.metadrive.driving_env import DrivingEnv
        from ppl.sb3.common.monitor import Monitor
        eval_env = DrivingEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env, eval_freq = SubprocVecEnv([_make_eval_env]), 150

    # ===== Setup the callbacks =====
    save_freq = args.save_freq  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PPL(**config["algo"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from ppl.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    train_env.env.env.model = model
    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=20_000,
        callback=callbacks,
        reset_num_timesteps=True,
        # eval
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),
        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
    )
