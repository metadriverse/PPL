import argparse
import os
import uuid
from pathlib import Path

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.pvp_td3 import COMB
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
import pathlib
FOLDER_PATH = pathlib.Path(__file__).parent.parent
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="pvp_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="Drive0723", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default=FOLDER_PATH.parent.parent, help="Folder to store the logs.")
    parser.add_argument("--bc_loss_weight", type=float, default=1.0)
    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--future_steps_predict", default=20, type=int)
    parser.add_argument("--update_future_freq", default=10, type=int)
    parser.add_argument("--future_steps_preference", default=3, type=int)
    parser.add_argument("--expert_noise", default=0, type=float)
    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    parser.add_argument("--dpo_loss_weight", default=1.0, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--bias", default=0.5, type=float)
    
    args = parser.parse_args()

    # ===== Set up some arguments =====
    #experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    experiment_batch_name = "{}_bcw={}_0410".format("Ours", args.bc_loss_weight)
    if (args.only_bc_loss=="True") or (args.dpo_loss_weight == 0):
        experiment_batch_name = "BCLossOnly_"
    seed = args.seed
    #trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    trial_name = "{}_{}".format(experiment_batch_name, uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(

            # Original real human exp env config:
            # use_render=True,  # Open the interface
            # manual_control=True,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),

            # FakeHumanEnv config:
            use_render=False,
            future_steps_predict=args.future_steps_predict,
            update_future_freq=args.update_future_freq,
            future_steps_preference=args.future_steps_preference,
            expert_noise=args.expert_noise,
        ),

        # Algorithm config
        algo=dict(
            # intervention_start_stop_td=args.intervention_start_stop_td,
            adaptive_batch_size=args.adaptive_batch_size,
            bc_loss_weight=args.bc_loss_weight,
            only_bc_loss=args.only_bc_loss,
            dpo_loss_weight = args.dpo_loss_weight,
            alpha = args.alpha,
            bias = args.bias,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
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
            # Here we set num_scenarios to 1, remove all traffic, and fix the map to be a very simple one.
            num_scenarios=1,
            traffic_density=0.0,
            map="COT",
            use_render=True
        )
        
    # ===== Setup the training environment =====
    train_env = FakeHumanEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    # Store all shared control data to the files.
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    if config["env_config"]["use_render"]:
        eval_env, eval_freq = None, -1
    else:
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
    model = COMB(**config["algo"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    train_env.env.env.model = model
    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
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
