from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
from .env import DungeonMasterEnv


def train_ppo(total_timesteps: int = 5000, save_path: str = "dm_ppo.zip") -> None:
    def make_env():
        return DungeonMasterEnv()

    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "figures")
    os.makedirs(logs_dir, exist_ok=True)

    def make_monitored():
        return Monitor(DungeonMasterEnv())

    env = DummyVecEnv([make_monitored])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)

    # Plot episode rewards if available
    monitor_files = [f for f in os.listdir('.') if f.startswith('monitor') and f.endswith('.csv')]
    if monitor_files:
        try:
            import pandas as pd
            rewards = []
            for f in monitor_files:
                df = pd.read_csv(f, comment='#')
                rewards.extend(df['r'].tolist())
            if rewards:
                plt.figure(figsize=(6,4))
                plt.plot(rewards)
                plt.title('PPO Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.tight_layout()
                plt.savefig(os.path.join(logs_dir, 'ppo_rewards.png'))
                plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    train_ppo()


