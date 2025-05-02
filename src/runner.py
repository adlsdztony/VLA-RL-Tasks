import os
import gymnasium as gym
import time
import importlib
import csv
from mani_skill.utils.wrappers.record import RecordEpisode

def generate_videos(import_path, cls_name, env_name, n_episodes=10, max_steps_per_episode=100, video_dir="task_videos"):
    module = importlib.import_module(import_path)
    getattr(module, cls_name)
    env = gym.make(env_name, obs_mode="state", render_mode="rgb_array", )
    
    video_dir = os.path.join(video_dir, env_name)
    os.makedirs(video_dir, exist_ok=True)
    env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
                        trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(max_steps_per_episode):
            action = env.action_space.sample()  # Take random action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()
    print(f"Videos for {env_name} saved in {video_dir}")

def process_csv(csv_file):
    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                import_path, env_name, cls_name = row
                generate_videos(import_path, cls_name, env_name)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    csv_file = 'env_classes.csv'
    process_csv(csv_file)
