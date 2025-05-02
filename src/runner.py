import os
import gymnasium as gym
import time
import importlib
import csv
from mani_skill.utils.wrappers.record import RecordEpisode

def generate_videos(import_path, cls_name, env_name, n_episodes=10, max_steps_per_episode=100, video_dir="task_videos"):
    """
    Generate and save videos of random agent interactions in a custom environment.
    """
    # Dynamically import the environment class using the import path
    module = importlib.import_module(import_path)
    getattr(module, cls_name)

    # Create the environment instance using the env_name and other configurations
    env = gym.make(env_name, obs_mode="state", render_mode="rgb_array", )
    
    # Set up the video saving directory
    video_dir = os.path.join(video_dir, env_name)
    os.makedirs(video_dir, exist_ok=True)

    # Wrap the environment to record videos
    env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
                        trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    
    # Generate the videos by taking random actions
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
    """
    Read the CSV file and process each environment listed to generate videos.
    """
    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            for row in reader:
                import_path, env_name, cls_name = row
                # Call the video generation function for each environment
                generate_videos(import_path, cls_name, env_name)
                
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    # Path to the CSV file
    csv_file = 'env_classes.csv'
    
    # Process the CSV file and generate videos for each environment
    process_csv(csv_file)
