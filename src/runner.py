import os
import gymnasium as gym
import time
import argparse
import importlib

from mani_skill.utils.wrappers.record import RecordEpisode

def generate_videos(import_path, cls_name, env_name, n_episodes=10, max_steps_per_episode=100, video_dir="task_videos"):
    """
    Generate and save videos of random agent interactions in a custom environment.
    """
    # Dynamically import the environment class using the import path
    module = importlib.import_module(import_path)
    getattr(module, cls_name)

    # Create the environment instance using the env_name and other configurations
    env = gym.make(env_name, obs_mode="state", render_mode="rgb_array")
    
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate videos of agent interactions with custom environments.")
    parser.add_argument('--import_path', type=str, required=True, help="The module path to import the environment class (e.g., 'JunhaoChen.card_stack_env')")
    parser.add_argument('--cls_name', type=str, required=True, help="The environment class name (e.g., 'CardStackEnv')")
    parser.add_argument('--env_name', type=str, required=True, help="The environment name (e.g., 'CardStack-v1')")
    parser.add_argument('--n_episodes', type=int, default=5, help="Number of episodes to run")
    parser.add_argument('--max_steps_per_episode', type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument('--video_dir', type=str, default="task_videos", help="Directory to save videos")
    
    args = parser.parse_args()
    
    # Generate videos using the provided arguments
    generate_videos(import_path=args.import_path, 
                    cls_name=args.cls_name, 
                    env_name=args.env_name,
                    n_episodes=args.n_episodes, 
                    max_steps_per_episode=args.max_steps_per_episode, 
                    video_dir=args.video_dir)
