# VLA-RL-Tasks

This repository is for submitting custom-designed simulation tasks for RL training VLA models. The automated CI will verify the correct implementation under randomly sampled actions, but it will not validate the reward code.

> Currently only supporting Maniskill CI!

## Instructions

1. Create a folder under `src/tasks` named after yourself, with your name properly spelled and capitalized (e.g., JunhaoChen).
2. Define your tasks in `.py` files within your folder.
3. Check the correctness of your custom tasks by running `src/local_runner.py`, ensuring imports and `gym.make` parameters are adjusted accordingly. **Do not modify this file in the repository before submitting pull requests. Instead, use your own local version for testing.**
4. After submitting a pull request, the repository will trigger a CI process that automatically generates trajectory videos with randomly sampled actions. Once the CI passes, pull the repository again to sync the generated videos and review them for any potential bugs.
5. Lastly, update the table below to track your progress.

## Task List

| Task ID | Task Name   | Env Name   | Task Instruction | Sim Env | Name |
|:-------:|:-----------:|:-----------:|:----------------:|:-------:|:----:|
|   0     | Card Stacking (Example) | CardStack-v1 | Stack the cards onto each other. | Maniskill | Junhao Chen |
|   1     | Find Treasure    | FindTreasure-v1 | Identify and move the correct green-marked card to the target place from a 3x4 grid of cards.Each card has a colored marker underneath (green=target, yellow=near target, red=no target nearby). | Maniskill | Zilong Zhou |
|   2     | ...    | ... | ... | ... | ... |

