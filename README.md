# CommonRoad Multi-Agent RL with Centralized Critic

This project trains multiple vehicles to reach their goals in CommonRoad traffic scenarios using PPO with a centralized critic.

**Stack:** Python, PyTorch, Tianshou, PettingZoo, CommonRoad

This project builds on the single-agent CommonRoad RL implementation by A. Kasselmann and extends it to a multi-agent setting as part of my Bachelor's thesis.

Original work:
https://gitlab.lrz.de/cps/cps-rl/safe-rl-autodrive

After the thesis submission, the project was extended with:

- a centralized critic (CTDE) significantly improving training stability and performance
- trajectory export to CommonRoad XML for visualization
- video rendering of trained agent trajectories

## Demo

Three trained agents navigating a CommonRoad scenario, avoiding an obstacle and reaching their respective goals.

![Demo](demo.mp4)