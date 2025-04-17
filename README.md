# CartPole Reinforcement Learning Report

## Overview

This project implements and compares three reinforcement learning approaches—Monte Carlo, Q‑Learning, and Deep Q‑Network (DQN)—to solve the OpenAI Gym's CartPole‑v1 environment. Through discrete tabular methods and neural network approximation, exploring trade‑offs in convergence speed, generalization, and scalability.

## Table of Contents

- [Overview](#overview)
- [Introduction](#introduction)
- [Methods](#methods)
  - [Shared Foundations](#shared-foundations)
  - [Monte Carlo](#monte-carlo)
  - [Q‑Learning](#q-learning)
  - [Deep Q‑Network (DQN)](#deep-q-network-dqn)
- [Rendering an Episode](#rendering-an-episode)
- [Results & Evaluation](#results--evaluation)
- [Conclusion](#conclusion)
- [Installation & Usage](#installation--usage)
- [Dependencies](#dependencies)
## Introduction

The CartPole‑v1 task challenges an agent to balance a pole on a cart by applying discrete left or right forces. Rewards are given for each timestep the pole remains upright. We compare:
- **Monte Carlo** (first‑visit returns, tabular updates)
- **Q‑Learning** (off‑policy, tabular updates)
- **Deep Q‑Network (DQN)** (neural network approximation, including Double DQN)

Our goals are to measure convergence rates, stability, and final performance when training under identical conditions.

## Methods

### Shared Foundations

- **State Discretization**: Continuous state variables are binned to enable tabular methods.
- **ε‑Greedy Policy**: Balances exploration and exploitation with decay scheduling.
- **Agent Evaluation**: Standardized evaluation runs to compute average rewards.

### Monte Carlo

- Implements first‑visit Monte Carlo control.
- Tabulates returns for state–action pairs.
- Sensitive to bin granularity and requires many episodes to converge.

### Q‑Learning

- Off‑policy, bootstrapped updates at each timestep.
- Faster convergence than Monte Carlo with appropriate learning rate.
- Robust to discretization choices.

### Deep Q‑Network (DQN)

- Neural network approximates Q‑values.
- Supports experience replay and target networks.
- Includes Double DQN variant to reduce overestimation bias.

## Rendering an Episode

A utility to render a single episode using the trained policy—visualizes CartPole dynamics and demonstrates learned behavior.

## Results & Evaluation

- **Convergence**: Q‑Learning reached the success threshold faster than Monte Carlo.
- **Generalization**: DQN provided stable long‑term performance and generalized across unseen states.
- **Scalability**: Neural methods scale gracefully in high‑dimensional tasks compared to tabular methods.

Plots and animations are included in the notebook to illustrate learning curves and policy behavior.

## Conclusion

This study highlights the trade‑offs between tabular and function‑approximation methods in continuous control tasks. Key takeaways:

- Tabular approaches require careful discretization.
- Exploration strategy tuning critically impacts learning efficiency.
- Neural networks enable scalable solutions but demand more computational resources and careful hyperparameter tuning.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/NomadicSasquatch/CartPole_Reinforcement_Learning_Exp
   cd CartPole_Reinforcement_Learning_Exp
2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook CartPole_Report.ipynb
4. Run cells in order. To skip long training, load precomputed weights and Q‑tables (training code is commented out by default).

## Dependencies
- Python 3.7+
- gym (CartPole‑v1)
- TensorFlow
- numpy, pandas
- matplotlib, seaborn
- pygame (for rendering)
- pickle
Install with:
```bash
  pip install gym tensorflow numpy pandas matplotlib seaborn pygame
