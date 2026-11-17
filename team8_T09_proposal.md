# Robust Humanoid Manipulation with SkillMimic-V2

Baseline Reproduction and Research Proposal

October 15, 2026 · [KimSeongMin / ChaSeungJun]

## From WristMimic to SkillMimic-V2

## Baseline Selection

## ① Initial Baseline: WristMimic<sup>[1]</sup>

• Wrist-guided humanoid manipulation.

• Trained for <u>6,000 epochs (\~11.9 hours)</u> using <u>RTX 5070 (12 GB)</u>.

• Result: 0% task success rate in kettle manipulation.

## ② Limitations

• Successful task execution was not achieved in our experiment.

Limited GPU resources constrained further training and tuning.

Lack of environment reconstruction code required challenging manual setup.

## ③ New Baseline: SkillMimic-V2<sup>[2]</sup>

• Inspired by SkillMimic, a physics-based interaction skill learning framework.

• Designed to learn robust interaction skills from sparse and noisy demonstrations.

• Selected as an alternative baseline for further investigation.

• Status: PUBLIC

[1] W. Yu et al., “WristMimic: Full-Body Humanoid Control with Wrist-Guided Manipulation,” ECCV, 2026.

[2] Y. Wang et al., “SkillMimic: Learning Basketball Interaction Skills from Demonstrations,” CVPR, 2025.

## Problem Definition

## Humanoid Interaction Skill Learning

• Learning humanoid control policies to perform diverse physical interactions with objects and the environment.

• Challenge

## Demonstration Sparsity

Limited demonstrations lack diverse states and transitions between skills.



## Demonstration Noise

Noisy demonstrations can lead to inaccurate or physically infeasible interactions.



Goal : Learn robust and generalizable interaction skills from imperfect demonstrations.

Robust Zero-Shot Recovery

Enriched Interactions

## SkillMimic-V2

**Input :** Sparse and Noisy Demonstrations + Robot and Object States

**Output** : Learned Interaction Policy + Whole-Body Control Actions

Sparse Demonstrations



Sparse Human-Object Trajectory



State Transition Field (STF)



Adaptive Trajectory Sampling (ATS)





Robust Zero-Shot Skill Transition

## Result Task

• Task & Environment : Humanoid Object Manipulation

• Cup & Book Manipulation

• A humanoid robot learns to grasp, lift, move, and place objects by imitating reference demonstrations.

• Simulation Environment

• Physics-based humanoid–object interaction in Isaac Gym, using ParaHome demonstration data.

Grasp → Lift → Move → Place





## Result Reproduction result

## • Experimental Setup

• GPU: RTX 5070 (12 GB)

• Simulator: Isaac Gym Preview 4

• Parallel Environments: 2,048

• Training: 3,000 epochs / clip

• Evaluation: 8 environments × 300 steps

Table 1. Matched-budget comparison between original and reconstructed environments. Results are measured at the final checkpoint.

<table><tr><td>Task</td><td>Environment</td><td>Lift success ↑ (%)</td><td>Max lift height ↑ (cm)</td><td>Placement error ↓ (m)</td><td>Peak VRAM (GB)</td><td>Train time (h)</td></tr><tr><td rowspan="2">Cup</td><td>Original</td><td>0</td><td>8</td><td>0.49</td><td>11.4</td><td>9.1</td></tr><tr><td>Reconstructed</td><td>75</td><td>60</td><td>0.40</td><td>11.5</td><td>9.1</td></tr><tr><td rowspan="2">Book</td><td>Original</td><td>0</td><td>4</td><td>0.72</td><td>10.0</td><td>8.3</td></tr><tr><td>Reconstructed</td><td>38</td><td>26</td><td>0.68</td><td>9.3</td><td>8.1</td></tr></table>

Figure 1. Object lifting success rate in the original and reconstructed environments (final checkpoint).



## Research Question

Book manipulation: reward and task performance diverge



• Higher reward may not reflect actual task completion

**Does imitation reward optimization unintentionally encourage task avoidance in humanoid manipulation?**

## Research Plan

| Task | Oct. 4th week | Nov. 1st week | Nov. 2nd week | Nov. 3rd week | Nov. 4th week | Dec. 1st week | Dec. 2nd week |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hypothesis (Task Avoidance) |  |  |  |  |  |  |  |
| Task Avoidance Experiment |  |  |  |  |  |  |  |
| Reward Design &amp; Implementation |  |  |  |  |  |  |  |
| Controlled Training |  |  |  |  |  |  |  |
| Evaluation &amp; Analysis |  |  |  |  |  |  |  |
| Final Report |  |  |  |  |  |  |  |