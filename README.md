# 3D Machine Intelligence Team Project: Research Topic Guide

3D Machine Intelligence (59358-01), Fall 2026, instructor Jongmin Lee (jmlee@cau.ac.kr). As of 2026-09-21.

This folder is the **research topic guide** for the team project. Under six themes it covers **15 main-body tasks (T01 to T15)** and **16 appendix tasks (A01 to A16)**. The project is self-directed research. A team picks one task, runs the baseline, turns a failure case into a research question, and presents at the week 7 midterm and the week 15 final. This guide does not hand you a finished research idea. It gives you **questions and comparison criteria so that you define the problem, the hypothesis, and the experiment yourself**.

## 1. What is in this folder

| File | Content |
| --- | --- |
| `README.md` | This file. How the project runs, status labels, the full task list (15 main + 16 appendix), presentation checklists, what to verify before you start |
| `01_reconstruction_mapping.md` | Theme 1. 3D reconstruction & mapping (main T01 to T03, appendix A01, A02) |
| `02_correspondence_registration_localization.md` | Theme 2. Correspondence, registration & localization (main T04, appendix A03 to A07) |
| `03_4d_reconstruction_spatial_memory.md` | Theme 3. 4D reconstruction & spatial memory (main T06, T07, appendix A08) |
| `04_object_pose_affordance.md` | Theme 4. Object pose, affordance (main T13, T14, appendix A10, A12, A13) |
| `05_robotic_manipulation.md` | Theme 5. Robotic manipulation (main T08 to T11, appendix A14, A16) |
| `06_world_models_representation.md` | Theme 6. World models & representation learning (main T05, T12, T15, appendix A09, A11, A15) |
| `*.pdf` | A4 PDF of each of the seven documents (English) |
| `all.pdf` | All seven documents in one PDF (English) |
| `3DV_project_topics.pdf` | The slides presented in class |
| `mds/`, `pdfs_korean/` | The Korean version: the seven documents as Markdown and as PDF (with `all.pdf`) |
| `mds_en/` | The English Markdown sources of the PDFs above |

Every theme document has the same structure: an overview (main-body table, appendix table, how neighboring tasks differ, common preparation), the main-body tasks, the appendix tasks, and references. Each task is described in the following eight parts.

1. Task name and a one-sentence definition
2. Input and output, key assumptions, example uses
3. Why it is hard, typical failure cases
4. Executable baseline candidates (A) and reference papers (B). The two are always kept apart
5. Minimum scope on a small GPU and the first experiment
6. Two research questions a student can develop (a hypothesis and an axis of comparison, not a finished solution)
7. Datasets, metrics, minimum set of comparisons
8. Expected deliverables and evaluation mistakes to avoid

## 2. Schedule and deliverables

**By the week 7 midterm presentation**

- Input, output, and problem definition of the chosen task.
- Related work and the reason for the baseline choice.
- Code, environment, data, and a **minimum baseline run** (a quantitative evaluation under identical conditions). A screenshot of a successful install does not count as a completed baseline.
- An observed failure case, one research question built on it, and a falsifiable hypothesis.
- The module you will change, the comparisons, the metrics, and the ablation plan.
- An experiment plan that fits the resources you have and the time that remains.

**At the week 15 final presentation**

- A fair quantitative and qualitative comparison of the baseline and the proposed method.
- Ablations of the key components, resource usage (time, VRAM), failure cases.
- Whether the hypothesis was supported, the scope of validity, and the limits.
- Reproducible code, configuration, run instructions, and a results report.

**Research principle.** Aim at problem definition and validation of CVPR, ICCV, ICML, or CoRL quality, and submission is encouraged. You do not have to hit a deadline. Submission, acceptance, and beating the state of the art are not required outcomes. **A negative result counts as a research result when the experiment is well controlled and the cause is analyzed.** The criteria are a clear research question, a fair comparison, ablations, failure analysis, and reproducibility.

**Not decided here.** Team size, lecture hours, and grading weights are not set in this document. Team formation follows the sign-up sheet announced in the week 3 class. Do not assume a grading split until it is announced separately.

## 3. Status labels and resource principles

**Default environment.** A single GPU with about 12 to 16 GB of VRAM, or Colab. This is the **design target** of the course, not the measured requirement of every baseline. Distinguish what can run on Colab from what needs a local Linux GPU. Check not only VRAM but also installation and CUDA dependencies, data size, simulators, checkpoints, licenses, and access rights.

**Status labels.**

| Label | Meaning |
| --- | --- |
| Public candidate | Code and the needed checkpoint, or a training-free path, are public. **GPU execution in the student environment is for the team to confirm.** |
| Conditional candidate | A condition must be resolved first: input reduction, old dependencies, a simulator, data or model access. Not assigned for sure until it passes. |
| Reduced re-implementation | Instead of the paper's full pipeline, you implement the small comparison experiment this guide proposes. It is not called a reproduction of the paper. |
| Reference only | Read for ideas and background. Full reproduction is not required. The reason (large-scale training, high inference cost, no code, background) is stated for each. |

**Always separate three conditions.** (1) What the official documentation or paper reports, (2) what the official repository and project page state, (3) the **unverified reduced setting** this guide proposes for a student project. "A pretrained model exists" does not by itself guarantee that it runs on a small GPU.

**Six inference/test-time-only tasks.** Main-body T06, T07, T08, T10 and appendix A13, A16 do **inference or test-time improvement of an existing model or policy only**. Allowed: per-input pose or deformation optimization, tracking/refinement, filtering, candidate re-ranking, planning at inference time, retargeting, frozen-policy evaluation. Not allowed: retraining a backbone or a policy on a dataset, training a new RL policy (even a small one). **Separate per-input optimization (allowed) from dataset-based training (not allowed).** For T05 (video world model) the scope in this guide is also inference-side improvement of a pretrained model.

**All other tasks.** Prefer small data, a frozen feature extractor, training a small module, cached features, limited resolution and frame counts. Do not hide the difference between a full reproduction and a reduced experiment.

**Papers.** Each task lists one or two executable baselines, reference papers, and recent and related papers. Recent and related papers are reading material, not executable baselines. Tasks without a verified executable candidate (T04, A11) are marked as reduced re-implementation.

## 4. Full task list

**15 main-body tasks.** The actual scope, the default baseline, and the status were set by the instructor.

| ID | Theme | Task | Actual scope | Default baseline | Status | Difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | 1 | Streaming 3D reconstruction | Keyframe, memory, and fusion improvements on a pretrained reconstructor | Spann3R | Conditional candidate | Medium |
| T02 | 1 | Sparse-view 3D reconstruction | Input selection, confidence, small corrections for 2-view reconstruction | MVSplat | Public candidate | Medium |
| T03 | 1 | 3D Gaussian Splatting compression | Pruning, quantization, and a short recovery on a trained small scene | LightGaussian | Conditional candidate | Medium |
| T04 | 2 | VLM-based scene registration | A small VLM's object matching combined with geometric verification | PROSE-inspired reduced implementation | Reduced re-implementation | Medium |
| T05 | 6 | Video world model | Sampler, rollout, and candidate selection improvements on a pretrained model | DIAMOND; optionally Wan2.1-T2V-1.3B | Public candidate / conditional | Low |
| T06 | 3 | Dynamic 4D reconstruction | Pretrained inference on short videos, window optimization | MonST3R | Conditional candidate, **inference/test-time only** | High |
| T07 | 3 | Animal pose and shape reconstruction | Pretrained animal model inference, 2D keypoint fitting and smoothing | AniMer, AniMer+ | Public candidate, **inference/test-time only** | Medium |
| T08 | 5 | Language-guided dexterous grasping | Pretrained inference and test-time refinement of language-conditioned multi-finger grasps | DexGYSNet, DextER | Conditional candidate, **inference/test-time only** | High |
| T09 | 5 | Wrist-guided humanoid manipulation | Adjust only the wrist terms of WristMimic, measure whole-body stability | WristMimic | Conditional candidate | High |
| T10 | 5 | Cross-embodiment robot manipulation | Input and output conversion for a pretrained policy, execution constraints | CrossFormer | Public candidate (real transfer evaluation is conditional), **inference/test-time only** | High |
| T11 | 5 | Grasp pose generation | Candidate reranking, diversity, and confidence calibration on a trained grasp network | Contact-GraspNet | Conditional candidate | Medium |
| T12 | 6 | Latent world model | A frozen visual encoder with a small dynamics model and planner | DINO-WM | Public candidate | Low |
| T13 | 4 | Novel object 6D pose estimation | Pose candidate generation and selection, mask refinement, for a novel object with a CAD model | FoundationPose (model-based) | Conditional candidate | High |
| T14 | 4 | 3D affordance prediction | Post-processing, prompts, and a small head on a pretrained point-text alignment | OpenAD | Public candidate | Low |
| T15 | 6 | Test-time adaptation for 3D perception | Few-gradient-step adaptation of a pretrained 3D model | MATE | Conditional candidate | Medium |

**16 appendix tasks.** You may choose them, but they are not presented in class. Each is described in section 3 of its theme document.

| ID | Theme | Task | Default baseline | Status | Difficulty |
| --- | --- | --- | --- | --- | --- |
| A01 | 1 | Feed-forward 3D reconstruction | DUSt3R, VGGT | Conditional candidate | Medium |
| A02 | 1 | Robust 3D reconstruction | DUSt3R (frozen), VGGT | Conditional candidate | Medium |
| A03 | 2 | 3D point cloud registration | GeoTransformer, FPFH+RANSAC | Conditional candidate | Medium |
| A04 | 2 | Low-overlap registration | PREDATOR | Conditional candidate | High |
| A05 | 2 | Multi-view correspondence | LightGlue + student-implemented tracks | Public candidate | Low |
| A06 | 2 | Cross-modal registration | CoFiI2P | Conditional candidate | High |
| A07 | 2 | Visual localization | hloc (+ LightGlue) | Conditional candidate | Medium |
| A08 | 3 | Static-dynamic scene decomposition | RGB-D residual re-implementation of the DynaSLAM idea | Reduced re-implementation / original conditional | Low |
| A09 | 6 | Object-centric world model | SlotFormer | Public candidate | Medium |
| A10 | 4 | 6D object pose tracking | FoundationPose (tracking) | Conditional candidate | High |
| A11 | 6 | Representation alignment for generation | Reduced REPA experiment | Reduced re-implementation | High |
| A12 | 4 | Language-conditioned affordance grounding | OpenAD + extension re-implementation | Public baseline + extension | Medium |
| A13 | 4 | Hand-object interaction reconstruction | ContactOpt | Conditional candidate, **inference/test-time only** | High |
| A14 | 5 | Language-conditioned grasping | GraspGPT (public lightweight variant) | Conditional candidate | Medium |
| A15 | 6 | Action-conditioned 3D prediction | DPI-Net or a reduced GNN | Conditional / reduced re-implementation | Medium |
| A16 | 5 | Human-to-robot interaction transfer | DexPilot-style method + dex-retargeting | Public candidate, **inference/test-time only** | Low |

Difficulty is about environment preparation and implementation load. **Low**: you start from a public checkpoint or a CPU run and change little. **Medium**: an install, a cache, input reduction, or data access comes first. **High**: at least one of a simulator, old dependencies or a CUDA build, 100 GB-scale data, or model training.


## 5. Choosing a topic and the minimum checks before committing

**Selection criteria.** Weigh **the team's interest and the actual execution environment** together, rather than ranking research value. Comparatively light paths: T12 (small training on cached features), T05 (pretrained inference), T14 (cached text features), T07 (pretrained animal model). Heavy preparation: T08 (hand model and simulator), T09 (Isaac Gym training), T10 (robot simulator), T13 (CUDA rasterizer and BOP data), T06 (memory reduction first). In the appendix, A16 (kinematic retargeting, CPU is enough), A05 (pairwise matching and track building), A08 (residual mask starting from GT poses) are light, while A06 (100 GB-scale RGB-LiDAR data), A11 (reduced model implementation), A13 (mesh and MANO inputs) are heavy. Do not assume that Docker, old CUDA, or a graphics simulator works out of the box on Colab.

**Four minimum checks before you commit to a topic.**

1. Record the repository commit, the checkpoint name, the input data subset, and how it is accessed.
2. Confirm on a real GPU that inference, adaptation, or a short training run succeeds on **one input**, and measure the GPU model, peak VRAM, CPU RAM, and time.
3. Budget the repeated experiments, including preprocessing, cache generation, simulator installation, and download size.
4. Record every change from the original: resolution, number of points, number of frames, iterations, evaluation scope. If execution is blocked, choose a small geometric comparison or a reduced re-implementation **within the same task** and state the change.

**Common evaluation principles.**

- Split train/validation/test and check for **leakage through scene, object, or temporal overlap**. Another frame of the same scene or another view of the same object on both sides is leakage.
- For test-time adaptation (T15), never use test labels, and state the adaptation scope (which layers), whether the state is reset, and the extra compute.
- In the robotics tasks (T08 to T11, A14, A16), **separate offline pose/action error from actual closed-loop success rate**. Without a simulator run, do not claim a success rate.
- Do not just list dataset names; write **what a small subset lets you evaluate**.
- Compare at the same compute budget (steps, wall-clock, RANSAC iterations).

## 6. Presentation checklists

**Week 7 midterm**

- [ ] Task ID and name, input and output explained on one slide
- [ ] Reason for the baseline choice and its status (public / conditional / reduced) stated
- [ ] A run on one input with measurements (GPU, VRAM, time)
- [ ] A quantitative evaluation under identical conditions (at least one metric)
- [ ] An observed failure case shown
- [ ] One research question and a falsifiable hypothesis
- [ ] The module to change, comparisons, ablations, and an experiment plan for the remaining weeks
- [ ] For an inference-only task, no retraining in the plan

**Week 15 final**

- [ ] Quantitative and qualitative comparison against the baseline under identical conditions
- [ ] The key ablation
- [ ] Failure cases and cause analysis
- [ ] Time and memory cost reported
- [ ] Whether the hypothesis was supported, the scope and the limits
- [ ] Reproducible code, configuration, run instructions
- [ ] The difference between a reproduction of the paper and a reduced experiment stated

## 7. What the team verifies before starting

The items below are taken from official documentation. Read them as conditional, and verify them yourselves before committing to the project.

- Execution requirements are copied from official documentation (for example, MonST3R about 33 GB by default and about 23 GB non-batchified, Contact-GraspNet inference 8 GB or more, MVSplat training on an A100 80 GB, Wan2.1-T2V-1.3B 8.19 GB). Whether they actually run on a 12 to 16 GB GPU or on Colab is for the team to confirm.
- For models without an official minimum VRAM (Spann3R, MVSplat inference, DINO-WM, DIAMOND, and others), measure it yourselves.
- For old dependencies (GeoTransformer, PREDATOR, the TensorFlow of Contact-GraspNet, MATE, the PyFleX of DPI-Net, ContactOpt, the Isaac Gym used by WristMimic and by DexGYS evaluation), first confirm that they install in the current environment.
- Confirm data access first: the CoFiI2P preprocessed data (100 GB scale), MANO for ContactOpt (separate registration), BOP data, the TaskGrasp family, the Atari recordings for DIAMOND, and the download path of the Wan2.1 checkpoint.
- The run sizes written in the documents (number of scenes, seeds, horizon, resolution, and so on) are proposals, not measurements.
- The official PROSE code was not yet public as of September 2026. When it is released, revisit the baseline setup of T04. PANY also has no public code.
- LMAffordance3D is reference only because an official checkpoint has not been confirmed.
- Papers without a stated venue are listed with the arXiv year only. Authors and venues follow the arXiv abstract and the official page.
