# Is the ACT model too small, and what limits picking a hidden target?

Web research on 2026-11-22 for the hidden-ball failure (the policy dives to a typical depth, closes short, lifts and dives again). Most quotes came through a page-summary tool; the key numbers were checked on arXiv or ar5iv. Anything marked UNVERIFIED was not checked against a primary source. Our own numbers are in `kb/act-model-facts.md`; what we chose is in `decisions/hidden-ball-picking.md`.

**Short answer:** the literature does not support model size as the cause. It points to observability, a memoryless deterministic policy, the demonstrations, and executing 100 actions open loop, which is LeRobot's default and not what the ACT paper ran.

## Size of ACT

- The ACT paper reports "around 80M" parameters, ResNet-18, 4 encoder and 7 decoder layers, hidden size 512, trained from scratch per task on 50 demonstrations (100 for one task); real-task success 88, 96, 84, 92, 64 and 20% [1].
- LeRobot's ACT has 1 decoder layer: its config says the original code has a bug "that means only the first layer is used" [2]. The official `act_aloha_sim_transfer_cube_human` checkpoint shows 51.7M parameters [4].
- The paper queries the policy "at every timestep" with temporal ensembling; LeRobot's default executes all 100 actions with ensembling off [1, 2].
- Ablations in [1]: chunking helped most (1% at k=1, 44% at k=100, then "slightly tapers down"); temporal ensembling added 3.3%; removing the CVAE dropped human-data success from 35.3% to 2%.

## Is capacity the bottleneck at our data scale?

| Source | Finding |
|---|---|
| robomimic [5] | A larger MLP "reduces performance uniformly". Operator quality matters: 66.0 with the better operators against 39.3 with the worse. |
| Diffusion Policy [6] | From scratch, ResNet-18 0.94 and ResNet-34 0.92. |
| Data Scaling Laws [7] | Diversity of environments and objects is "far more important" than the number of demonstrations; a larger action network did not help. |
| ALOHA Unleashed [8] | "ACT with a smaller network is easier to tune and often times performs better in a low data regime." |
| Octo [9] | A large transformer trained "from scratch overfit quickly on the small datasets" (about 100 demonstrations). |
| SmolVLA [12] | SO-101 pick-place: ACT 70% in distribution, 40% out of distribution; SmolVLA 90 and 50%. |

No ablation of ACT's ResNet depth or transformer width was found.

## Hidden targets

- Mechanical search treats uncovering a hidden target as its own search behaviour: repeated push, suction and grasp actions until the target is out [13, 14]. Deep RL from images "fails when applied directly" [15].
- After a failed grasp "the bin is often left unchanged", so a policy without memory retries the same grasp; remembering failures raised picks per hour by 107% [17].
- Observation history cuts both ways: it helps in robomimic [5], but "more information can yield worse performance" [19] and image policies get worse with longer history [6].
- A wrist camera "consistently improves training efficiency and out-of-distribution generalization" [23]. Point-cloud input helps (+24.2% relative) [24].
- Nothing was found that targets reach height from RGB only.

## Chunk length and re-planning

- Executing a long chunk open loop "reduces performance due to slow reaction time"; 8 steps was best for Diffusion Policy [6].
- With no noise longer horizons win, with high noise executing 1 step wins; "there is no universally optimal choice" [26].
- SmolVLA in simulation: executing 1, 10, 30 or 50 steps before re-planning gave 80.3, 82.8, 70.8 and 51.8% [12]. pi0 executes 16 or 25 of 50 [11].

## Demonstrations and corrections

- Keeping only the shortest 50% of episodes raised success from 30% to 55% [8].
- Intervention data beats the same number of ordinary demonstration samples [36]; 50 correction episodes on 150 to 405 demonstrations gave +50 to 60 points, fine-tuning was unstable [38]; recovery data "scales linearly in the number of recovery maneuvers" [39].
- For one failure mode the papers use tens to about 50 correction episodes; no general number.

## Other

- Mean-squared-error regression "assumes that the data distribution is unimodal" [28]. A single-author 2026 preprint says "ACT permits posterior collapse" and could not reproduce the CVAE ablation [30]. UNVERIFIED beyond the abstract.
- DETR does worse on small objects; dilating the last backbone stage raises small-object AP from 20.5 to 22.5 at 2x compute [40].
- Shortening the horizon with waypoints gave +4 to 28% [44]; no controlled small-data comparison of a supervisor with sub-skills against one policy was found.

## Reading for our case (inference, not sourced)

Ranked by support: observability; no memory with a deterministic policy; demonstration quality and coverage; open-loop horizon; averaging over several valid actions; visual resolution; model capacity (least). The conversion of "0.5 to 1 s before re-planning" to 15 to 30 steps at 30 fps is ours.

## Sources

1. Zhao, Kumar, Levine, Finn 2023, Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, https://arxiv.org/abs/2304.13705
2. LeRobot `configuration_act.py`, https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py
4. `lerobot/act_aloha_sim_transfer_cube_human` model card, https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human
5. Mandlekar et al. 2021, robomimic, https://arxiv.org/abs/2108.03298
6. Chi et al. 2023, Diffusion Policy, https://arxiv.org/abs/2303.04137
7. Lin et al. 2024, Data Scaling Laws in Imitation Learning for Robotic Manipulation, https://arxiv.org/abs/2410.18647
8. Zhao et al. 2024, ALOHA Unleashed, https://arxiv.org/abs/2410.13126
9. Octo Model Team 2024, Octo, https://arxiv.org/abs/2405.12213
11. Black et al. 2024, pi0, https://arxiv.org/abs/2410.24164
12. Shukor et al. 2025, SmolVLA, https://arxiv.org/abs/2506.01844
13. Danielczuk et al. 2019, Mechanical Search, https://arxiv.org/abs/1903.01588
14. Danielczuk et al. 2020, X-Ray, https://arxiv.org/abs/2004.09039
15. Kurenkov et al. 2020, Visuomotor Mechanical Search, https://arxiv.org/abs/2008.06073
17. Sanders et al. 2020, Non-Markov Policies to Reduce Sequential Failures in Robot Bin Picking, https://arxiv.org/abs/2007.10420
19. de Haan, Jayaraman, Levine 2019, Causal Confusion in Imitation Learning, https://arxiv.org/abs/1905.11979
23. Hsu et al. 2022, Vision-Based Manipulators Need to Also See from Their Hands, https://arxiv.org/abs/2203.12677
24. Ze et al. 2024, 3D Diffusion Policy, https://arxiv.org/abs/2403.03954
26. Liu et al. 2024, Bidirectional Decoding, https://arxiv.org/abs/2408.17355
28. Shafiullah et al. 2022, Behavior Transformers, https://arxiv.org/abs/2206.11251
30. Kang 2026, The Latent That Never Was, https://arxiv.org/abs/2609.16745
36. Mandlekar et al. 2020, Human-in-the-Loop Imitation Learning using Remote Teleoperation, https://arxiv.org/abs/2012.06733
38. Xu et al. 2025, Compliant Residual DAgger, https://arxiv.org/abs/2506.16685
39. Hu et al. 2025, RaC, https://arxiv.org/abs/2509.07953
40. Carion et al. 2020, DETR, https://arxiv.org/abs/2005.12872
44. Shi et al. 2023, AWE, https://arxiv.org/abs/2307.14326
