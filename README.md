# Awesome-Large-Language-Diffusion-Models

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A comprehensive list of papers about **Large-Language-Diffusion-Models**.

---

> [!IMPORTANT]
> Contributions welcome:
> - If you have a relevant paper not included in the library, please [contact us](#contact)!  Or, you may also consider submitting 'Pull requests' directly, thank you!
>
> - If you think your paper is more suitable for another category, please [contact us](#contact) or submit 'Pull requests'.
>   
> - If your paper is accepted, you may consider updating the relevant information.
>   
> - Thank you!


---

## 💥 News 💥
- 🔥🔥🔥 Awesome-LLDM is now open!


---

## ⭐️ Useful Resources (Blogs & Technical Reports)

- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Dream-7B](https://hkunlp.github.io/blog/2025/dream/)
- [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/)
- [What are Diffusion Language Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)  

---

## ⚙️ Framework
- [Survey Papers](#survey-papers)
- [Diffusion Language Models](#diffusion-language-models)
  - [Large Diffusion Language Models (>7B)](#large-diffusion-language-models-7b)
    - [Scaling](#scaling)
    - [Caching](#accelerating)
    - [Reasoning](#reasoning)
    - [Others](#others)
  - [Diffusion Language Models (<7B)](#diffusion-language-models-7b)
- [Multi-Modal Diffusion Models](#multi-modal-diffusion-models)
- [Seminal Diffusion Papers](#seminal-diffusion-papers)
    
---
## Survey Papers

| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759) | 2025 | Arxiv
| [Diffusion-based Large Language Models Survey](https://www.researchgate.net/profile/Junhao-Song-3/publication/394262235_Diffusion-based_Large_Language_Models_Survey/links/68901ee37b62e240dd32d2af/Diffusion-based-Large-Language-Models-Survey.pdf) | 2025 | Arxiv
| [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/pdf/2508.08712v2) | 2025 | Arxiv


## Large Diffusion Language Models (>7B)

### Scaling
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs](https://arxiv.org/abs/2305.14771) | 2023 | NAACL
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | 2023 | Arxiv |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) | 2025 | ACL | Adapted from Mistral-7B-v0.1 |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://openreview.net/forum?id=j1tSLYKwg8) | 2025 | ICLR | 127M~7B (GPT2, LLaMA2) |
| [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) | 2025 |  Arxiv | LLaDA-8B
| [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223) | 2025 |  Arxiv |
| [Large Language Models to Diffusion Finetuning](https://arxiv.org/abs/2501.15781) | 2025 | Arxiv |
| [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429) | 2025 | Arxiv | Long context scaling |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487v1)| 2025 | Arxiv |
| [UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models](https://arxiv.org/abs/2510.10481)| 2025 | Arxiv |

### AR-to-Diffusion Adaptation
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://openreview.net/forum?id=j1tSLYKwg8) | 2025 | ICLR | 127M~7B (GPT2, LLaMA2) |
| [SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation](https://arxiv.org/abs/2510.06303) | 2025 | Arxiv |
| [From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs](https://arxiv.org/abs/2512.06776) | 2025 | Arxiv |

### Accelerating

#### Caching
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/pdf/2505.21467) | 2025 |  Arxiv |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025 |  Arxiv |
| [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025 |  Arxiv |
| [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328)| 2025 | Arxiv | 
| [d^2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094)| 2025 | Arxiv | 
| [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/abs/2510.14973)| 2025 | Arxiv | 


#### Decoding
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413) | 2025 |  Arxiv |
| [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025 |  Arxiv |
| [Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](https://arxiv.org/pdf/2507.18578?) | 2025 | Arxiv |
| [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/pdf/2506.10848)| 2025 | Arxiv | 
| [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](https://arxiv.org/pdf/2509.26432)| 2025 | Arxiv | 
| [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328)| 2025 | Arxiv | 
| [Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding](https://arxiv.org/pdf/2509.18085)| 2025 | Arxiv | 
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488)| 2025 | Arxiv | 
| [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188)| 2025 | Arxiv | 
| [Self Speculative Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2510.04147)| 2025 | Arxiv | 
| [CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits](https://arxiv.org/abs/2510.06133)| 2025 | Arxiv | 
| [Accelerating Diffusion LLM Inference via Local Determinism Propagation](https://arxiv.org/abs/2510.07081)| 2025 | Arxiv | 
| [Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model](https://arxiv.org/abs/2510.18165)| 2025 | Arxiv | 
| [SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding](https://arxiv.org/abs/2511.00606)| 2025 | Arxiv | 
| [Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules](https://arxiv.org/abs/2512.02892)| 2025 | Arxiv | 

#### Distillation
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2410.21035)| 2025 | ICLR | 
| [FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](https://arxiv.org/abs/2509.20624)| 2025 | Arxiv | 
| [CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269)| 2025 | Arxiv | 

#### Sparsity
| [Attention Sinks in Diffusion Language Models](https://arxiv.org/abs/2510.15731)| 2025 | Arxiv | 
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014)| 2025 | Arxiv | 


### Reasoning & Alignment
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446) | 2025 |  Arxiv |
| [d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216) | 2025 |  Arxiv |
| [Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) | 2024 | NeurIPS |
| [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](https://arxiv.org/pdf/2507.08838) | 2025 | Arxiv |
| [Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs](https://arxiv.org/pdf/2508.10736) | 2025 | Arxiv |
| [Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation](https://arxiv.org/pdf/2507.08018v1) | 2025 | ICML |
| [Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models](https://arxiv.org/pdf/2509.06949) | 2025 | Arxiv |
| [DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning](https://arxiv.org/pdf/2510.02212) | 2025 | Arxiv |
| [Principled and Tractable RL for Reasoning with Diffusion Language Models](https://arxiv.org/pdf/2510.04019) | 2025 | Arxiv |
| [Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization](https://arxiv.org/pdf/2510.08554) | 2025 | Arxiv |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025 | Arxiv |
| [d2: Improved Techniques for Training Reasoning Diffusion Language Models](https://www.arxiv.org/abs/2509.21474) | 2025 | Arxiv |
| [Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step](https://arxiv.org/abs/2509.23924) | 2025 | Arxiv |
| [Inpainting-Guided Policy Optimization for Diffusion Large Language Models](https://arxiv.org/abs/2509.10396) | 2025 | Arxiv |
| [Beyond Surface Reasoning: Unveiling the True Long Chain-of-Thought Capacity of Diffusion Large Language Models](https://arxiv.org/abs/2510.09544) | 2025 | Arxiv |
| [Inpainting-Guided Policy Optimization for Diffusion Large Language Models](https://arxiv.org/abs/2509.10396) | 2025 | Arxiv |
| [Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models](https://arxiv.org/abs/2510.01544) | 2025 | Arxiv |
| [MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](https://arxiv.org/abs/2510.21473) | 2025 | Arxiv |
| [Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization](https://arxiv.org/abs/2510.08233) | 2025 | Arxiv |
| [Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models](https://arxiv.org/abs/2510.11683) | 2025 | Arxiv |
| [Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025 | Arxiv |
| [SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models](https://arxiv.org/abs/2510.09541) | 2025 | Arxiv |
| [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573) | 2025 | Arxiv |
| [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171) | 2025 | Arxiv |
| [MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models](https://arxiv.org/abs/2508.13148) | 2025 | Arxiv |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](https://arxiv.org/abs/2510.19304) | 2025 | Arxiv |
| [RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance](https://arxiv.org/abs/2509.25604) | 2025 | Arxiv |
| [Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295) | 2025 | Arxiv |

### Others
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) | 2025 |  Arxiv | Constrained decoding
| [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639) | 2025 | Arxiv | Coder
| [Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025 | Arxiv | Coder
| [Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models](https://arxiv.org/pdf/2508.09138v1) | 2025 | Arxiv | 
| [The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs](https://arxiv.org/pdf/2507.11097v1) | 2025 | Arxiv |
| [Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](https://arxiv.org/abs/2508.20072) | 2025 | Arxiv | VLA
| [LLaDA-VLA: Vision Language Diffusion Action Models](https://arxiv.org/abs/2509.06932) | 2025 | Arxiv | VLA
| [Beyond Autoregression: An Empirical Study of Diffusion Large Language Models for Code Generation](https://arxiv.org/abs/2509.11252) | 2025 | Arxiv | Coder
| [Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs](https://arxiv.org/pdf/2508.14896) | 2025 | Arxiv | Quantization
| [Sequential Diffusion Language Models](https://www.arxiv.org/abs/2509.24007) | 2025 | Arxiv |
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014) | 2025 | Arxiv | Sparse Attention
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025 | Arxiv | MoE
| [dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought](https://arxiv.org/pdf/2509.25681) | 2025 | Arxiv | VLA
| [Test-Time Anchoring for Discrete Diffusion Posterior Sampling](https://arxiv.org/pdf/2510.02291) | 2025 | Arxiv | Sampling
| [What Makes Diffusion Language Models Super Data Learners?](https://arxiv.org/pdf/2510.04071) | 2025 | Arxiv
| [Why mask diffusion does not work](https://arxiv.org/pdf/2510.03289) | 2025 | Arxiv
| [DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding](https://www.arxiv.org/pdf/2510.02358) | 2025 | Arxiv
| [Think While You Generate: Discrete Diffusion with Planned Denoising](https://arxiv.org/abs/2410.06264) | 2025 | ICLR
| [Diffusion Language Models Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) | 2025 | Arxiv
| [CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation](https://arxiv.org/abs/2505.14455) | 2025 | Arxiv
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632) | 2025 | Arxiv



## Diffusion Language Models (<7B)
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----:  
| [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) | 2022 | NeurIPS | Embedding |
| [DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models](https://arxiv.org/abs/2210.08933) | 2023 | ICLR | Embedding |
| [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://aclanthology.org/2023.acl-long.248.pdf) | 2023 | ACL | Masked |
| [Latent Diffusion for Language Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html) | 2023 | NeurIPS | Latent |
| [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://proceedings.mlr.press/v235/lou24a.html) | 2024 | ICML | Masked |
| [SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control](https://aclanthology.org/2023.acl-long.647.pdf) | 2023 | ACL | Simplex, Blockwise |
| [AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html) | 2023 | NeurIPS | AR-like noise |
| [Likelihood-Based Diffusion Language Models](https://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html) | 2023 | NeurIPS | Plaid1B
| [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) | 2024 | ICLR | 1.1B |
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) | 2025 | ICLR |
| [The Diffusion Duality](https://arxiv.org/abs/2506.10892) | 2025 | ICML |
| [Generalized Interpolating Discrete Diffusion](https://openreview.net/pdf?id=rvZv7sDPV9) | 2025 | ICML |
| [Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](https://arxiv.org/pdf/2502.06768) | 2025 | ICML |
| [Esoteric Language Models](https://arxiv.org/pdf/2506.01928) | 2025 |  Arxiv |
| [Reinforced Context Order Recovery for Adaptive Reasoning and Planning](https://arxiv.org/pdf/2508.13070) | 2025 | Arxiv | 
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/pdf/2410.14157) | 2025 | ICLR |
| [Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior](https://arxiv.org/pdf/2507.07586) | 2025 | ArXiv 
| [Any-Order Flexible Length Masked Diffusion](https://arxiv.org/pdf/2509.01025) | 2025 | Arxiv
| [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/pdf/2506.09018) | 2025 | Arxiv
| [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/pdf/2506.00290) | 2025 | Arxiv
| [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/pdf/2406.04329) | 2024 | NeurIPS
---

## Multi-Modal Diffusion Models

| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces](https://www.arxiv.org/abs/2506.07903)  | 2025 |  ICML |
| [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)  | 2025 |  Arxiv |
| [LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) | 2025 |  Arxiv |
| [Unified Multimodal Discrete Diffusion](https://arxiv.org/abs/2503.20853) | 2025 |  Arxiv |
| [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) | 2025 |  Arxiv |
| [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839) | 2025 |  Arxiv |
| [Dual Diffusion for Unified Image Generation and Understanding](https://arxiv.org/pdf/2501.00289) | 2025 |  Arxiv |
| [Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/pdf/2505.23606) | 2025 |  Arxiv |
| [Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564) | 2025 |  Arxiv |
| [Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding](https://arxiv.org/abs/2510.06308) | 2025 |  Arxiv |
| [MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation](https://arxiv.org/pdf/2511.09611) | 2025 |  Arxiv |

---
## Seminal Diffusion Papers

| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----:  
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) | 2015 |  ICML | Diffusion Formulation
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 |  NeurIPS |
| [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 2021 | ICLR |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | 2021 |  ICLR |
| [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) | 2022 | NeurIPS |
| [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 2022 | CVPR |
| [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 2023 |  ICCV |
| [Score-based Generative Modeling in Latent Space](https://arxiv.org/abs/2106.05931) | 2021 | NeurIPS | Latent |
| [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) | 2021 | NeurIPS | Discrete |
| [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822) | 2022 | CVPR | VQ |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 2021 | NeurIPS | CG |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2021 | NeurIPS | CFG |
| [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202) | 2023 | ICLR | Self-conditioning |
| [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) | 2022 | ICLR | Distillation |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023 | ICML | 

## Contact
<!-- **Contact** -->

We welcome all researchers to contribute to this repository.

If you have a related paper that was not added to the library, please contact us.

Email: jake630@snu.ac.kr / wjk9904@snu.ac.kr / qicher@snu.ac.kr

