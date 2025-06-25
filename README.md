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
- [What are Diffusion Language Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)  

---

## ⚙️ Framework
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
## Diffusion Language Models

### Large Diffusion Language Models (>7B)

#### Scaling
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs](https://arxiv.org/abs/2305.14771) | 2023 | NAACL
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | 2023 | Arxiv |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) | 2025 | ACL | Adapted from Mistral-7B-v0.1 |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://openreview.net/forum?id=j1tSLYKwg8) | 2025 | ICLR | 127M~7B (GPT2, LLaMA2) |
| [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) | 2025 |  Arxiv | LLaDA-8B
| [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223) | 2025 |  Arxiv |
| [Large Language Models to Diffusion Finetuning](https://arxiv.org/abs/2501.15781) | 2025 | Arxiv |
| [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429) | 2025 | Arxiv | Long context scaling

#### Accelerating
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413) | 2025 |  Arxiv |
| [Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/pdf/2505.21467) | 2025 |  Arxiv |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025 |  Arxiv |
| [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025 |  Arxiv |

#### Reasoning
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446) | 2025 |  Arxiv |
| [d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216) | 2025 |  Arxiv |
| [Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) | 2024 | NeurIPS |

#### Others
| **Paper Title** | **Year** | **Conference/Journal** | **Remark** |
| --------------- | :----: | :----: | :----: |
| [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) | 2025 |  Arxiv | Constrained decoding

### Diffusion Language Models (<7B)
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
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) | 2025 |  ICLR |
| [The Diffusion Duality](https://arxiv.org/abs/2506.10892) | 2025 |  ICML |

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

Email: jake630@snu.ac.kr / wjk9904@snu.ac.kr
