<h1 align="center">
Meta-Learning Literature Overview 
</h1>

This repository contains a curated list of meta-learning papers closely related to my [PhD thesis](https://arxiv.org/abs/2406.09713). The research papers are primarily focused on optimization-based meta-learning approaches for learning loss functions, optimizers, and parameter initialization.

![meta-learning-banner](https://github.com/user-attachments/assets/cd132fac-0bbe-4694-9f3c-59be0f33da47)

## Table of Contents

- [Meta-Learning Survey Papers](#meta-learning-survey-papers)
- [Meta-Learning Loss Functions](#meta-learning-loss-functions)
- [Meta-Learning Optimizers](#meta-learning-optimizers)
- [Meta-Learning Parameter Initializations](#meta-learning-parameter-initializations)
- [Meta-Learning Miscellaneous](#meta-learning-miscellaneous)
- [Meta-Optimization](#meta-optimization)
- [Meta-Learning Blog Posts](#meta-learning-blog-posts)
- [Meta-Learning Libraries](#meta-learning-libraries)

## Meta-Learning Survey Papers

- A Perspective View and Survey of Meta-Learning. (_AIR2002_), [[paper](https://axon.cs.byu.edu/Dan/678/papers/Meta/Vilalta.pdf)].
- Meta-Learning: A Survey. (_arXiv2018_), [[paper](https://arxiv.org/abs/1810.03548)]
- A Comprehensive Overview and Survey of Recent Advances in Meta-Learning. (_arXiv2020_), [[paper](https://arxiv.org/abs/2004.11149)].
- Meta-Learning in Neural Networks: A Survey. (_TPAMI2022_), [[paper](https://arxiv.org/abs/2004.05439)].

## Meta-Learning Loss Functions

- Learning to Teach with Dynamic Loss Functions. (_NeurIPS2018_), [[paper](https://arxiv.org/abs/1810.12081)].
- Learning to Learn by Self-Critique. (_NeurIPS2019_), [[paper](https://arxiv.org/abs/1905.10295)].
- A General and Adaptive Robust Loss Function. (_CVPR2019_), [[paper](https://arxiv.org/abs/1701.03077)].
- Learning Surrogate Losses. (_arXiv2019_), [[paper](https://arxiv.org/abs/1905.10108)].
- Addressing the Loss-Metric Mismatch with Adaptive Loss Alignment. (_ICML2019_), [[paper](https://arxiv.org/abs/1905.05895)].
- AM-LFS: AutoML for Loss Function Search. (_ICCV2019_), [[paper](https://arxiv.org/abs/1905.07375)].
- Improved Training Speed, Accuracy, and Data Utilization Through Loss Function Optimization. (_CEC2020_), [[paper](https://arxiv.org/abs/1905.11528)].
- Effective Regularization through Loss Function Meta-Learning. (_arXiv2020_), [[paper](https://arxiv.org/abs/2010.00788)].
- Improving Deep Learning through Loss-Function Evolution. (_Thesis2020_), [[paper](https://www.cs.utexas.edu/~ai-lab/?gonzalez:diss20)].
- Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search. (_ICLR2020_), [[paper](https://arxiv.org/abs/2102.04700)].
- Learning State-Dependent Losses for Inverse Dynamics Learning. (_IROS2020_), [[paper](https://arxiv.org/abs/2003.04947)].
- Loss Function Search for Face Recognition. (_ICML2020_), [[paper](https://arxiv.org/abs/2007.06542)].
- Improving Generalization in Meta Reinforcement Learning using Learned Objectives. (_ICLR2020_), [[paper](https://arxiv.org/abs/1910.04098)].
- Meta-Learning via Learned Loss. (_ICPR2021_), [[paper](https://arxiv.org/abs/1906.05374)].
- Searching for Robustness: Loss Learning for Noisy Classification Tasks. (_ICCV2021_), [[paper](https://arxiv.org/abs/2103.00243)].
- Optimizing Loss Functions through Multi-Variate Taylor Polynomial Parameterization. (_GECCO2021_), [[paper](https://arxiv.org/abs/2002.00059)].
- Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning. (_ICCV2021_), [[paper](https://arxiv.org/abs/2110.03909)].
- Loss Function Learning for Domain Generalization by Implicit Gradient. (_ICML2022_), [[paper](https://proceedings.mlr.press/v162/gao22b/gao22b.pdf)].
- AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks. (_CVPR2022_), [[paper](https://arxiv.org/abs/2103.14026)].
- PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions. (_ICLR2022_), [[paper](https://arxiv.org/abs/2204.12511)].
- Meta-Learning PINN Loss Functions. (_JCP2022_), [[paper](https://arxiv.org/abs/2107.05544)].
- Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning. (_TPAMI2023_), [[paper](https://arxiv.org/abs/2209.08907)].
- Fast and Efficient Local-Search for Genetic Programming Based Loss Function Learning. (_GECCO2023_), [[paper](https://arxiv.org/abs/2403.00865)].
- Online Loss Function Learning. (_arXiv2023_), [[paper](https://arxiv.org/abs/2301.13247)].
- Meta-Learning to Optimise: Loss Functions and Update Rules. (_Thesis2023_), [[paper](https://era.ed.ac.uk/handle/1842/39821)].
- OWAdapt: An Adaptive Loss Function For Deep Learning using OWA Operators. (_arXiv2023_), [[paper](https://arxiv.org/abs/2305.19443)].
- Neural Loss Function Evolution for Large-Scale Image Classifier Convolutional Neural Networks. (_arXiv2024_), [[paper](https://arxiv.org/abs/2403.08793)].
- Evolving Loss Functions for Specific Image Augmentation Techniques. (_arXiv2024_), [[paper](https://arxiv.org/abs/2404.06633)].
- Meta-Learning Loss Functions for Deep Neural Networks. (_Thesis2024_), [[paper](https://arxiv.org/abs/2406.09713)].

## Meta-Learning Optimizers

- Learning to Learn by Gradient Descent by Gradient Descent. (_NeurIPS2016_), [[paper](https://arxiv.org/abs/1606.04474)].
- Meta-SGD: Learning to Learn Quickly for Few-Shot Learning. (_arXiv2017_), [[paper](https://arxiv.org/abs/1707.09835)].
- Learning to Learn Without Gradient Descent by Gradient Descent. (_ICML2017_), [[paper](https://arxiv.org/abs/1611.03824)].
- Learned Optimizers that Scale and Generalize. (_ICML2017_), [[paper](https://arxiv.org/abs/1703.04813)].
- Optimization as a Model for Few-Shot Learning. (_ICLR2017_), [[paper](https://openreview.net/pdf?id=rJY0-Kcll)].
- Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace. (_ICML2018_), [[paper](https://arxiv.org/abs/1801.05558)].
- Meta-Learning with Adaptive Layerwise Metric and Subspace. (_ICML2018_), [[paper](https://arxiv.org/abs/1801.05558)].
- Meta-Curvature. (_NeurIPS2019_), [[paper](https://arxiv.org/abs/1902.03356)].
- Understanding and Correcting Pathologies in the Training of Learned Optimizers. (_ICML2019_), [[paper](https://arxiv.org/abs/1810.10180)].
- Meta-Learning with Warped Gradient Descent. (_ICLR2020_), [[paper](https://arxiv.org/abs/1909.00025)].
- On Modulating the Gradient for Meta-Learning. (_ECCV2020_), [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530545.pdf)].
- Tasks, Stability, Architecture, and Compute: Training More Effective Learned Optimizers, and Using Them to Train Themselves. (_arXiv2020_), [[paper](https://arxiv.org/abs/2009.11243)].
- Learning to Optimize: A Primer and a Benchmark. (_JMLR2022_), [[paper](https://arxiv.org/abs/2103.12828)].
- VeLO: Training Versatile Learned Optimizers by Scaling Up. (_arXiv2022_), [[paper](https://arxiv.org/abs/2211.09760)]
- Practical Tradeoffs Between Memory, Compute, and Performance in Learned Optimizers. (_CoLLA2002_), [[paper](https://arxiv.org/abs/2203.11860)].
- Meta-Learning with a Geometry-Adaptive Preconditioner. (_CVPR2023_), [[paper](https://arxiv.org/abs/2304.01552)].
- Symbolic Discovery of Optimization Algorithms. (_arXiv2023_), [[paper](https://arxiv.org/abs/2302.06675)].

## Meta-Learning Parameter Initializations

- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. (_ICML2017_), [[paper](https://arxiv.org/abs/1703.03400)].
- On First-Order Meta-Learning Algorithms. (_arXiv2018_), [[paper](https://arxiv.org/abs/1803.02999)].
- Probabilistic Model-Agnostic Meta-Learning. (_NeurIPS2018_), [[paper](https://arxiv.org/abs/1806.02817)].
- Toward Multimodal Model-Agnostic Meta-Learning. (_NeurIPS2018_), [[paper](https://arxiv.org/abs/1812.07172)].
- Meta-Learning with Implicit Gradients. (_NeurIPS2019_), [[paper](https://arxiv.org/abs/1909.04630)].
- Alpha MAML: Adaptive Model-Agnostic Meta-Learning. (_arXiv2019_), [[paper](https://arxiv.org/abs/1905.07435)].
- How to Train Your MAML. (_ICLR2019_), [[paper](https://arxiv.org/abs/1810.09502)].
- Meta-Learning with Latent Embedding Optimization. (_ICLR2019_), [[paper](https://arxiv.org/abs/1807.05960)].
- Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation. (_NeurIPS2019_), [[paper](https://arxiv.org/abs/1910.13616)].
- Fast Context Adaptation via Meta-Learning. (_ICML2019_), [[paper](https://arxiv.org/abs/1810.03642)].
- Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML. (_arXiv2019_), [[paper](https://arxiv.org/abs/1909.09157)].
- ES-MAML: Simple Hessian-Free Meta-Learning. (_ICLR2020_), [[paper](https://arxiv.org/abs/1910.01215)].
- BOIL: Towards Representation Change for Few-Shot Learning. (_arXiv2020_), [[paper](https://arxiv.org/abs/2008.08882)].
- How to Train Your MAML to Excel in Few-Shot Classification. (_ICLR2021_), [[paper](https://arxiv.org/abs/2106.16245)].
- Meta-Learning Neural Procedural Biases. (arXiv2024), [[paper](https://arxiv.org/abs/2406.07983)].

## Meta-Learning Miscellaneous

- Siamese Neural Networks for One-Shot Image Recognition. (_ICML2015_), [[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)].
- Matching Networks for One-Shot Learning. (_NeurIPS2016_), [[paper](https://arxiv.org/abs/1606.04080)].
- Meta-Learning with Memory-Augmented Neural Networks. (_ICML2016_), [[paper](https://proceedings.mlr.press/v48/santoro16.pdf)].
- Prototypical Networks for Few-Shot Learning. (_NeurIPS2017_), [[paper](https://arxiv.org/abs/1703.05175)].
- Searching for Activation Functions. (_arXiv2017_), [[paper](https://arxiv.org/abs/1710.05941)].
- Learning to Learn: Meta-Critic Networks for Sample Efficient Learning. (_arXiv2017_), [[paper](https://arxiv.org/abs/1706.09529)].
- Meta-Learning with Differentiable Closed-Form Solvers. (_arXiv2018_), [[paper](https://arxiv.org/abs/1805.08136)].
- Learning to Reweight Examples for Robust Deep Learning. (_ICML2018_), [[paper](https://arxiv.org/abs/1803.09050)].
- Learning to Compare: Relation Network for Few-Shot Learning. (_CVPR2018_), [[paper](https://arxiv.org/abs/1711.06025)].
- Online Learning Rate Adaptation with Hypergradient Descent. (_ICLR2018_), [[paper](https://arxiv.org/abs/1703.04782)].
- TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning. (_NeurIPS2018_), [[paper](https://arxiv.org/abs/1805.10123)].
- MetaReg: Towards Domain Generalization using Meta-Regularization. (_NeurIPS2018_), [[paper](https://papers.nips.cc/paper_files/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html)].
- Learning to Learn with Conditional Class Dependencies. (_ICLR2018_), [[paper](https://openreview.net/forum?id=BJfOXnActQ)].
- Few-Shot Image Recognition by Predicting Parameters from Activations. (_CVPR2018_), [[paper](https://arxiv.org/abs/1706.03466)].
- Fast and Flexible Multi-Task Classification using Conditional Neural Adaptive Processes. (_NeurIPS2019_), [[paper](https://arxiv.org/abs/1906.07697)].
- Meta-Learning for Semi-Supervised Few-Shot Classification. (_ICLR2019_), [[paper](https://arxiv.org/abs/1803.00676)].
- Meta-Learning with Differentiable Convex Optimization. (_CVPR2019_), [[paper](https://arxiv.org/abs/1904.03758)].
- AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. (_ICML2020_), [[paper](https://arxiv.org/abs/2003.03384)].
- Evolving Normalization-Activation Layers. (_NeurIPS2020_), [[paper](https://arxiv.org/abs/2004.02967)].
- Meta-Learning with Adaptive Hyperparameters. (_NeurIPS2020_), [[paper](https://arxiv.org/abs/2011.00209)].
- Differentiable Automatic Data Augmentation. (_ECCV2020_), [[paper](https://arxiv.org/abs/2003.03780)].
- Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. (_CVPR2020_), [[paper](https://arxiv.org/abs/1812.03664)].
- Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples. (_ICLR2020_), [[paper](https://arxiv.org/abs/1903.03096)].
- Evolving Reinforcement Learning Algorithms. (_ICLR2021_), [[paper](https://arxiv.org/abs/2101.03958)].
- Learning to Learn Task-Adaptive Hyperparameters for Few-Shot Learning. (_TPAMI2023_), [[paper](https://ieeexplore.ieee.org/abstract/document/10080995)].

## Meta-Optimization

- Gradient-Based Hyperparameter Optimization through Reversible Learning. (_ICML2015_), [[paper](https://arxiv.org/abs/1502.03492)].
- Forward and Reverse Gradient-based Hyperparameter Optimization. (_ICML2017_), [[paper](https://arxiv.org/abs/1703.01785)].
- Bilevel Programming for Hyperparameter Optimization and Meta-Learning. (_ICML2018_), [[paper](https://arxiv.org/abs/1806.04910)].
- Understanding Short-Horizon Bias in Stochastic Meta-Optimization. (_ICLR2018_), [[paper](https://arxiv.org/abs/1803.02021)].
- Generalized Inner Loop Meta-Learning. (_arXiv2019_), [[paper](https://arxiv.org/abs/1910.01727)].
- Transferring Knowledge Across Learning Processes. (_ICLR2019_), [[paper](https://arxiv.org/abs/1812.01054)].
- Truncated Back-propagation for Bilevel Optimization. (_AISTATS2019_), [[paper](https://arxiv.org/abs/1810.10667)].
- Optimizing Millions of Hyperparameters by Implicit Differentiation. (_AISTATS2020_), [[paper](https://arxiv.org/abs/1911.02590)].
- EvoGrad: Efficient Gradient-Based Meta-Learning and Hyperparameter Optimization. (_NeurIPS2021_), [[paper](https://arxiv.org/abs/2106.10575)].
- Gradient-Based Bi-level Optimization for Deep Learning: A Survey. (_arXiv2022_), [[paper](https://arxiv.org/abs/2207.11719)].
- The Curse of Unrolling: Rate of Differentiating through Optimization. (_NeurIPS2022_), [[paper](https://arxiv.org/abs/2209.13271)].
- Bootstrapped Meta-Learning. (_ICLR2022_), [[paper](https://arxiv.org/abs/2109.04504)].
- Optimistic Meta-Gradients. (_NeurIPS2024_), [[paper](https://arxiv.org/abs/2301.03236)].

## Meta-Learning Blog Posts

- Learning to Learn [[link](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)]
- Meta-Learning: Learning to Learn Fast [[link](https://lilianweng.github.io/posts/2018-11-30-meta-learning/)]
- An Interactive Introduction to Model-Agnostic Meta-Learning [[link](https://interactive-maml.github.io)]
- Reptile: A Scalable Meta-Learning Algorithm [[link](https://openai.com/index/reptile/)]

## Meta-Learning Libraries

- Higher [[link](https://github.com/facebookresearch/higher)].
- TorchMeta [[link](https://github.com/tristandeleu/pytorch-meta)].
- Learn2learn [[link](https://github.com/learnables/learn2learn)].
- TorchOpt [[link](https://github.com/metaopt/torchopt)].
