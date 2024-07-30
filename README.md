<h1 align="center">
Meta-Learning Literature Overview
</h1>

This repository contains a curated list of meta-learning papers closely related to my PhD research. The research papers are primarily focused on optimization-based meta-learning approaches for learning loss functions, optimizers, and parameter initialization.

**Table of Contents**

- [Meta-Learning Survey Papers](#meta-learning-survey-papers)
- [Meta-Learning Loss Functions](#meta-learning-loss-functions)
- [Meta-Learning Optimizers](#meta-learning-optimizers)
- [Meta-Learning Parameter Initializations](#meta-learning-parameter-initializations)
- [Meta-Learning Miscellaneous](#meta-learning-miscellaneous)
- [Meta-Optimization](#meta-optimization)

## Meta-Learning Survey Papers

- A Perspective View and Survey of Meta-Learning. (_AIR2002_), [[paper]()].
- Meta-Learning: A Survey. (_arXiv2018_), [[paper]()]
- A Comprehensive Overview and Survey of Recent Advances in Meta-Learning. (_arXiv2020_), [[paper](https://arxiv.org/abs/2004.11149)].
- Meta-Learning in Neural Networks: A Survey. (_TPAMI2022_), [[paper](https://arxiv.org/abs/2004.05439)].

## Meta-Learning Loss Functions

- A General and Adaptive Robust Loss Function. (_CVPR2019_), [[paper]()].
- Improved Training Speed, Accuracy, and Data Utilization Through Loss Function Optimization. (_CEC2020_), [[paper]()].
- Meta-Learning via Learned Loss. (_ICPR2021_), [[paper]()].
- Searching for Robustness: Loss Learning for Noisy Classification Tasks. (_ICCV2021_), [[paper]()].
- Optimizing Loss Functions through Multi-Variate Taylor Polynomial Parameterization. (_GECCO2021_), [[paper]()].
- Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search. (_ICLR2020_), [[paper]()].
- AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks. (_CVPR2022_), [[paper]()].
- Learning Surrogate Losses. (_arXiv2019_), [[paper]()].
- Addressing the Loss-Metric Mismatch with Adaptive Loss Alignment. (_ICML2019_), [[paper]()].
- Learning State-Dependent Losses for Inverse Dynamics Learning. (_IROS2020_), [[paper]()].
- Effective Regularization through Loss Function Meta-Learning. (_arXiv2020_), [[paper]()].
- Loss Function Learning for Domain Generalization by Implicit Gradient. (_ICML2022_), [[paper]()].
- PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions. (_ICLR2022_), [[paper]()].
- Online Loss Function Learning. (_arXiv2023_), [[paper]()].
- Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning. (_ICCV2021_), [[paper]()].
- Learning to Teach with Dynamic Loss Functions. (_NeurIPS2018_), [[paper]()].
- Improving Generalization in Meta Reinforcement Learning using Learned Objectives. (_ICLR2020_), [[paper]()].
- Meta-Learning PINN Loss Functions. (_JCP2022_), [[paper]()].
- Learning to Learn by Self-Critique. (_NeurIPS2019_), [[paper]()].
- Meta-Learning to Optimise: Loss Functions and Update Rules. (_Thesis2023_), [[paper]()].
- Improving Deep Learning through Loss-Function Evolution. (_Thesis2020_), [[paper]()].
- Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning. (_TPAMI2023_), [[paper]()].
- Fast and Efficient Local-Search for Genetic Programming Based Loss Function Learning. (_GECCO2023_), [[paper]()].
- AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks. (_CVPR2022_), [[paper]()].
- AM-LFS: AutoML for Loss Function Search. (_ICCV2019_), [[paper]()].
- Loss Function Search for Face Recognition. (_ICML2020_), [[paper]()].
- Neural Loss Function Evolution for Large-Scale Image Classifier Convolutional Neural Networks. (_arXiv2024_), [[paper]()].
- Evolving Loss Functions for Specific Image Augmentation Techniques. (_arXiv2024_), [[paper]()].

## Meta-Learning Optimizers

@inproceedings{andrychowicz2016learning,
  title={Learning to Learn by Gradient Descent by Gradient Descent},
  author={Andrychowicz, Marcin and Denil, Misha and Gomez, Sergio and Hoffman, Matthew W and Pfau, David and Schaul, Tom and Shillingford, Brendan and De Freitas, Nando},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2016}
}

@article{li2017meta,
  title={Meta-SGD: Learning to Learn Quickly for Few-Shot Learning},
  author={Li, Zhenguo and Zhou, Fengwei and Chen, Fei and Li, Hang},
  journal={arXiv preprint arXiv:1707.09835},
  year={2017}
}

@inproceedings{ravi2017optimization,
  title={Optimization as a Model for Few-Shot Learning},
  author={Ravi, Sachin and Larochelle, Hugo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@article{flennerhag2019meta,
  title={Meta-Learning with Warped Gradient Descent},
  author={Flennerhag, Sebastian and Rusu, Andrei A and Pascanu, Razvan and Visin, Francesco and Yin, Hujun and Hadsell, Raia},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@inproceedings{wichrowska2017learned,
  title={Learned Optimizers that Scale and Generalize},
  author={Wichrowska, Olga and Maheswaranathan, Niru and Hoffman, Matthew W and Colmenarejo, Sergio Gomez and Denil, Misha and Freitas, Nando and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@inproceedings{lee2018gradient,
  title={Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace},
  author={Lee, Yoonho and Choi, Seungjin},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@article{park2019meta,
  title={Meta-Curvature},
  author={Park, Eunbyung and Oliva, Junier B},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@article{chen2022learning,
  title={Learning to Optimize: A Primer and a Benchmark},
  author={Chen, Tianlong and Chen, Xiaohan and Chen, Wuyang and Heaton, Howard and Liu, Jialin and Wang, Zhangyang and Yin, Wotao},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2022}
}

@inproceedings{chen2017learning,
  title={Learning to Learn Without Gradient Descent by Gradient Descent},
  author={Chen, Yutian and Hoffman, Matthew W and Colmenarejo, Sergio G{\'o}mez and Denil, Misha and Lillicrap, Timothy P and Botvinick, Matt and Freitas, Nando},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@article{lee2018meta,
  title={Meta-Learning with Adaptive Layerwise Metric and Subspace},
  author={Yoonho Lee and Seungjin Choi},
  journal={International Conference on Machine Learning (ICML)},
  year={2018}
}

@inproceedings{metz2019understanding,
  title={Understanding and Correcting Pathologies in the Training of Learned Optimizers},
  author={Metz, Luke and Maheswaranathan, Niru and Nixon, Jeremy and Freeman, Daniel and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}


@inproceedings{simon2020modulating,
  title={On Modulating the Gradient for Meta-Learning},
  author={Simon, Christian and Koniusz, Piotr and Nock, Richard and Harandi, Mehrtash},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020},
}

@article{flennerhag2020meta,
  title={Meta-Learning with Warped Gradient Descent},
  author={Flennerhag, Sebastian and Rusu, Andrei A and Pascanu, Razvan and Visin, Francesco and Yin, Hujun and Hadsell, Raia},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@inproceedings{kang2023meta,
  title={Meta-Learning with a Geometry-Adaptive Preconditioner},
  author={Kang, Suhyun and Hwang, Duhun and Eo, Moonjung and Kim, Taesup and Rhee, Wonjong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}


## Meta-Learning Parameter Initializations

@inproceedings{finn2017model,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@article{nichol2018first,
  title={On First-Order Meta-Learning Algorithms},
  author={Nichol, Alex and Achiam, Joshua and Schulman, John},
  journal={arXiv preprint arXiv:1803.02999},
  year={2018}
}

@article{rajeswaran2019meta,
  title={Meta-Learning with Implicit Gradients},
  author={Rajeswaran, Aravind and Finn, Chelsea and Kakade, Sham M and Levine, Sergey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={32},
  year={2019}
}

@article{song2019maml,
  title={ES-MAML: Simple Hessian-Free Meta Learning},
  author={Song, Xingyou and Gao, Wenbo and Yang, Yuxiang and Choromanski, Krzysztof and Pacchiano, Aldo and Tang, Yunhao},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@article{nichol2018reptile,
  title={Reptile: A Scalable Meta-Learning Algorithm},
  author={Nichol, Alex and Schulman, John},
  journal={arXiv preprint arXiv:1803.02999},
  year={2018}
}

@article{behl2019alpha,
  title={Alpha MAML: Adaptive Model-Agnostic Meta-Learning},
  author={Behl, Harkirat Singh and Baydin, At{\i}l{\i}m G{\"u}ne{\c{s}} and Torr, Philip HS},
  journal={arXiv preprint arXiv:1905.07435},
  year={2019}
}

@article{antoniou2019train,
  title={How to Train Your MAML},
  author={Antoniou, Antreas and Edwards, Harrison and Storkey, Amos},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@inproceedings{rusu2019metalearning,
  title={Meta-Learning with Latent Embedding Optimization},
  author={Andrei A. Rusu and Dushyant Rao and Jakub Sygnowski and Oriol Vinyals and Razvan Pascanu and Simon Osindero and Raia Hadsell},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
}

@article{vuorio2019multimodal,
  title={Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation},
  author={Vuorio, Risto and Sun, Shao-Hua and Hu, Hexiang and Lim, Joseph J},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@inproceedings{zintgraf2019fast,
  title={Fast Context Adaptation via Meta-Learning},
  author={Zintgraf, Luisa and Shiarli, Kyriacos and Kurin, Vitaly and Hofmann, Katja and Whiteson, Shimon},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}

@inproceedings{ye2021unicorn,
  title={How to Train Your MAML to Excel in Few-Shot Classification},
  author={Han-Jia Ye and Wei-Lun Chao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{oh2020boil,
  title={BOIL: Towards Representation Change for Few-Shot Learning},
  author={Oh, Jaehoon and Yoo, Hyungjun and Kim, ChangHwan and Yun, Se-Young},
  journal={arXiv preprint arXiv:2008.08882},
  year={2020}
}

@article{raghu2019rapid,
  title={Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML},
  author={Raghu, Aniruddh and Raghu, Maithra and Bengio, Samy and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1909.09157},
  year={2019}
}

@article{raymond2024meta,
  title={Meta-Learning Neural Procedural Biases},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  journal={arXiv preprint arXiv:2406.07983},
  year={2024}
}

## Meta-Learning Miscellaneous

@article{ramachandran2017searching,
  title={Searching for Activation Functions},
  author={Ramachandran, Prajit and Zoph, Barret and Le, Quoc V},
  journal={arXiv preprint arXiv:1710.05941},
  year={2017}
}

@inproceedings{real2020automl,
  title={AutoML-Zero: Evolving Machine Learning Algorithms From Scratch},
  author={Real, Esteban and Liang, Chen and So, David and Le, Quoc},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}

@article{co2021evolving,
  title={Evolving Reinforcement Learning Algorithms},
  author={Co-Reyes, John D and Miao, Yingjie and Peng, Daiyi and Real, Esteban and Levine, Sergey and Le, Quoc V and Lee, Honglak and Faust, Aleksandra},
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{liu2020evolving,
  title={Evolving Normalization-Activation Layers},
  author={Liu, Hanxiao and Brock, Andy and Simonyan, Karen and Le, Quoc},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{stanley2002evolving,
  title={Evolving Neural Networks through Augmenting Topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary Computation},
  year={2002},
  publisher={MIT Press}
}

@article{balaji2018metareg,
  title={MetaReg: Towards Domain Generalization using Meta-Regularization},
  author={Balaji, Yogesh and Sankaranarayanan, Swami and Chellappa, Rama},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

@inproceedings{baydin2018hypergradient,
  title={Online Learning Rate Adaptation with Hypergradient Descent},
  author={Baydin, Atılım Güneş and Cornish, Robert and Rubio, David Martínez and Schmidt, Mark and Wood, Frank},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@article{snell2017prototypical,
  title={Prototypical Networks for Few-Shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@inproceedings{koch2015siamese,
  title={Siamese Neural Networks for One-Shot Image Recognition},
  author={Koch, Gregory and Zemel, Richard and Salakhutdinov, Ruslan and others},
  booktitle={International Conference on Machine Learning (ICML) Deep Learning Workshop},
  year={2015}
}

@article{sung2017learning,
  title={Learning to Learn: Meta-Critic Networks for Sample Efficient Learning},
  author={Sung, Flood and Zhang, Li and Xiang, Tao and Hospedales, Timothy and Yang, Yongxin},
  journal={arXiv preprint arXiv:1706.09529},
  year={2017}
}

@article{baik2020meta,
  title={Meta-Learning with Adaptive Hyperparameters},
  author={Baik, Sungyong and Choi, Myungsub and Choi, Janghoon and Kim, Heewon and Lee, Kyoung Mu},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{baik2023meta,
  title={Learning to Learn Task-Adaptive Hyperparameters for Few-Shot Learning}, 
  author={Baik, Sungyong and Choi, Myungsub and Choi, Janghoon and Kim, Heewon and Lee, Kyoung Mu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
  year={2023},
  organization={IEEE}
}

@inproceedings{sung2018learning,
  title={Learning to Compare: Relation Network for Few-Shot Learning},
  author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}

@article{oreshkin2018tadam,
  title={TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning},
  author={Oreshkin, Boris and Rodr{\'\i}guez L{\'o}pez, Pau and Lacoste, Alexandre},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

@inproceedings{jiang2018learning,
  title={Learning to Learn with Conditional Class Dependencies},
  author={Jiang, Xiang and Havaei, Mohammad and Varno, Farshid and Chartrand, Gabriel and Chapados, Nicolas and Matwin, Stan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@article{requeima2019fast,
  title={Fast and Flexible Multi-Task Classification using Conditional Neural Adaptive Processes},
  author={Requeima, James and Gordon, Jonathan and Bronskill, John and Nowozin, Sebastian and Turner, Richard E},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@inproceedings{qiao2018few,
  title={Few-Shot Image Recognition by Predicting Parameters from Activations},
  author={Qiao, Siyuan and Liu, Chenxi and Shen, Wei and Yuille, Alan L},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}

@inproceedings{ye2020few,
  title={Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions},
  author={Ye, Han-Jia and Hu, Hexiang and Zhan, De-Chuan and Sha, Fei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

@inproceedings{triantafillou2020meta,
  title={Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples},
  author={Eleni Triantafillou and Tyler Zhu and Vincent Dumoulin and Pascal Lamblin and Utku Evci and Kelvin Xu and Ross Goroshin and Carles Gelada and Kevin Jordan Swersky and Pierre-Antoine Manzagol and Hugo Larochelle},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@article{vinyals2016matching,
  title={Matching Networks for One-Shot Learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Wierstra, Daan and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2016}
}

@inproceedings{santoro2016meta,
  title={Meta-Learning with Memory-Augmented Neural Networks},
  author={Santoro, Adam and Bartunov, Sergey and Botvinick, Matthew and Wierstra, Daan and Lillicrap, Timothy},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2016}
}

@inproceedings{ren18fewshotssl,
  title={Meta-Learning for Semi-Supervised Few-Shot Classification},
  author={Mengye Ren and Eleni Triantafillou and Sachin Ravi and Jake Snell and Kevin Swersky and Joshua B. Tenenbaum and Hugo Larochelle and Richard S. Zemel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018},
}

@article{bertinetto2018meta,
  title={Meta-Learning with Differentiable Closed-Form Solvers},
  author={Bertinetto, Luca and Henriques, Joao F and Torr, Philip HS and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:1805.08136},
  year={2018}
}

@inproceedings{ren2018learning,
  title={Learning to Reweight Examples for Robust Deep Learning},
  author={Ren, Mengye and Zeng, Wenyuan and Yang, Bin and Urtasun, Raquel},
  booktitle={International Conference on Machine Learning},
  year={2018},
}

@inproceedings{li2020differentiable,
  title={Differentiable Automatic Data Augmentation},
  author={Li, Yonggang and Hu, Guosheng and Wang, Yongtao and Hospedales, Timothy and Robertson, Neil M and Yang, Yongxin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020},
  organization={Springer}
}

@inproceedings{lee2019meta,
  title={Meta-Learning with Differentiable Convex Optimization},
  author={Lee, Kwonjoon and Maji, Subhransu and Ravichandran, Avinash and Soatto, Stefano},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

## Meta-Optimization

@inproceedings{maclaurin2015gradient,
  title={Gradient-Based Hyperparameter Optimization through Reversible Learning},
  author={Maclaurin, Dougal and Duvenaud, David and Adams, Ryan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2015}
}

@article{grefenstette2019generalized,
  title={Generalized Inner Loop Meta-Learning},
  author={Grefenstette, Edward and Amos, Brandon and Yarats, Denis and Htut, Phu Mon and Molchanov, Artem and Meier, Franziska and Kiela, Douwe and Cho, Kyunghyun and Chintala, Soumith},
  journal={arXiv preprint arXiv:1910.01727},
  year={2019}
}

@article{chen2022gradient,
  title={Gradient-Based Bi-level Optimization for Deep Learning: A Survey},
  author={Chen, Can and Chen, Xi and Ma, Chen and Liu, Zixuan and Liu, Xue},
  journal={arXiv preprint arXiv:2207.11719},
  year={2022}
}

@inproceedings{franceschi2017forward,
  title={Forward and Reverse Gradient-based Hyperparameter Optimization},
  author={Franceschi, Luca and Donini, Michele and Frasconi, Paolo and Pontil, Massimiliano},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@inproceedings{shaban2019truncated,
  title={Truncated Back-propagation for Bilevel Optimization},
  author={Shaban, Amirreza and Cheng, Ching-An and Hatch, Nathan and Boots, Byron},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2019}
}

@article{scieur2022curse,
  title={The Curse of Unrolling: Rate of Differentiating through Optimization},
  author={Scieur, Damien and Gidel, Gauthier and Bertrand, Quentin and Pedregosa, Fabian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{lorraine2020optimizing,
  title={Optimizing Millions of Hyperparameters by Implicit Differentiation},
  author={Lorraine, Jonathan and Vicol, Paul and Duvenaud, David},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2020}
}

@inproceedings{franceschi2018bilevel,
  title={Bilevel Programming for Hyperparameter Optimization and Meta-Learning},
  author={Franceschi, Luca and Frasconi, Paolo and Salzo, Saverio and Grazzi, Riccardo and Pontil, Massimiliano},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@article{wu2018understanding,
  title={Understanding Short-Horizon Bias in Stochastic Meta-Optimization},
  author={Wu, Yuhuai and Ren, Mengye and Liao, Renjie and Grosse, Roger},
  journal={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@article{flennerhag2018transferring,
  title={Transferring Knowledge Across Learning Processes},
  author={Flennerhag, Sebastian and Moreno, Pablo G and Lawrence, Neil D and Damianou, Andreas},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@article{flennerhag2024optimistic,
  title={Optimistic Meta-Gradients},
  author={Flennerhag, Sebastian and Zahavy, Tom and O'Donoghue, Brendan and van Hasselt, Hado P and Gy{\"o}rgy, Andr{\'a}s and Singh, Satinder},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@article{bohdal2021evograd,
  title={Evograd: Efficient Gradient-Based Meta-Learning and Hyperparameter Optimization},
  author={Bohdal, Ondrej and Yang, Yongxin and Hospedales, Timothy},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

