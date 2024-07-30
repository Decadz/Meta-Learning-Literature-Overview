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

- [Meta-Learning in Neural Networks: A Survey (TPAMI2022)](https://arxiv.org/abs/2004.05439)

@article{peng2020comprehensive,
  title={A Comprehensive Overview and Survey of Recent Advances in Meta-Learning},
  author={Peng, Huimin},
  journal={arXiv preprint arXiv:2004.11149},
  year={2020}
}

@article{vilalta2002perspective,
  title={A Perspective View and Survey of Meta-Learning},
  author={Vilalta, Ricardo and Drissi, Youssef},
  journal={Artificial Intelligence Review},
  year={2002},
  publisher={Springer}
}

@article{vanschoren2018meta,
  title={Meta-Learning: A Survey},
  author={Vanschoren, Joaquin},
  journal={arXiv preprint arXiv:1810.03548},
  year={2018}
}

## Meta-Learning Loss Functions

@inproceedings{gonzalez2020improved,
  title={Improved Training Speed, Accuracy, and Data Utilization Through Loss Function Optimization},
  author={Gonzalez, Santiago and Miikkulainen, Risto},
  booktitle={IEEE Congress on Evolutionary Computation (CEC)},
  year={2020}
}

@inproceedings{bechtle2021meta,
  title={Meta-Learning via Learned Loss},
  author={Bechtle, Sarah and Molchanov, Artem and Chebotar, Yevgen and Grefenstette, Edward and Righetti, Ludovic and Sukhatme, Gaurav and Meier, Franziska},
  booktitle={International Conference on Pattern Recognition},
  year={2021},
  organization={IEEE}
}

@inproceedings{gao2021searching,
  title={Searching for Robustness: Loss Learning for Noisy Classification Tasks},
  author={Gao, Boyan and Gouk, Henry and Hospedales, Timothy M},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

@inproceedings{gonzalez2021optimizing,
  title={Optimizing Loss Functions through Multi-Variate Taylor Polynomial Parameterization},
  author={Gonzalez, Santiago and Miikkulainen, Risto},
  booktitle={ACM Genetic and Evolutionary Computation Conference (GECCO)},
  year={2021}
}

@inproceedings{liu2021loss,
  title={Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search},
  author={Liu, Peidong and Zhang, Gengwei and Wang, Bochao and Xu, Hang and Liang, Xiaodan and Jiang, Yong and Li, Zhenguo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@inproceedings{li2022autoloss,
  title={AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks},
  author={Li, Hao and Fu, Tianwen and Dai, Jifeng and Li, Hongsheng and Huang, Gao and Zhu, Xizhou},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@article{grabocka2019learning,
  title={Learning Surrogate Losses},
  author={Grabocka, Josif and Scholz, Randolf and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1905.10108},
  year={2019}
}

@inproceedings{huang2019addressing,
  title={Addressing the Loss-Metric Mismatch with Adaptive Loss Alignment},
  author={Huang, Chen and Zhai, Shuangfei and Talbott, Walter and Martin, Miguel Bautista and Sun, Shih-Yu and Guestrin, Carlos and Susskind, Josh},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}

@inproceedings{morse2020learning,
  title={Learning State-Dependent Losses for Inverse Dynamics Learning},
  author={Morse, Kristen and Das, Neha and Lin, Yixin and Wang, Austin S and Rai, Akshara and Meier, Franziska},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE}
}

@article{gonzalez2020effective,
  title={Effective Regularization through Loss Function Meta-Learning},
  author={Gonzalez, Santiago and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2010.00788},
  year={2020}
}

@inproceedings{gao2022loss,
  title={Loss Function Learning for Domain Generalization by Implicit Gradient},
  author={Gao, Boyan and Gouk, Henry and Yang, Yongxin and Hospedales, Timothy},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}

## Meta-Learning Optimizers

@inproceedings{andrychowicz2016learning,
  title={Learning to Learn by Gradient Descent by Gradient Descent},
  author={Andrychowicz, Marcin and Denil, Misha and Gomez, Sergio and Hoffman, Matthew W and Pfau, David and Schaul, Tom and Shillingford, Brendan and De Freitas, Nando},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2016}
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

























@article{kim2018auto,
  title={Auto-Meta: Automated Gradient-Based Meta Learner Search},
  author={Kim, Jaehong and Lee, Sangyeul and Kim, Sungwan and Cha, Moonsu and Lee, Jung Kwon and Choi, Youngduck and Choi, Yongseok and Cho, Dong-Yeon and Kim, Jiwon},
  journal={arXiv preprint arXiv:1806.06927},
  year={2018}
}

@inproceedings{elsken2020meta,
  title={Meta-Learning of Neural Architectures for Few-Shot Learning},
  author={Elsken, Thomas and Staffler, Benedikt and Metzen, Jan Hendrik and Hutter, Frank},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

@inproceedings{ding2022learning,
  title={Learning to Learn by Jointly Optimizing Neural Architecture and Weights},
  author={Ding, Yadong and Wu, Yu and Huang, Chengyue and Tang, Siliang and Yang, Yi and Wei, Longhui and Zhuang, Yueting and Tian, Qi},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@article{leng2022polyloss,
  title={PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions},
  author={Leng, Zhaoqi and Tan, Mingxing and Liu, Chenxi and Cubuk, Ekin Dogus and Shi, Xiaojie and Cheng, Shuyang and Anguelov, Dragomir},
  journal = {International Conference on Learning Representations (ICLR)},
  year={2022}
}

@misc{collet2022loss,
  title={Loss Meta-Learning for Forecasting},
  author={Alan Collet and Antonio Bazco-Nogueras and Albert Banchs and Marco Fiore},
  year={2022}
}

@article{chen2022gradient,
  title={Gradient-Based Bi-level Optimization for Deep Learning: A Survey},
  author={Chen, Can and Chen, Xi and Ma, Chen and Liu, Zixuan and Liu, Xue},
  journal={arXiv preprint arXiv:2207.11719},
  year={2022}
}

@article{raymond2023online,
  title={Online Loss Function Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  journal={arXiv preprint arXiv:2301.13247},
  year={2023}
}

@article{wengert1964simple,
  title={A Simple Automatic Derivative Evaluation Program},
  author={Wengert, Robert Edwin},
  journal={Communications of the ACM},
  year={1964},
  publisher={ACM}
}

@inproceedings{domke2012generic,
  title={Generic Methods for Optimization-Based Modeling},
  author={Domke, Justin},
  booktitle={Artificial Intelligence and Statistics},
  year={2012}
}

@article{deledalle2014stein,
  title={Stein Unbiased GrAdient Estimator of the Risk (SUGAR) for Multiple Parameter Selection},
  author={Deledalle, Charles-Alban and Vaiter, Samuel and Fadili, Jalal and Peyr{\'e}, Gabriel},
  journal={SIAM Journal on Imaging Sciences},
  year={2014},
  publisher={SIAM}
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

@inproceedings{barron2019general,
  title={A General and Adaptive Robust Loss Function},
  author={Barron, Jonathan T},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@article{chaudhari2019entropy,
  title={Entropy-SGD: Biasing Gradient Descent into Wide Valleys},
  author={Chaudhari, Pratik and Choromanska, Anna and Soatto, Stefano and LeCun, Yann and Baldassi, Carlo and Borgs, Christian and Chayes, Jennifer and Sagun, Levent and Zecchina, Riccardo},
  journal={Journal of Statistical Mechanics: Theory and Experiment},
  year={2019},
  publisher={IOP Publishing}
}

@article{keskar2016large,
  title={On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima},
  author={Keskar, Nitish Shirish and Mudigere, Dheevatsa and Nocedal, Jorge and Smelyanskiy, Mikhail and Tang, Ping Tak Peter},
  journal={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@inproceedings{franceschi2018bilevel,
  title={Bilevel Programming for Hyperparameter Optimization and Meta-Learning},
  author={Franceschi, Luca and Frasconi, Paolo and Salzo, Saverio and Grazzi, Riccardo and Pontil, Massimiliano},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@inproceedings{baik2021meta,
  title={Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning},
  author={Baik, Sungyong and Choi, Janghoon and Kim, Heewon and Cho, Dohee and Min, Jaesik and Lee, Kyoung Mu},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
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

@article{wu2018learning,
  title={Learning to Teach with Dynamic Loss Functions},
  author={Wu, Lijun and Tian, Fei and Xia, Yingce and Fan, Yang and Qin, Tao and Jian-Huang, Lai and Liu, Tie-Yan},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

@article{kirsch2019improving,
  title={Improving Generalization in Meta Reinforcement Learning using Learned Objectives},
  author={Kirsch, Louis and van Steenkiste, Sjoerd and Schmidhuber, J{\"u}rgen},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@article{psaros2022meta,
  title={Meta-Learning PINN Loss Functions},
  author={Psaros, Apostolos F and Kawaguchi, Kenji and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  year={2022},
  publisher={Elsevier}
}

@book{mccullagh2019generalized,
  title={Generalized Linear Models},
  author={McCullagh, Peter and Nelder, John A and others},
  year={1989},
  publisher={Routledge}
}

@inproceedings{baydin2018hypergradient,
  title={Online Learning Rate Adaptation with Hypergradient Descent},
  author={Baydin, Atılım Güneş and Cornish, Robert and Rubio, David Martínez and Schmidt, Mark and Wood, Frank},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@article{javed2022simplification,
  title={Simplification of Genetic Programs: A Literature Survey},
  author={Javed, Noman and Gobet, Fernand and Lane, Peter},
  journal={Data Mining and Knowledge Discovery},
  year={2022},
  publisher={Springer}
}

@article{kinzett2009numerical,
  title={Numerical Simplification for Bloat Control and Analysis of Building Blocks in Genetic Programming},
  author={Kinzett, David and Johnston, Mark and Zhang, Mengjie},
  journal={Evolutionary Intelligence},
  year={2009},
  publisher={Springer}
}

@inproceedings{wong2006algebraic,
  title={Algebraic Simplification of GP Programs During Evolution},
  author={Wong, Phillip and Zhang, Mengjie},
  booktitle={ACM Genetic and Evolutionary Computation Conference (GECCO)},
  year={2006}
}

@article{antoniou2019learning,
  title={Learning to Learn by Self-Critique},
  author={Antoniou, Antreas and Storkey, Amos J},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@article{wu2018understanding,
  title={Understanding Short-Horizon Bias in Stochastic Meta-Optimization},
  author={Wu, Yuhuai and Ren, Mengye and Liao, Renjie and Grosse, Roger},
  journal={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@misc{asuncion2007uci,
  title={UCI Machine Learning Repository},
  author={Asuncion, Arthur and Newman, David},
  year={2007},
  publisher={Irvine, CA, USA}
}

@article{gao2023meta,
  title={Meta-Learning to Optimise: Loss Functions and Update Rules},
  author={Gao, Boyan},
  year={2023},
  journal={The University of Edinburgh}
}

@article{gonzalez2020improving,
  title={Improving Deep Learning through Loss-Function Evolution},
  author={Gonzalez, Santiago and others},
  year={2020},
  journal={The University of Texas at Austin}
}

@article{wolpert1997no,
  title={No Free Lunch Theorems for Optimization},
  author={Wolpert, David H and Macready, William G},
  journal={IEEE Transactions on Evolutionary Computation (TEVC)},
  year={1997},
  organization={IEEE}
}

@article{muller2019does,
  title={When Does Label Smoothing Help?},
  author={M{\"u}ller, Rafael and Kornblith, Simon and Hinton, Geoffrey E},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@inproceedings{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}

@article{black1996unification,
  title={On the Unification of Line Processes, Outlier Rejection, and Robust Statistics with Applications in Early Vision},
  author={Black, Michael J and Rangarajan, Anand},
  journal={International Journal of Computer Vision (IJCV)},
  year={1996},
  publisher={Springer}
}

@article{liu2017survey,
  title={A Survey of Deep Neural Network Architectures and their Applications},
  author={Liu, Weibo and Wang, Zidong and Liu, Xiaohui and Zeng, Nianyin and Liu, Yurong and Alsaadi, Fuad E},
  journal={Neurocomputing},
  year={2017},
  publisher={Elsevier}
}

@article{gu2018recent,
  title={Recent Advances in Convolutional Neural Networks},
  author={Gu, Jiuxiang and Wang, Zhenhua and Kuen, Jason and Ma, Lianyang and Shahroudy, Amir and Shuai, Bing and Liu, Ting and Wang, Xingxing and Wang, Gang and Cai, Jianfei and others},
  journal={Pattern Recognition},
  year={2018},
  publisher={Elsevier}
}

@incollection{bengio2012practical,
  title={Practical Recommendations for Gradient-Based Training of Deep Architectures},
  author={Bengio, Yoshua},
  booktitle={Neural Networks: Tricks of the Trade},
  year={2012},
  publisher={Springer}
}

@article{gordon1995evaluation,
  title={Evaluation and Selection of Biases in Machine Learning},
  author={Gordon, Diana F and Desjardins, Marie},
  journal={Machine Learning},
  year={1995},
  publisher={Springer}
}

@article{bergstra2011algorithms,
  title={Algorithms for Hyper-parameter Optimization},
  author={Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2011}
}

@incollection{feurer2019hyperparameter,
  title={Hyperparameter Optimization},
  author={Feurer, Matthias and Hutter, Frank},
  booktitle={Automated Machine Learning},
  year={2019},
  publisher={Springer}
}

@book{reed1999neural,
  title={Neural Smithing: Supervised Learning in Feedforward Artificial Neural Networks},
  author={Reed, Russell and MarksII, Robert J},
  year={1999},
  publisher={Mit Press}
}

@article{raymond2023learning,
  title={Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2023},
  organization={IEEE}
}

@inproceedings{raymond2023fast,
  title={Fast and Efficient Local-Search for Genetic Programming Based Loss Function Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  booktitle={ACM Genetic and Evolutionary Computation Conference (GECCO)},
  year={2023}
}

@inproceedings{maas2013rectifier,
  title={Rectifier Nonlinearities Improve Neural Network Acoustic Models},
  author={Maas, Andrew L and Hannun, Awni Y and Ng, Andrew Y and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2013},
}

@article{dugas2000incorporating,
  title={Incorporating Second-Order Functional Knowledge for Better Option Pricing},
  author={Dugas, Charles and Bengio, Yoshua and B{\'e}lisle, Fran{\c{c}}ois and Nadeau, Claude and Garcia, Ren{\'e}},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2000}
}

@book{koza1994genetic,
  title={Genetic Programming II},
  author={Koza, John R},
  year={1994},
  publisher={MIT press Cambridge}
}

@article{hansen2001completely,
  title={Completely Derandomized Self-Adaptation in Evolution Strategies},
  author={Hansen, Nikolaus and Ostermeier, Andreas},
  journal={Evolutionary Computation},
  year={2001},
  publisher={MIT Press}
}

@article{paszke2017automatic,
  title={Automatic Differentiation in Pytorch},
  author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
  year={2017}
}

@article{abadi2016tensorflow,
  title={TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems},
  author={Abadi, Mart{\'\i}n and Agarwal, Ashish and Barham, Paul and Brevdo, Eugene and Chen, Zhifeng and Citro, Craig and Corrado, Greg S and Davis, Andy and Dean, Jeffrey and Devin, Matthieu and others},
  journal={arXiv preprint arXiv:1603.04467},
  year={2016}
}

@article{jax2018github,
  title = {JAX: Composable Transformations of Python+NumPy Programs},
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  year = {2018},
}

@article{baydin2018automatic,
  title={Automatic Differentiation in Machine Learning: A Survey},
  author={Baydin, Atilim Gunes and Pearlmutter, Barak A and Radul, Alexey Andreyevich and Siskind, Jeffrey Mark},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2018},
}

@article{yao2007early,
  title={On Early Stopping in Gradient Descent Learning},
  author={Yao, Yuan and Rosasco, Lorenzo and Caponnetto, Andrea},
  journal={Constructive Approximation},
  year={2007},
  publisher={Springer}
}

@article{jordan2015machine,
  title={Machine learning: Trends, Perspectives, and Prospects},
  author={Jordan, Michael I and Mitchell, Tom M},
  journal={Science},
  year={2015},
  publisher={American Association for the Advancement of Science}
}

@book{abu2012learning,
  title={Learning from Data},
  author={Abu-Mostafa, Yaser S and Magdon-Ismail, Malik and Lin, Hsuan-Tien},
  year={2012},
  publisher={AMLBook}
}

@article{mitchell1997machine,
  title={Machine Learning},
  author={Mitchell, Tom},
  year={1997},
  journal={McGraw hill Burr Ridge}
}

@book{domingos2015master,
  title={The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake our World},
  author={Domingos, Pedro},
  year={2015},
  publisher={Basic Books}
}

@article{russell2002artificial,
  title={Artificial Intelligence: A Modern Approach},
  author={Russell, Stuart and Norvig, Peter},
  year={2002},
  journal={Prentice Hall}
}

@book{hinton1999unsupervised,
  title={Unsupervised Learning: Foundations of Neural Computation},
  author={Hinton, Geoffrey E and Sejnowski, Terrence Joseph and others},
  year={1999},
  publisher={MIT press}
}

@article{kaelbling1996reinforcement,
  title={Reinforcement Learning: A Survey},
  author={Kaelbling, Leslie Pack and Littman, Michael L and Moore, Andrew W},
  journal={Journal of Artificial Intelligence Research (JAIR)},
  year={1996}
}

@article{silver2016mastering,
  title={Mastering the Game of Go with Deep Neural Networks and Tree Search},
  author={Silver, David and Huang, Aja and Maddison, Chris J and Guez, Arthur and Sifre, Laurent and Van Den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and others},
  journal={Nature},
  year={2016},
  publisher={Nature Publishing Group}
}

@article{vinyals2019grandmaster,
  title={Grandmaster Level in StarCraft II using Multi-Agent Reinforcement Learning},
  author={Vinyals, Oriol and Babuschkin, Igor and Czarnecki, Wojciech M and Mathieu, Micha{\"e}l and Dudzik, Andrew and Chung, Junyoung and Choi, David H and Powell, Richard and Ewalds, Timo and Georgiev, Petko and others},
  journal={Nature},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{colson2007overview,
  title={An Overview of Bilevel Optimization},
  author={Colson, Beno{\^\i}t and Marcotte, Patrice and Savard, Gilles},
  journal={Annals of Operations Research},
  year={2007},
  publisher={Springer}
}

@book{bard2013practical,
  title={Practical Bilevel Optimization: Algorithms and Applications},
  author={Bard, Jonathan F},
  year={2013},
  publisher={Springer Science \& Business Media}
}

@article{hornik1989multilayer,
  title={Multilayer Feedforward Networks are Universal Approximators},
  author={Hornik, Kurt and Stinchcombe, Maxwell and White, Halbert},
  journal={Neural Networks},
  year={1989},
  publisher={Elsevier}
}

@incollection{michalski1983theory,
  title={A Theory and Methodology of Inductive Learning},
  author={Michalski, Ryszard S},
  booktitle={Machine Learning},
  year={1983},
  publisher={Elsevier}
}

@book{thrun2012learning,
  title={Learning to Learn},
  author={Thrun, Sebastian and Pratt, Lorien},
  year={2012},
  publisher={Springer Science \& Business Media}
}

@inproceedings{bengio2012deep,
  title={Deep Learning of Representations for Unsupervised and Transfer Learning},
  author={Bengio, Yoshua},
  booktitle={International Conference on Machine Learning (ICML) Workshop on Unsupervised and Transfer Learning},
  year={2012}
}

@article{caruana1997multitask,
  title={Multitask Learning},
  author={Caruana, Rich},
  journal={Machine Learning},
  year={1997},
  publisher={Springer}
}

@article{baxter2000model,
  title={A Model of Inductive Bias Learning},
  author={Baxter, Jonathan},
  journal={Journal of Artificial Intelligence Research (JAIR)},
  year={2000}
}

@article{yao2018taking,
  title={Taking Human out of Learning Applications: A Survey on Automated Machine Learning},
  author={Yao, Quanming and Wang, Mengshuo and Chen, Yuqiang and Dai, Wenyuan and Li, Yu-Feng and Tu, Wei-Wei and Yang, Qiang and Yu, Yang},
  journal={arXiv preprint arXiv:1810.13306},
  year={2018}
}

@book{hutter2019automated,
  title={Automated Machine Learning: Methods, Systems, Challenges},
  author={Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
  year={2019},
  publisher={Springer Nature}
}

@article{bergstra2012random,
  title={Random Search for Hyper-parameter Optimization.},
  author={Bergstra, James and Bengio, Yoshua},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2012}
}

@article{snoek2012practical,
  title={Practical Bayesian Optimization of Machine Learning Algorithms},
  author={Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2012}
}

@article{snell2017prototypical,
  title={Prototypical Networks for Few-Shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@article{boudiaf2020transductive,
  title={Information Maximization for Few-Shot Learning},
  author={Boudiaf, Malik and Ziko, Imtiaz and Rony, J{\'e}r{\^o}me and Dolz, Jos{\'e} and Piantanida, Pablo and Ben Ayed, Ismail},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{al2019survey,
  title={A Survey on Evolutionary Machine Learning},
  author={Al-Sahaf, Harith and Bi, Ying and Chen, Qi and Lensen, Andrew and Mei, Yi and Sun, Yanan and Tran, Binh and Xue, Bing and Zhang, Mengjie},
  journal={Journal of the Royal Society of New Zealand},
  year={2019},
  publisher={Taylor \& Francis}
}

@inproceedings{koch2015siamese,
  title={Siamese Neural Networks for One-Shot Image Recognition},
  author={Koch, Gregory and Zemel, Richard and Salakhutdinov, Ruslan and others},
  booktitle={International Conference on Machine Learning (ICML) Deep Learning Workshop},
  year={2015}
}

@article{li2021autoloss,
  title={AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks},
  author={Li, Hao and Fu, Tianwen and Dai, Jifeng and Li, Hongsheng and Huang, Gao and Zhu, Xizhou},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@inproceedings{bello2017neural,
  title={Neural Optimizer Search with Reinforcement Learning},
  author={Bello, Irwan and Zoph, Barret and Vasudevan, Vijay and Le, Quoc V},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@article{flennerhag2019meta,
  title={Meta-Learning with Warped Gradient Descent},
  author={Flennerhag, Sebastian and Rusu, Andrei A and Pascanu, Razvan and Visin, Francesco and Yin, Hujun and Hadsell, Raia},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
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

@article{kingma2014adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={International Conference on Learning Representations (ICLR)},
  year={2015}
}

@article{hinton2012neural,
  title={Neural Networks for Machine Learning Lecture 6a Overview of Mini-Batch Gradient Descent},
  author={Hinton, Geoffrey and Srivastava, Nitish and Swersky, Kevin},
  year={2012}
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

@article{elsken2019neural,
  title={Neural Architecture Search: A Survey},
  author={Elsken, Thomas and Metzen, Jan Hendrik and Hutter, Frank},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2019},
  publisher={JMLR}
}

@inproceedings{real2019regularized,
  title={Regularized Evolution for Image Classifier Architecture Search},
  author={Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}

@inproceedings{wang2018evolving,
  title={Evolving Deep Convolutional Neural Networks by Variable-Length Particle Swarm Optimization for Image Classification},
  author={Wang, Bin and Sun, Yanan and Xue, Bing and Zhang, Mengjie},
  booktitle={IEEE Congress on Evolutionary Computation (CEC)},
  year={2018}
}

@article{sun2019completely,
  title={Completely Automated CNN Architecture Design Based on Blocks},
  author={Sun, Yanan and Xue, Bing and Zhang, Mengjie and Yen, Gary G},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  year={2019},
  organization={IEEE}
}

@article{liu2018darts,
  title={DARTS: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@article{xue2015survey,
  title={A Survey on Evolutionary Computation Approaches to Feature Selection},
  author={Xue, Bing and Zhang, Mengjie and Browne, Will N and Yao, Xin},
  journal={IEEE Transactions on Evolutionary Computation (TEVC)},
  year={2015},
  organization={IEEE}
}

@inproceedings{kennedy1995particle,
  title={Particle Swarm Optimization},
  author={Kennedy, James and Eberhart, Russell},
  booktitle={International Conference on Neural Networks},
  year={1995},
  organization={IEEE}
}

@book{mitchell1998introduction,
  title={An Introduction to Genetic Algorithms},
  author={Mitchell, Melanie},
  year={1998},
  publisher={MIT press}
}

@article{storn1997differential,
  title={Differential Evolution -- A Simple and Efficient Heuristic for Global Optimization Over Continuous Spaces},
  author={Storn, Rainer and Price, Kenneth},
  journal={Journal of Global Optimization},
  year={1997},
  publisher={Springer}
}

@article{cao2019learning,
  title={Learning to Optimize in Swarms},
  author={Cao, Yue and Chen, Tianlong and Wang, Zhangyang and Shen, Yang},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@article{eiben2015evolutionary,
  title={From Evolutionary Computation to the Evolution of Things},
  author={Eiben, Agoston E and Smith, Jim},
  journal={Nature},
  year={2015},
  publisher={Nature Publishing Group}
}

@article{bishop2006pattern,
  title={Pattern Recognition and Machine Learning},
  author={Bishop, Christopher M},
  journal={Machine Learning},
  year={2006}
}

@article{hoerl1970ridge,
  title={Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  author={Hoerl, Arthur E and Kennard, Robert W},
  journal={Technometrics},
  year={1970},
  publisher={Taylor \& Francis}
}

@article{freund1999large,
  title={Large Margin Classification using the Perceptron Algorithm},
  author={Freund, Yoav and Schapire, Robert E},
  journal={Machine Learning},
  year={1999},
  publisher={Springer}
}

@book{minsky2017perceptrons,
  title={Perceptrons: An Introduction to Computational Geometry},
  author={Minsky, Marvin and Papert, Seymour A},
  year={2017},
  publisher={MIT press}
}

@book{taylor1717methodus,
  title={Methodus Incrementorum Directa et Inversa},
  author={Taylor, Brook},
  year={1717},
  publisher={Innys}
}

@article{collobert2011natural,
  title={Natural Language Processing (Almost) From Scratch},
  author={Collobert, Ronan and Weston, Jason and Bottou, L{\'e}on and Karlen, Michael and Kavukcuoglu, Koray and Kuksa, Pavel},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2011}
}

@article{yi2014learning,
  title={Learning Face Representation from Scratch},
  author={Yi, Dong and Lei, Zhen and Liao, Shengcai and Li, Stan Z},
  journal={arXiv preprint arXiv:1411.7923},
  year={2014}
}

@article{lecun2015deep,
  title={Deep Learning},
  author={LeCun, Yann and Bengio, Yoshua and Hinton, Geoffrey},
  journal={Nature},
  year={2015},
  publisher={Nature Publishing Group}
}

@inproceedings{alet2018modular,
  title={Modular Meta-learning},
  author={Alet, Ferran and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie P},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2018}
}

@article{janocha2017loss,
  title={On Loss Functions for Deep Neural Networks in Classification},
  author={Janocha, Katarzyna and Czarnecki, Wojciech Marian},
  journal={Theoretical Foundations of Machine Learning},
  year={2017}
}

@article{huber1964robust,
  title={Robust Estimation of a Location Parameter},
  author={Huber, Peter J},
  journal={The Annals of Mathematical Statistics},
  year={1964},
  publisher={Institute of Mathematical Statistics}
}

@inproceedings{charbonnier1994two,
  title={Two Deterministic Half-Quadratic Regularization Algorithms for Computed Imaging},
  author={Charbonnier, Pierre and Blanc-Feraud, Laure and Aubert, Gilles and Barlaud, Michel},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={1994}
}

@article{ganan1985bayesian,
  title={Bayesian Image Analysis: An Application to Single Photon Emission Tomography},
  author={Ganan, Stuart and McClure, D},
  journal={American Statistics Association},
  year={1985}
}

@article{dennis1978techniques,
  title={Techniques for Nonlinear Least Squares and Robust Regression},
  author={Dennis Jr, John E and Welsch, Roy E},
  journal={Communications in Statistics Simulation and Computation},
  year={1978},
  publisher={Taylor \& Francis}
}

@article{leclerc1989constructing,
  title={Constructing Simple Stable Descriptions for Image Partitioning},
  author={Leclerc, Yvan G},
  journal={International Journal of Computer Vision (IJCV)},
  year={1989},
  publisher={Springer}
}

@article{sung2017learning,
  title={Learning to Learn: Meta-Critic Networks for Sample Efficient Learning},
  author={Sung, Flood and Zhang, Li and Xiang, Tao and Hospedales, Timothy and Yang, Yongxin},
  journal={arXiv preprint arXiv:1706.09529},
  year={2017}
}

@article{zhou2020online,
  title={Online Meta-Critic Learning for Off-Policy Actor-Critic Methods},
  author={Zhou, Wei and Li, Yiying and Yang, Yongxin and Wang, Huaimin and Hospedales, Timothy},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{boney2018semi,
  title={Semi-Supervised Few-Shot Learning with MAML},
  author={Boney, Rinu and Ilin, Alexander},
  year={2018}
}

@article{hansen2003reducing,
  title={Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)},
  author={Hansen, Nikolaus and M{\"u}ller, Sibylle D and Koumoutsakos, Petros},
  journal={Evolutionary Computation},
  year={2003},
  publisher={MIT Press}
}

@inproceedings{li2019lfs,
  title={AM-LFS: AutoML for Loss Function Search},
  author={Li, Chuming and Yuan, Xin and Lin, Chen and Guo, Minghao and Wu, Wei and Yan, Junjie and Ouyang, Wanli},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}

@inproceedings{wang2020loss,
  title={Loss Function Search for Face Recognition},
  author={Wang, Xiaobo and Wang, Shuo and Chi, Cheng and Zhang, Shifeng and Mei, Tao},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}

@article{park2019meta,
  title={Meta-Curvature},
  author={Park, Eunbyung and Oliva, Junier B},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@article{baik2020meta,
  title={Meta-Learning with Adaptive Hyperparameters},
  author={Baik, Sungyong and Choi, Myungsub and Choi, Janghoon and Kim, Heewon and Lee, Kyoung Mu},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{van2008visualizing,
  title={Visualizing Data using t-SNE.},
  author={Van der Maaten, Laurens and Hinton, Geoffrey},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2008}
}

@inproceedings{perez2018film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Perez, Ethan and Strub, Florian and De Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  booktitle={AAAI conference on artificial intelligence},
  year={2018}
}

@inproceedings{kohavi1996bias,
  title={Bias Plus Variance Decomposition for Zero-One Loss Functions},
  author={Kohavi, Ron and Wolpert, David H and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={1996}
}

@article{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@article{ho2002simple,
  title={Simple Explanation of the No-Free-Lunch Theorem and its Implications},
  author={Ho, Yu-Chi and Pepyne, David L},
  journal={Journal of Optimization Theory and Applications},
  year={2002},
  publisher={Springer}
}

@article{morgan2024neural,
  title={Neural Loss Function Evolution for Large-Scale Image Classifier Convolutional Neural Networks},
  author={Morgan, Brandon and Hougen, Dean},
  journal={arXiv preprint arXiv:2403.08793},
  year={2024}
}

@article{morgan2024evolving,
  title={Evolving Loss Functions for Specific Image Augmentation Techniques},
  author={Morgan, Brandon and Hougen, Dean},
  journal={arXiv preprint arXiv:2404.06633},
  year={2024}
}

@article{yu2015multi,
  title={Multi-Scale Context Aggregation by Dilated Convolutions},
  author={Yu, Fisher and Koltun, Vladlen},
  journal={International Conference on Learning Representations (ICLR)},
  year={2016}
}

@article{oord2016wavenet,
  title={WaveNet: A Generative Model for Raw Audio},
  author={Oord, Aaron van den and Dieleman, Sander and Zen, Heiga and Simonyan, Karen and Vinyals, Oriol and Graves, Alex and Kalchbrenner, Nal and Senior, Andrew and Kavukcuoglu, Koray},
  journal={arXiv preprint arXiv:1609.03499},
  year={2016}
}

@inproceedings{huang2017densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}

@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}

@article{de2021continual,
  title={A Continual Learning Survey: Defying Forgetting in Classification Tasks},
  author={De Lange, Matthias and Aljundi, Rahaf and Masana, Marc and Parisot, Sarah and Jia, Xu and Leonardis, Ale{\v{s}} and Slabaugh, Gregory and Tuytelaars, Tinne},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2021},
  organization={IEEE}
}

@article{wang2024comprehensive,
  title={A Comprehensive Survey of Continual Learning: Theory, Method and Application},
  author={Wang, Liyuan and Zhang, Xingxing and Su, Hang and Zhu, Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024},
  organization={IEEE}
}

@article{parisi2019continual,
  title={Continual Lifelong Learning with Neural Networks: A Review},
  author={Parisi, German I and Kemker, Ronald and Part, Jose L and Kanan, Christopher and Wermter, Stefan},
  journal={Neural Networks},
  year={2019},
  publisher={Elsevier}
}

@article{bohdal2021evograd,
  title={Evograd: Efficient Gradient-Based Meta-Learning and Hyperparameter Optimization},
  author={Bohdal, Ondrej and Yang, Yongxin and Hospedales, Timothy},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

@article{mildenhall2021nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  journal={Communications of the ACM},
  year={2021},
  publisher={ACM}
}

@article{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  year={2023},
  publisher={ACM}
}

@article{beck2023survey,
  title={A Survey of Meta-Reinforcement Learning},
  author={Beck, Jacob and Vuorio, Risto and Liu, Evan Zheran and Xiong, Zheng and Zintgraf, Luisa and Finn, Chelsea and Whiteson, Shimon},
  journal={arXiv preprint arXiv:2301.08028},
  year={2023}
}

@article{duchi2011adaptive,
  title={Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.},
  author={Duchi, John and Hazan, Elad and Singer, Yoram},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2011}
}

@article{zeiler2012adadelta,
  title={AdaDelta: An Adaptive Learning Rate Method},
  author={Zeiler, Matthew D},
  journal={arXiv preprint arXiv:1212.5701},
  year={2012}
}

@article{polyak1964some,
  title={Some Methods of Speeding up the Convergence of Iteration Methods},
  author={Polyak, Boris T},
  journal={USSR Computational Mathematics and Mathematical Physics},
  year={1964},
  publisher={Elsevier}
}

@article{chen2022learning,
  title={Learning to Optimize: A Primer and a Benchmark},
  author={Chen, Tianlong and Chen, Xiaohan and Chen, Wuyang and Heaton, Howard and Liu, Jialin and Wang, Zhangyang and Yin, Wotao},
  journal={Journal of Machine Learning Research (JMLR)},
  year={2022}
}

@article{li2016learning,
  title={Learning to Optimize},
  author={Li, Ke and Malik, Jitendra},
  journal={arXiv preprint arXiv:1606.01885},
  year={2016}
}

@inproceedings{chen2017learning,
  title={Learning to Learn Without Gradient Descent by Gradient Descent},
  author={Chen, Yutian and Hoffman, Matthew W and Colmenarejo, Sergio G{\'o}mez and Denil, Misha and Lillicrap, Timothy P and Botvinick, Matt and Freitas, Nando},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@inproceedings{schafer2006recurrent,
  title={Recurrent Neural Networks are Universal Approximators},
  author={Sch{\"a}fer, Anton Maximilian and Zimmermann, Hans Georg},
  booktitle={International Conference on Artificial Neural Networks (ICANN)},
  year={2006},
  organization={Springer}
}

@article{jaeger2002tutorial,
  title={Tutorial on Training Recurrent Neural Networks, Covering BPPT, RTRL, EKF and The Echo State Network Approach},
  author={Jaeger, Herbert},
  year={2002},
  journal={GMD-Forschungszentrum Informationstechnik Bonn}
}

@article{lee2018meta,
  title={Meta-Learning with Adaptive Layerwise Metric and Subspace},
  author={Yoonho Lee and Seungjin Choi},
  journal={International Conference on Machine Learning (ICML)},
  year={2018}
}

@book{wright2006numerical,
  title={Numerical Optimization},
  author={Wright, Stephen J},
  year={2006},
  publisher={Springer}
}

@book{amari2000methods,
  title={Methods of Information Geometry},
  author={Amari, Shunichi and Nagaoka, Hiroshi},
  year={2000},
  publisher={American Mathematical Society}
}

@inproceedings{he2015delving,
  title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2015}
}

@inproceedings{metz2019understanding,
  title={Understanding and Correcting Pathologies in the Training of Learned Optimizers},
  author={Metz, Luke and Maheswaranathan, Niru and Nixon, Jeremy and Freeman, Daniel and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}

@inproceedings{zoph2018learning,
  title={Learning Transferable Architectures for Scalable Image Recognition},
  author={Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
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

@article{baik2023meta,
  title={Learning to Learn Task-Adaptive Hyperparameters for Few-Shot Learning}, 
  author={Baik, Sungyong and Choi, Myungsub and Choi, Janghoon and Kim, Heewon and Lee, Kyoung Mu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
  year={2023},
  organization={IEEE}
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

@article{amari1998natural,
  title={Natural Gradient Works Efficiently in Learning},
  author={Amari, Shunichi},
  journal={Neural Computation},
  year={1998},
  publisher={MIT Press}
}

@article{vapnik2006transductive,
  title={Transductive Inference and Semi-Supervised Learning},
  author={Vapnik, Vladimir},
  year={2006}
}

@inproceedings{sung2018learning,
  title={Learning to Compare: Relation Network for Few-Shot Learning},
  author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}

@inproceedings{rusu2019metalearning,
  title={Meta-Learning with Latent Embedding Optimization},
  author={Andrei A. Rusu and Dushyant Rao and Jakub Sygnowski and Oriol Vinyals and Razvan Pascanu and Simon Osindero and Raia Hadsell},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
}

@article{dumoulin2018feature,
  title = {Feature-Wise Transformations},
  author = {Dumoulin, Vincent and Perez, Ethan and Schucher, Nathan and Strub, Florian and Vries, Harm de and Courville, Aaron and Bengio, Yoshua},
  journal = {Distill},
  year = {2018},
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

@inproceedings{ioffe2015batch,
  title={Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
  author={Ioffe, Sergey and Szegedy, Christian},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2015},
}

@article{de2017modulating,
  title={Modulating Early Visual Processing by Language},
  author={De Vries, Harm and Strub, Florian and Mary, J{\'e}r{\'e}mie and Larochelle, Hugo and Pietquin, Olivier and Courville, Aaron C},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
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

@inproceedings{ye2021unicorn,
  title={How to Train Your MAML to Excel in Few-Shot Classification},
  author={Han-Jia Ye and Wei-Lun Chao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{smith2017don,
  title={Don't Decay the Learning Rate, Increase the Batch Size},
  author={Smith, Samuel L and Kindermans, Pieter-Jan and Ying, Chris and Le, Quoc V},
  journal={arXiv preprint arXiv:1711.00489},
  year={2017}
}

@article{goyal2017accurate,
  title={Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour},
  author={Goyal, Priya and Doll{\'a}r, Piotr and Girshick, Ross and Noordhuis, Pieter and Wesolowski, Lukasz and Kyrola, Aapo and Tulloch, Andrew and Jia, Yangqing and He, Kaiming},
  journal={arXiv preprint arXiv:1706.02677},
  year={2017}
}

@article{smith2017bayesian,
  title={A Bayesian Perspective on Generalization and Stochastic Gradient Descent},
  author={Smith, Samuel L and Le, Quoc V},
  journal={arXiv preprint arXiv:1710.06451},
  year={2017}
}

@article{oh2020boil,
  title={BOIL: Towards Representation Change for Few-Shot Learning},
  author={Oh, Jaehoon and Yoo, Hyungjun and Kim, ChangHwan and Yun, Se-Young},
  journal={arXiv preprint arXiv:2008.08882},
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

@article{raghu2019rapid,
  title={Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML},
  author={Raghu, Aniruddh and Raghu, Maithra and Bengio, Samy and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1909.09157},
  year={2019}
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

@inproceedings{deng2009imagenet,
  title={Imagenet: A Large-Scale Hierarchical Image Database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2009}
}

@inproceedings{lee2019meta,
  title={Meta-Learning with Differentiable Convex Optimization},
  author={Lee, Kwonjoon and Maji, Subhransu and Ravichandran, Avinash and Soatto, Stefano},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@article{raymond2024meta,
  title={Meta-Learning Neural Procedural Biases},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  journal={arXiv preprint arXiv:2406.07983},
  year={2024}
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

@article{javed2019meta,
  title={Meta-Learning Representations for Continual Learning},
  author={Javed, Khurram and White, Martha},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
