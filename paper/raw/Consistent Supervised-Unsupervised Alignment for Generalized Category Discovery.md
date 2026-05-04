---
title: "Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery"
source: "https://arxiv.org/html/2507.04725v3"
author:
published:
created: 2026-05-04
description:
tags:
  - "clippings"
---
Jizhou Han <sup>1</sup>, Shaokun Wang <sup>2</sup>,  Yuhang He <sup>1</sup>, Chenhao Ding <sup>3</sup>, Qiang Wang <sup>1</sup>,  
Xinyuan Gao <sup>4</sup>,  Songlin Dong <sup>5</sup>,  Yihong Gong <sup>1</sup> <sup>1</sup>  
<sup>1</sup> National Key Laboratory of Human-Machine Hybrid Augmented Intelligence,  
National Engineering Research Center for Visual Information and Applications,  
Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University  
<sup>2</sup> School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen  
<sup>3</sup> School of Software Engineering, Xi’an Jiaotong University   <sup>4</sup> Kuaishou Technology  
<sup>5</sup> Faculty of Microelectronics, Shenzhen University of Advanced Technology  
{jizhou-han, dch225739, qwang}@stu.xjtu.edu.cn, heyuhang@xjtu.edu.cn,  
wangshaokun@hit.edu.cn, dongsl@suat-sz.edu.cn, ygong@mail.xjtu.edu.cn Yihong Gong and Shaokun Wang are the corresponding authors.

###### Abstract

Generalized Category Discovery (GCD) focuses on classifying known categories while simultaneously discovering novel categories from unlabeled data. However, previous GCD methods suffer from inconsistent optimization objectives. This inconsistency leads to feature overlap and ultimately hinders performance on novel categories. To address these issues, we propose the Neural Collapse-inspired Generalized Category Discovery (NC-GCD) framework. By pre-assigning and fixing Equiangular Tight Frame (ETF) prototypes, our method ensures an optimal geometric structure and a consistent optimization objective for both known and novel categories. We introduce a Consistent ETF Alignment Loss that unifies supervised and unsupervised ETF alignment while enhancing category separability. Additionally, a Semantic Consistency Matcher (SCM) is designed to maintain stable and consistent label assignments across clustering iterations. Our method achieves state-of-the-art performance on multiple GCD benchmarks, significantly enhancing novel category accuracy and demonstrating its effectiveness.

## 1 Introduction

Generalized Category Discovery (GCD) has raised emerging attention in recent years, which aims to simultaneously classify known categories and discover novel ones from unlabeled data. Unlike traditional semi-supervised learning, which assumes all categories are predefined during training, GCD operates in a more realistic open-world scenario [^24] [^1] [^20] where only a small portion of known-category data is labeled, and all novel-category samples lack both labels and category information. It requires models to utilize limited supervision from labeled known categories while autonomously discovering semantically coherent clusters in the unlabeled subset without assuming predefined category structures or a clear distinction between known and novel distributions.

Recent research in GCD has witnessed substantial progress. Early research [^38] introduces a contrastive learning framework to distinguish known and novel categories. On this basis, contrastive learning-based methods [^33] [^46] refine category boundaries using dynamic contrastive learning and prompt-based affinity learning. Additionally, representation learning-based methods [^5] [^6] [^27] [^34] focus on optimizing feature representations to enhance category separability.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/Nips-img1.jpg)

Figure 1: Illustration of the differences between prior studies, our proposed NC-GCD framework, and Performance Overview. (a) Our method pre-assigns fixed ETF prototypes rather than dynamically learning prototypes, ensuring consistent optimization objectives for known and novel categories. (b) The overview of the NC-GCD. (c) Compared to previous studies, our method exhibits superior overall accuracy, with a notable improvement in the accuracy of novel categories.

However, most existing methods dynamically optimize cluster prototypes or classification weights, leading to two key issues. First, Inconsistent Optimization Objective: Existing frameworks lack a consistent optimization objective for known and novel categories. As a result, the models tend to prioritize learning from labeled known categories while failing to establish proper decision boundaries for the novel ones. Second, Category Confusion: As shown in Fig. 1a, existing methods lack geometric constraints on feature distributions. Moreover, novel categories are optimized entirely in an unsupervised manner, making it even more difficult to achieve sufficient separation between feature-similar categories, leading to category overlap and reduced accuracy. Based on the above observations, we wonder:

Can we pre-assign an optimal geometric structure where both known and novel categories are equally well-separated, enabling consistent learning for all categories by aligning features to this structure?

Neural Collapse (NC) is a phenomenon in well-trained classification networks, where features of each category align with a Simplex Equiangular Tight Frame (ETF) structure [^30]. In this structure, within-category features collapse to their respective category mean, and the means of different categories are at the vertices of a simplex. This alignment maximizes inter-category separation while maintaining within-category compactness, creating an optimal geometric arrangement for classification tasks.

Motivated by Neural Collapse theory, we propose the Neural Collapse-inspired Generalized Category Discovery (NC-GCD) framework, which leverages NC principles to create a structured feature space for both known and novel categories. Unlike existing methods that dynamically learn prototypes, as shown in Fig. 1b, our approach pre-assigns ETF prototypes before training, ensuring an optimal geometric structure and a consistent optimization objective. Our framework integrates three key components: 1) Unsupervised ETF Alignment periodically clustering all categories and aligning the top $\alpha$ % most confident samples to their respective ETF prototypes. This process enhances feature separability, particularly for novel categories. 2) Supervised ETF Alignment stabilizes known categories by aligning labeled features with their corresponding ETF prototypes. Afterwards, we unify supervised and unsupervised alignment by designing a Consistent ETF alignment loss. However, clustering instability may lead to inconsistent label assignments, where samples of the same category are mapped to different clusters across multiple iterations, thereby disrupting ETF alignment. Even more critically, the direct mapping between ground-truth labels and ETF prototypes in supervised alignment can introduce mismatches, further compromising model stability. 3) To address these issues, we introduce the Semantic Consistency Matcher (SCM), which enforces one-to-one alignment between clusters across iterations, stabilizing pseudo-label assignments and ensuring proper mapping of supervised labels to ETF prototypes. By integrating these components, our method establishes a consistent optimization objective, building a structured feature space that enhances category separability and effectively mitigates category confusion. Our contributions are summarized as follows:

(1) We propose a fixed ETF prototype framework that establishes a consistent optimization objective for both known and novel categories, improving category separability through a consistent supervised-unsupervised alignment.

(2) We introduce the Semantic Consistency Matcher to maintain consistency in cluster labels across multiple iterations, stabilizing the training process and reducing clustering-induced fluctuations.

(3) Our method, as demonstrated in Fig. 1c, achieves strong performance on six GCD benchmarks, with significant improvements in novel category accuracy.

## 2 Related Work

### 2.1 Generalized Category Discovery (GCD)

GCD identifies known and novel categories within a partially labeled dataset. Early methods [^38] leveraged contrastive learning to distinguish categories. Later, DCCL [^33] and PromptCAL [^46] refined category boundaries using dynamic contrastive learning and prompt-based affinity mechanisms. GPC [^47] and SimGCD [^42] introduced Gaussian Mixture Models and parametric classification frameworks to estimate category distributions. Other methods have focused on self-supervised clustering and feature refinement. PIM [^5] and CMS [^6] introduced contrastive mean-shift learning and representation alignment strategies to improve novel category separability. Reciprocal learning and class-wise regularization have been introduced in RLCD [^25], while ProtoGCD [^26] unifies prototype optimization and debiasing for category discovery. Meanwhile, approaches like UNO [^10] and ORCA [^2] explored semi-supervised representation learning, while RankStats [^14] employed ranking-based statistics to facilitate unknown category discovery. Meanwhile, there is a growing line of work on continual generalized category discovery [^21] [^4] [^31] [^35]. However, most existing methods rely on dynamically learned prototypes and lack a consistent optimization target. This results in category confusion and imbalanced learning.

### 2.2 Neural Collapse and its Applications

Neural Collapse (NC) describes a geometric phenomenon in deep networks where, in the final stage of training, category feature representations converge to form a Simplex Equiangular Tight Frame structure [^30]. This structure ensures optimal category separation and minimal intra-category variance, making it highly effective for classification. Several works have provided theoretical insights into Neural Collapse under different loss functions, demonstrating that it emerges naturally in both cross-entropy loss [^12] [^9] [^18] and mean squared error (MSE) loss [^29] [^32] [^49] [^15]. Beyond theoretical exploration, NC has been applied to various machine-learning tasks. In imbalanced learning, NC principles have been leveraged to stabilize classifier prototypes under category-imbalanced training [^44] [^48]. In few-shot class-incremental learning, studies have explored ETF-based classifiers to enhance feature-classifier alignment and mitigate forgetting [^45]. In federated learning, NC has been shown to improve model generalization under non-iid settings [^17]. Similar prototype-based mechanisms have also been applied in domain-incremental scenarios, where consistent concept representations are maintained across domains [^41]. Furthermore, NC-inspired approaches have been developed to address large-scale classification problems, such as optimizing one-vs-rest margins for generalized NC [^19]. TRAILER [^43] also draws inspiration from NC; however, it employs a fixed classifier with a cross-entropy–based ETF loss applied to both labeled and pseudo-labeled data, which can cause optimization bias and instability. Our NC-GCD framework extends NC principles by pre-assigning ETF prototypes to known and novel categories. By integrating supervised and unsupervised ETF alignment, our method effectively addresses GCD’s misalignment and category confusion.

## 3 Preliminaries

### 3.1 Problem Setting of GCD

Generalized Category Discovery aims to identify novel categories in an unlabeled dataset while maintaining classification performance for known categories. Formally, we define a dataset $\mathcal{D}=\mathcal{D}^{l}\cup\mathcal{D}^{u}=\{(x_{i},y_{i})\}$, where the labeled dataset is given by $\mathcal{D}^{l}=\{(x_{i}^{l},y_{i}^{l})\}\subset\mathcal{X}\times\mathcal{Y}^{l}$, containing samples $x_{i}^{l}$ with corresponding labels $y_{i}^{l}$ from the set of known categories $\mathcal{Y}^{l}$. The unlabeled dataset is defined as $\mathcal{D}^{u}=\{x_{i}^{u}\}\subset\mathcal{X}$, which consists of samples from both known and novel categories. The overall category set is $\mathcal{Y}^{u}=\mathcal{Y}^{l}\cup\mathcal{Y}^{n}$, where $\mathcal{Y}^{l}\cap\mathcal{Y}^{n}=\emptyset$ and $\mathcal{Y}^{n}$ represents the unknown novel categories. The goal of GCD is to train a model that can accurately classify samples from $\mathcal{Y}^{l}$ while discovering and organizing samples from $\mathcal{Y}^{n}$.

### 3.2 Neural Collapse and Definition of Simplex ETF

Neural Collapse, observed during the terminal training phase in well-regularized neural networks on balanced datasets [^30], describes a symmetric geometric structure in the last-layer features and classifier weights. Features of the same category collapse to their within-category mean, which aligns with the corresponding classifier weights, forming a Simplex ETF.

Definition of Simplex ETF. A Simplex ETF consists of $K$ vectors in $\mathbb{R}^{d}$ that form an optimal geometric configuration. These vectors, denoted as $P=[p_{1},p_{2},\dots,p_{K}]\in\mathbb{R}^{d\times K}$, satisfy:

$$
P=\sqrt{\frac{K}{K-1}}U\left(I_{K}-\frac{1}{K}\mathbf{1}_{K}\mathbf{1}_{K}^{\top}\right),
$$

where $U\in\mathbb{R}^{d\times K}$ is an orthogonal matrix satisfying $U^{\top}U=I_{K}$, $I_{K}$ is the $K\times K$ identity matrix, and $\mathbf{1}_{K}$ is a $K$ -dimensional all-ones vector. Each prototype $p_{k}$ has unit $\ell_{2}$ -norm ($\|p_{k}\|=1$) and fixed pairwise cosine similarity, where $\delta_{k,j}$ is the Kronecker delta:

$$
p_{k}^{\top}p_{j}=\frac{K}{K-1}\delta_{k,j}-\frac{1}{K-1},\quad\forall k,j\in[1,K],
$$

Neural Collapse Phenomenon. The neural collapse phenomenon can be summarized as follows:

(NC1) Feature Collapse: Last-layer features of the same category collapse to their within-category mean: $\{\Sigma_{W}^{(k)}\to 0$, $\Sigma_{W}^{(k)}=\text{Avg}\{(\mu_{k,i}-\mu_{k})(\mu_{k,i}-\mu_{k})^{\top}\}$ where $\mu_{k,i}$ is the feature of the $i$ -th sample in the category $k$ and $\mu_{k}$ is the within-category mean.

(NC2) Convergence to Simplex ETF: The within-category means of all categories, centered by the global mean $\mu_{G}=\text{Avg}_{k,i}(\mu_{k,i})$, converge to the vertices of a Simplex ETF: $\hat{\mu}_{k}={\mu_{k}-\mu_{G}}/{\|\mu_{k}-\mu_{G}\|}$, $\forall k\in[1,K]$.

(NC3) Self-Duality: Within-category means align with the corresponding classifier weights: $\hat{\mu}_{k}={w_{k}}/{\|w_{k}\|}$, $w_{k}$ is the weight of the category $k$.

(NC4) Simplified Prediction: Model prediction simplifies to a nearest-category-center rule: $\hat{y}=\arg\min_{k}\|\mu-\mu_{k}\|=\arg\max_{k}\langle\mu,w_{k}\rangle,$ where $\mu$ is the feature of a test sample.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/main.jpg)

Figure 2: Overview of the NC-GCD Framework. The framework uses periodic clustering to group features, aligning them with pre-assigned ETF prototypes. The Semantic Consistency Matcher (SCM) ensures consistent label assignments across clustering iterations. This process stabilizes feature alignment and maintains a consistent geometric structure for both known and novel categories.

## 4 Method

### 4.1 Overview

As shown in Fig.2, our NC-GCD framework comprises four main components: a pre-trained visual encoder $f(\cdot)$, a periodic clustering module $g(\cdot)$, a pre-assigned ETF prototype set $P$, and a Semantic Consistency Matcher $\phi_{\text{SCM}}(\cdot)$.

First, we utilize the pre-trained visual encoder $f(\cdot)$ to extract the image embedding $e$ and augmented image embedding $e^{\prime}$ for each sample. These embeddings are passed to the periodic clustering module $g(\cdot)$, which groups samples into clusters and computes the clustering prototypes $\{c_{1},c_{2},\dots,c_{K}\}$. On the one hand, unsupervised ETF alignment pulls the top $\alpha$ % of samples most similar to their cluster centers toward their corresponding pre-assigned ETF prototypes using a dot-regression loss. On the other hand, supervised ETF alignment ensures that labeled samples are aligned with their corresponding ETF prototypes, maintaining geometric consistency.

However, two key challenges arise. First, clustering is inherently unstable, leading to inconsistencies where the same category may be assigned to different clusters across iterations, making it difficult to maintain alignment with pre-assigned ETF prototypes. Second, in supervised alignment, the direct mapping between true labels and ETF prototypes may not always hold, creating mismatches that affect model stability. To address these issues, we introduce the Semantic Consistency Matcher (SCM), which guarantees one-to-one matching between clusters over time, reducing inconsistencies caused by clustering dynamics. By integrating these components, our method effectively mitigates category confusion and label imbalance, ensuring a structured feature space and improved performance.

### 4.2 ETF Alignment

Pre-assigned ETF Prototype. We construct a Simplex ETF as pre-assigned classifier prototypes to enforce a structured feature space and maximize category separability. Given or estimated the number of categories $K$, we can get its categories set $\mathcal{Y}_{t}$. Then we define the optimal ETF prototypes $\mathbf{P}=\{p_{1},p_{2},\dots,p_{K}\}$ as Eq.1.

Each prototype $p_{i}\in\mathbb{R}^{d}$ maintains unit $\ell_{2}$ -norm and follows a fixed pairwise similarity:

$$
p_{i}^{\top}p_{j}=\frac{K}{K-1}q_{i}^{\top}q_{j}-\frac{1}{K-1},\quad\forall i,j\in[1,K].
$$

Here, $q_{i}$ denotes a centered orthonormal basis of a $(K-1)$ -dimensional space, with unit vectors satisfying $\sum_{i}q_{i}=\mathbf{0}$ and $q_{i}^{\top}q_{j}=\delta_{ij}$. By pre-assigning the ETF prototypes, we provide a stable and optimal alignment structure for both known and novel categories.

Unsupervised ETF Alignment. For each sample $x_{i}\in\mathcal{D}$, we first extract its image embedding $e_{i}=f(x_{i})$ and augmented embedding $e_{i}^{\prime}$. To introduce structural constraints and stabilize feature learning, we perform periodic clustering on the unlabeled dataset $\mathcal{D}$ every $T$ epochs. Given the extracted image embeddings $\{e_{1},\dots,e_{N}\}$, we apply clustering to obtain $K$ cluster centers $\{c_{1},\dots,c_{K}\}$. Then, each sample embedding $e_{i}$ is assigned to the nearest cluster based on the cosine similarity:

$$
\hat{y}_{i}=\arg\max_{k}\frac{e_{i}^{\top}c_{k}}{\|e_{k}\|\|c_{k}\|},\quad\forall x_{i}\in\mathcal{D}.
$$

Once pseudo-labels $\hat{y}_{i}$ are assigned, we compute the similarity between each sample and its respective cluster center, where $\mathcal{D}_{k}$ is the set of samples assigned to cluster k:

$$
s_{i}=\frac{e_{i}^{\top}c_{k}}{\|e_{i}\|\|c_{k}\|},\quad\forall x_{i}\in\mathcal{D}_{k}.
$$

To align features with the pre-assigned ETF prototype $\mathbf{P}=\{p_{1},p_{2},\dots,p_{K}\}$, we select the top $\alpha\%$ of samples with the highest similarity $s_{i}$ within each cluster. These high-confidence samples are pulled toward the corresponding ETF prototype $p_{k}$ using a Dot-Regression Loss [^44]:

$$
\mathcal{L}_{\text{ETF}}^{u}=\frac{1}{|\tilde{D}_{k}|}\sum_{e_{i}\in\tilde{D}_{k}}\|e_{i}-p_{k}\|^{2},
$$

where $\tilde{D}_{k}=\{x_{i}\mid s_{i}\text{ is in the top }\alpha\%\text{ of }\mathcal{D}_{k}\}$ represents the selected high-confidence samples assigned to cluster $c_{k}$.

Supervised ETF Alignment. For each sample $x_{i}^{l}\in\mathcal{D}^{l}$, we extract its feature embedding $e_{i}^{l}$. The SCM establishes a one-to-one mapping between predicted clusters and ground-truth labels, transforming the original labels $y_{i}^{l}$ into ETF-aligned labels $y_{i}^{\text{{ETF}}}=\phi_{\text{{SCM}}}({y}_{i}^{l})$ (detailed in Section 4.3). Subsequently, each labeled sample is encouraged to align with its assigned ETF prototype:

$$
\mathcal{L}_{\text{ETF}}^{s}=\frac{1}{|\mathcal{D}^{l}|}\sum_{x_{i}^{l}\in\mathcal{D}^{l}}\|e_{i}^{l}-p_{a}\|^{2},a\in[1,K]
$$

where $a=y_{i}^{\text{{ETF}}}$. This loss enforces geometric consistency between known category features and their corresponding ETF prototypes, ensuring a structured feature space.

Consistent ETF Alignment Loss. To balance the contributions of supervised and unsupervised ETF alignment, we define the hybrid ETF loss as:

$$
\mathcal{L}_{\text{ETF}}=(1-\gamma)\mathcal{L}_{\text{ETF}}^{u}+\gamma\mathcal{L}_{\text{ETF}}^{s},
$$

where $\gamma$ is a weighting coefficient determining.

### 4.3 Semantic Consistency Matcher (SCM)

In periodically clustering-based learning, two key challenges arise: 1) Pseudo-label instability across iterations. Clustering fluctuations introduce inconsistencies, disrupting the learning process and degrading feature alignment. 2) Misalignment between the predicted clusters and the true labels in supervised alignment. Supervised ETF alignment may misassign cluster labels, leading to further optimization instability. To mitigate these issues, we propose the SCM, which stabilizes pseudo-label assignments across iterations and ensures consistent alignment between predicted clusters and true labels. By enforcing a one-to-one mapping between clustering results from consecutive iterations, SCM reduces clustering-induced fluctuations.

Specifically, SCM aligns pseudo-labels between consecutive clustering iterations through an optimal assignment strategy. Given prediction sets $\{\hat{y}_{1}^{t},\dots,\hat{y}_{N}^{t}\},\quad\forall\hat{y}_{i}^{t}\in\mathcal{Y}_{t}$ for the current iteration and $\{\hat{y}_{1}^{t-1},\dots,\hat{y}_{N}^{t-1}\},\quad\forall\hat{y}_{i}^{t-1}\in\mathcal{Y}_{t-1}$ for the previous iteration, where $\mathcal{Y}_{t}$ and $\mathcal{Y}_{t-1}$ represent the corresponding label sets. Then, we seek an optimal permutation $\sigma^{*}$ that maximizes label consistency:

$$
\sigma^{*}=\arg\max_{\sigma\in S_{K}}\sum_{k=1}^{K}\sum_{i=1}^{N}\mathbb{I}(\hat{y}_{i}^{t}=k)\mathbb{I}(\hat{y}_{i}^{t-1}=\sigma(k)),
$$

where $\sigma$ is a bi-jective mapping function ensuring a one-to-one correspondence between clusters from consecutive iterations, and $S_{K}$ is the space of all valid permutations over $K$ clusters. The indicator function $\mathbb{I}(\cdot)$ returns 1 if the condition holds and 0 otherwise. The label set in the current iteration is updated as $\mathcal{Y}_{t}=\sigma^{*}(\mathcal{Y}_{t-1})$.

SCM also ensures consistency in supervised ETF alignment by optimally mapping predicted clusters to their corresponding ETF-aligned labels. Given the prediction of the labeled samples and the set of ground truth labels, SCM applies the same optimal assignment strategy to obtain the permutation $\sigma^{l}$. Using this mapping, the ETF-aligned label set is updated as: $\mathcal{Y}_{t}^{\text{{ETF}}}=\sigma^{l}(\mathcal{Y}^{l})$.

### 4.4 Comparative Representation Learning

Following CMS [^6], we adopt the unsupervised loss encourages similarity between embeddings of the same sample while distinguishing them from other samples:

$$
\mathcal{L}_{\text{REP}}^{u}=-\frac{1}{|B|}\sum_{i\in B}\log\frac{\exp(e_{i}\cdot e_{i}^{\prime}/\tau)}{\sum_{j\neq i}\exp(e_{i}\cdot e_{j}/\tau)},
$$

where $e_{i}^{\prime}$ is the augmented view of $e_{i}$, $\tau$ is the temperature parameter, and $|B|$ is the batch size.

For labeled samples, we apply a supervised loss to enforce compact intra-category representations:

$$
\mathcal{L}_{\text{REP}}^{s}=-\frac{1}{|B_{l}|}\sum_{i\in B_{l}}\frac{1}{|H(i)|}\sum_{h\in H(i)}\log\frac{\exp(e_{i}^{l}\cdot e_{h}/\tau)}{\sum_{j\neq i}\exp(e_{i}^{l}\cdot e_{j}/\tau)},
$$

where $H(i)$ represents the samples sharing the same label $i$. The representation learning objective is:

$$
\mathcal{L}_{\text{REP}}=(1-\lambda)\mathcal{L}_{\text{REP}}^{u}+\lambda\mathcal{L}_{\text{REP}}^{s},
$$

where $\lambda$ balances the contributions of unsupervised and supervised components.

### 4.5 Final Objective

The final loss function integrates all components to ensure stable feature alignment and category separability:

$$
\mathcal{L}=\beta\mathcal{L}_{\text{ETF}}+\mathcal{L}_{\text{REP}},
$$

where $\beta$ controls the contributions of ETF loss and representation learning loss.

Table 1: Comparison with the state of the arts on GCD, evaluated with or without the GT K for clustering, with DINOv1 backbone.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="3">CUB</td><td colspan="3">Stanford Cars</td><td colspan="3">FGVC Aircraft</td><td colspan="3">Herbarium 19</td><td colspan="3">CIFAR100</td><td colspan="3">ImageNet100</td></tr><tr><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><th colspan="19">(a) Clustering with the ground-truth number of categories <math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math> given</th></tr><tr><th>GCD (CVPR22)</th><td>51.3</td><td>56.6</td><td>48.7</td><td>39.0</td><td>57.6</td><td>29.9</td><td>45.0</td><td>41.1</td><td>46.9</td><td>35.4</td><td>51.0</td><td>27.0</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td></tr><tr><th>DCCL (CVPR23)</th><td>63.5</td><td>60.8</td><td>64.9</td><td>43.1</td><td>55.7</td><td>36.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>75.3</td><td>76.8</td><td>70.2</td><td>80.5</td><td>90.5</td><td>76.2</td></tr><tr><th>PromptCAL</th><td>62.9</td><td>64.4</td><td>62.1</td><td>50.2</td><td>70.1</td><td>40.6</td><td>52.2</td><td>52.2</td><td>52.3</td><td>37.0</td><td>52.0</td><td>28.9</td><td>81.2</td><td>84.2</td><td>75.3</td><td>83.1</td><td>92.7</td><td>78.3</td></tr><tr><th>GPC (ICCV23)</th><td>55.4</td><td>58.2</td><td>53.1</td><td>42.8</td><td>59.2</td><td>32.8</td><td>46.3</td><td>42.5</td><td>47.9</td><td>-</td><td>-</td><td>-</td><td>77.9</td><td>85.0</td><td>63.0</td><td>76.9</td><td>94.3</td><td>71.0</td></tr><tr><th>SimGCD (ICCV23)</th><td>60.3</td><td>65.6</td><td>57.7</td><td>53.8</td><td>71.9</td><td>45.0</td><td>54.2</td><td>59.1</td><td>51.8</td><td>44.0</td><td>58.0</td><td>36.4</td><td>80.1</td><td>81.2</td><td>77.8</td><td>83.0</td><td>93.1</td><td>77.9</td></tr><tr><th>PIM (CVPR23)</th><td>62.7</td><td>75.7</td><td>56.2</td><td>43.1</td><td>66.9</td><td>31.6</td><td>-</td><td>-</td><td>-</td><td>42.3</td><td>56.1</td><td>34.8</td><td>78.3</td><td>84.2</td><td>66.5</td><td>83.1</td><td>95.3</td><td>77.0</td></tr><tr><th>TRAILER (CVPR24)</th><td>65.1</td><td>71.3</td><td>61.9</td><td>55.4</td><td>71.7</td><td>47.6</td><td>54.5</td><td>62.6</td><td>50.5</td><td>44.5</td><td>57.0</td><td>37.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>SelEx (ECCV24)</th><td>73.6</td><td>75.3</td><td>72.8</td><td>58.5</td><td>75.6</td><td>50.3</td><td>57.1</td><td>64.7</td><td>53.3</td><td>-</td><td>-</td><td>-</td><td>82.3</td><td>85.3</td><td>76.3</td><td>83.1</td><td>93.6</td><td>77.8</td></tr><tr><th>CMS (CVPR24)</th><td>68.2</td><td>76.5</td><td>64.0</td><td>56.9</td><td>76.1</td><td>47.6</td><td>56.0</td><td>63.4</td><td>52.3</td><td>36.4</td><td>54.9</td><td>26.4</td><td>82.3</td><td>85.7</td><td>75.5</td><td>84.7</td><td>95.6</td><td>79.2</td></tr><tr><th>SPT (ICLR24)</th><td>65.8</td><td>68.8</td><td>65.1</td><td>59.0</td><td>79.2</td><td>49.3</td><td>59.3</td><td>61.8</td><td>58.1</td><td>43.4</td><td>58.7</td><td>35.2</td><td>81.3</td><td>84.3</td><td>75.6</td><td>85.4</td><td>93.2</td><td>81.4</td></tr><tr><th>Ours (NC-GCD)</th><td>74.8</td><td>76.8</td><td>73.8</td><td>59.9</td><td>77.8</td><td>51.2</td><td>60.0</td><td>57.6</td><td>61.2</td><td>46.4</td><td>58.4</td><td>40.7</td><td>82.7</td><td>85.5</td><td>77.3</td><td>88.4</td><td>94.1</td><td>85.5</td></tr><tr><th colspan="19">(b) Clustering without the ground-truth number of categories <math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math> given</th></tr><tr><th>GCD (CVPR22)</th><td>51.1</td><td>56.4</td><td>48.4</td><td>39.1</td><td>58.6</td><td>29.7</td><td>-</td><td>-</td><td>-</td><td>37.2</td><td>51.7</td><td>29.4</td><td>70.8</td><td>77.6</td><td>57.0</td><td>77.9</td><td>91.1</td><td>71.3</td></tr><tr><th>GPC (ICCV23)</th><td>52.0</td><td>55.5</td><td>47.5</td><td>38.2</td><td>58.9</td><td>27.4</td><td>43.3</td><td>40.7</td><td>44.8</td><td>36.5</td><td>51.7</td><td>27.9</td><td>75.4</td><td>84.6</td><td>60.1</td><td>75.3</td><td>93.4</td><td>66.7</td></tr><tr><th>PIM (CVPR23)</th><td>62.0</td><td>75.7</td><td>55.1</td><td>42.4</td><td>65.3</td><td>31.3</td><td>-</td><td>-</td><td>-</td><td>42.0</td><td>55.5</td><td>34.7</td><td>75.6</td><td>81.6</td><td>63.6</td><td>83.0</td><td>95.3</td><td>76.9</td></tr><tr><th>CMS (CVPR24)</th><td>64.4</td><td>68.2</td><td>62.2</td><td>51.7</td><td>68.9</td><td>43.4</td><td>55.2</td><td>60.6</td><td>52.4</td><td>37.4</td><td>56.5</td><td>27.1</td><td>79.6</td><td>83.2</td><td>72.3</td><td>81.3</td><td>95.6</td><td>74.2</td></tr><tr><th>Ours (NC-GCD)</th><td>70.3</td><td>72.1</td><td>69.4</td><td>54.0</td><td>73.1</td><td>44.8</td><td>55.4</td><td>57.3</td><td>54.5</td><td>42.3</td><td>56.2</td><td>34.8</td><td>80.5</td><td>83.7</td><td>74.0</td><td>85.7</td><td>95.9</td><td>80.6</td></tr></tbody></table>

## 5 Experiments

### 5.1 Experimental Setup

Datasets. Our evaluation spans six image classification benchmarks, including both generic and fine-grained datasets. For generic datasets, we use CIFAR-100 [^23] and ImageNet-100 [^7]. For fine-grained datasets, we evaluate on CUB-200 [^39], Stanford Cars [^22], FGVC Aircraft [^28], and Herbarium19 [^36]. To separate categories into known and novel categories, we follow the SSB split protocol [^37] for the fine-grained datasets. For CIFAR-100 and ImageNet-100, we perform a random category split using a fixed seed, consistent with previous studies.

Implementation Details. Our method utilizes the pre-trained DINO ViT-B/16 [^3] [^8] as the backbone network. We fine-tune the final layer of the image encoder along with a projection head, following prior work [^38] [^6]. The projection head consists of an MLP with a 768-dimensional input, a 2048-dimensional hidden layer, and a 768-dimensional output, followed by a GeLU activation function [^16]. The learning rate is set to 0.1. Other hyperparameters, including batch size, temperature $\tau_{s}$, weight decay $\lambda$, and the number of augmentations, are set to 128, 0.07, $1e^{-4}$, and 2, respectively, in accordance with previous studies. All experiments are conducted on an NVIDIA 3090 GPU. Further implementation details can be found in the Appendix.

### 5.2 Comparison with State of the Art Methods

We evaluate our NC-GCD framework on several fine-grained and generic classification benchmarks to demonstrate its efficacy in GCD. Table 1 presents a comparison between our method and the previous methods [^38] [^33] [^46] [^47] [^42] [^5] [^6] [^40] [^34] [^43], evaluated on both fine-grained and generic datasets with and without access to the ground-truth (GT) number of categories K for clustering. For the case without access to GT K, we estimate the K through clustering, which is then used to align the ETF prototypes. Meanwhile, to comprehensively evaluate the performance of various methods across Fine-Grained Datasets, Generic Classification Datasets, and All datasets, Table 2 presents the average performance.

Fine-Grained Datasets. On fine-grained benchmarks such as CUB-200, Stanford Cars, FGVC Aircraft, and Herbarium19, NC-GCD outperforms previous methods, whether or not the ground-truth K is provided. As shown in Table 2a, when the GT K is available, NC-GCD achieves an average all-category accuracy of 60.3%, surpassing SPT by 3.4% and improving novel category accuracy over SPT by 4.8%. Even without GT K, as shown in Table 2b, NC-GCD remains robust, surpassing CMS by 3.3% in all-category accuracy. Notably, on CUB-200, NC-GCD outperforms CMS by 5.9% in all-category accuracy and 7.2% in novel category accuracy, demonstrating strong generalization without the number of categories. Fine-grained datasets present higher category confusion due to minimal intra-category variation, and our method effectively mitigates this and enhances category separability.

Generic Classification Datasets. NC-GCD achieves strong performance on standard classification datasets such as CIFAR-100 and ImageNet-100, as shown in Table 2. With known K, NC-GCD attains an all-category accuracy of 82.7% on CIFAR-100 and 88.4% on ImageNet-100, outperforming CMS and SPT. Notably, novel category accuracy improves by 1.7% on CIFAR-100 and 4.1% on ImageNet-100. When K must be estimated, NC-GCD maintains its competitive edge, achieving 85.7% all-category accuracy on ImageNet-100, surpassing CMS by 4.4%, demonstrating its adaptability in generic classification scenarios.

Across both fine-grained and generic datasets, NC-GCD consistently achieves well performance. As shown in Table 2a, with GT K, NC-GCD reaches the highest all-category accuracy of 68.7%, surpassing SPT by 3.0% and improving novel category accuracy by 4.1%. Even without GT K, NC-GCD remains robust, surpassing CMS by 3.1% in all-category accuracy and 4.4% in novel category accuracy. By pre-assigning ETF prototypes, NC-GCD effectively improves category separability, mitigating category confusion and ensuring balanced learning across old and novel categories. These results highlight the robustness and versatility of NC-GCD for real-world GCD applications.

Table 2: Comparison of various methods across multiple metrics, including Fine-grained datasets, Classification datasets, and All datasets.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="3">Fine-grained Avg</td><td colspan="3">Classification Avg</td><td colspan="3">All Avg</td></tr><tr><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><th colspan="10">(a) Clustering with the ground-truth number of categories K given</th></tr><tr><th>SimGCD (ICCV 2023)</th><td>53.1</td><td>63.7</td><td>47.7</td><td>81.6</td><td>87.2</td><td>77.9</td><td>62.6</td><td>71.5</td><td>57.8</td></tr><tr><th>CMS (CVPR 2024)</th><td>54.4</td><td>67.7</td><td>47.6</td><td>83.5</td><td>90.7</td><td>77.4</td><td>64.1</td><td>75.4</td><td>57.5</td></tr><tr><th>SPT (ICLR 2024)</th><td>56.9</td><td>67.1</td><td>51.9</td><td>83.4</td><td>88.8</td><td>78.5</td><td>65.7</td><td>74.3</td><td>60.8</td></tr><tr><th>Ours (NC-GCD)</th><td>60.3</td><td>67.6</td><td>56.7</td><td>85.5</td><td>89.8</td><td>81.4</td><td>68.7</td><td>75.0</td><td>64.9</td></tr><tr><th>Improv. over SPT</th><td>+3.4</td><td>+0.5</td><td>+4.8</td><td>+2.1</td><td>+1.0</td><td>+2.9</td><td>+3.0</td><td>+0.7</td><td>+4.1</td></tr><tr><th colspan="10">(b) Clustering without the ground-truth number of categories K given</th></tr><tr><th>GPC (ICCV 2023)</th><td>46.2</td><td>50.8</td><td>38.9</td><td>75.4</td><td>89.0</td><td>63.4</td><td>53.5</td><td>64.1</td><td>45.7</td></tr><tr><th>CMS (CVPR 2024)</th><td>52.2</td><td>63.6</td><td>46.3</td><td>80.5</td><td>89.4</td><td>73.3</td><td>61.6</td><td>72.2</td><td>55.3</td></tr><tr><th>Ours (NC-GCD)</th><td>55.5</td><td>64.6</td><td>50.9</td><td>83.1</td><td>89.8</td><td>77.3</td><td>64.7</td><td>73.0</td><td>59.7</td></tr><tr><th>Improv. over CMS</th><td>+3.3</td><td>+1.0</td><td>+4.6</td><td>+2.6</td><td>+0.4</td><td>+4.0</td><td>+3.1</td><td>+0.8</td><td>+4.4</td></tr></tbody></table>

### 5.3 Ablation Study

The Effectiveness of ETF Alignment. To evaluate the impact of Unsupervised ETF Alignment and Supervised ETF Alignment, we conduct ablation studies with four configurations: (1) Baseline (both disabled), (2) Supervised ETF only, (3) Unsupervised ETF only, and (4) Full model (both enabled). Table 3 presents the results, revealing key insights into each component’s contribution.

Unsupervised ETF Alignment: This component yields the most significant improvements, especially in fine-grained datasets. As shown in Table 3, enabling unsupervised ETF alignment alone leads to substantial gains in novel category accuracy. The model effectively learns robust representations, particularly for novel categories, where differentiation between visually similar categories is crucial.

Supervised ETF Alignment: This component stabilizes and preserves old categories’ accuracy. By aligning labeled features with their corresponding ETF prototypes, it ensures the retention of learned knowledge while maintaining high performance on known categories. Though its contribution to novel category accuracy is smaller than that of unsupervised alignment, it plays a crucial role in preventing forgetting and maintaining model stability.

Synergistic Combination: The full model, integrating both alignments, achieves the highest overall accuracy, with an average all-category accuracy of 68.7%, significantly surpassing the baseline. This combination enhances novel category discovery while preserving old category performance. Specifically, the average novel category accuracy increases by 7.4%, and the overall accuracy improves by 5.4%. The results highlight the synergy between supervised and unsupervised alignment, demonstrating the NC-GCD effectiveness in balancing learning across known and novel categories.

Table 3: Performance comparison with Supervised and Unsupervised ETF Alignment.

<table><thead><tr><th rowspan="2">Sup ETF Alignment</th><th rowspan="2">Unsup ETF Alignment</th><th colspan="3">CUB</th><th colspan="3">Herbarium 19</th><th colspan="3">ImageNet100</th><th colspan="3">Six Datasets Avg</th></tr><tr><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><td>✗</td><td>✗</td><td>67.7</td><td>75.7</td><td>63.9</td><td>36.5</td><td>55.0</td><td>26.5</td><td>84.4</td><td>94.0</td><td>80.8</td><td>63.3</td><td>73.7</td><td>57.5</td></tr><tr><td>✓</td><td>✗</td><td>69.6</td><td>75.8</td><td>66.5</td><td>37.6</td><td>56.7</td><td>28.1</td><td>85.0</td><td>95.3</td><td>79.3</td><td>64.6</td><td>75.2</td><td>58.3</td></tr><tr><td>✗</td><td>✓</td><td>75.7</td><td>71.3</td><td>77.8</td><td>47.2</td><td>60.0</td><td>40.3</td><td>87.6</td><td>94.5</td><td>84.1</td><td>68.0</td><td>73.2</td><td>64.9</td></tr><tr><td>✓</td><td>✓</td><td>74.8</td><td>76.8</td><td>73.8</td><td>46.4</td><td>58.4</td><td>40.7</td><td>88.4</td><td>94.1</td><td>85.5</td><td>68.7</td><td>74.9</td><td>64.9</td></tr><tr><td colspan="2">Improv. over baseline</td><td>+7.1</td><td>+1.1</td><td>+9.9</td><td>+9.9</td><td>+3.4</td><td>+14.2</td><td>+4.0</td><td>+0.1</td><td>+4.7</td><td>+5.4</td><td>+1.2</td><td>+7.4</td></tr></tbody></table>

Table 4: Ablation study of our SCM module.

<table><thead><tr><th rowspan="2">SCM</th><th colspan="3">CUB</th><th colspan="3">Herbarium 19</th><th colspan="3">ImageNet100</th></tr><tr><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><td>×</td><td>70.3</td><td>71.0</td><td>70.0</td><td>42.6</td><td>54.3</td><td>36.3</td><td>84.3</td><td>88.9</td><td>82.0</td></tr><tr><td>✓</td><td>75.7</td><td>71.3</td><td>77.8</td><td>47.2</td><td>60.0</td><td>40.3</td><td>87.6</td><td>94.5</td><td>84.1</td></tr><tr><td>Improv.</td><td>+5.4</td><td>+0.3</td><td>+7.8</td><td>+4.6</td><td>+5.7</td><td>+4.0</td><td>+3.3</td><td>+5.6</td><td>+2.1</td></tr></tbody></table>

The Effectiveness of SCM. To evaluate the impact of the Semantic Consistency Matcher (SCM), we conduct ablation studies on CUB-200, ImageNet100, and Herbarium 19. As shown in Table 5.3, enabling SCM consistently improves performance on all three datasets. On Herbarium 19, which poses greater challenges due to its large number of categories and high intra-category variance, SCM provides more significant improvements, with gains of 4.6% for all categories and 5.7% for novel categories, demonstrating its effectiveness in complex classification tasks. By mitigating clustering fluctuations, SCM ensures stable pseudo-label assignments. These results highlight its crucial role in stabilizing training, particularly in datasets with large category numbers.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/Ab-ETF-Scope-cub.png)

Figure 3: Ablation study of the α \\alpha on CUB-200.

The Influence of the ETF Scope Coefficient $\alpha$. The ETF Scope Coefficient $\alpha$ plays a crucial role in determining the number of samples selected for alignment with the ETF prototype. By controlling the percentage of high-confidence samples, $\alpha$ influences the model’s ability to focus on either old or novel categories. As seen in Fig. 3, when $\alpha$ is too small, the model may not leverage enough high-confidence samples, leading to suboptimal performance. On the other hand, as $\alpha$ increases, novel category accuracy improves significantly, especially in fine-grained datasets where category distinctions are subtle. However, when $\alpha$ exceeds a certain threshold, the model’s focus on new categories comes at the expense of old category accuracy, highlighting the trade-off between the two. The best balance is achieved at $\alpha=0.8$, where both old and novel category accuracy reach optimal values, demonstrating the importance of selecting an appropriate $\alpha$.

Ablation study of ETF loss coefficient $\beta$, more results of the effectiveness of ETF Alignment and Parameter analysis will be presented in the Appendix.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/vis-cub-10class.jpg)

Figure 4: Visualization of Comparison with DINOv1, CMS, and our method on CUB-200.

### 5.4 Visualization

Comparison with Previous Methods. As shown in Fig. 4, we compare the embeddings extracted by DINOv1 (a), CMS (b), and our method (c) for ten categories, with (d) focusing on a subset of three categories from (c). The embeddings from DINO-V1 and CMS show significant overlap between categories, whereas our method clearly separates the categories into distinct clusters. Notably, categories 17, 74, and 104 collapse almost into a single point in (c), which aligns with the Neural Collapse hypothesis. This demonstrates that our method enhances feature discriminability, improving category separability across both known and novel categories.

## 6 Conclusion and Future Work

We propose NC-GCD, a framework for Generalized Category Discovery that integrates unsupervised and supervised ETF alignment to enhance category separability. Extensive experiments demonstrate that NC-GCD consistently outperforms previous methods in both known and novel category accuracy. Ablation studies validate the complementary effects of the two alignment strategies, while t-SNE visualizations highlight its effectiveness in distinguishing similar categories, aligning with the Neural Collapse hypothesis. A promising future research direction is extending our NC-GCD to incremental GCD settings.

## 7 Acknowledgements

This work is supported by the National Natural Science Foundation of China under Grant No.U21B2048 and No.62302382, the Shenzhen Key Technical Projects Under Grant CJGJZD2022051714160501, the China Postdoctoral Science Foundation No.2024M752584, and the Joint Fund Project of the National Natural Science Foundation of China under Grant No.U24A20328.

## References

## Appendix Contents

1. More Implementation Details
	1. Training Parameters and Details........................................................................................................................................................................A.1
	2. Dataset Details and Category Splits........................................................................................................................................................................A.2
	3. Pseudocode of SCM........................................................................................................................................................................A.3
2. More Experiment Results and Ablation Study
	1. Effectiveness of ETF Alignment........................................................................................................................................................................B.1
	2. Main Results with Different Backbones........................................................................................................................................................................B.2
	3. Influence of ETF Loss Coefficient $\beta$........................................................................................................................................................................B.3
	4. Impact of Incorrect Estimation of Category Number $K$........................................................................................................................................................................B.4
	5. Performance under Estimated Category Number $K$........................................................................................................................................................................B.5
	6. Clustering Frequency $T$ Analysis........................................................................................................................................................................B.6
3. Parameter and SCM Efficiency Analysis........................................................................................................................................................................C
	1. Parameter Analysis........................................................................................................................................................................C.1
	2. SCM Efficiency........................................................................................................................................................................C.2
4. More Visualization
	1. Comparison with State-of-the-Art Methods........................................................................................................................................................................D.1
	2. Visualization of Training Process........................................................................................................................................................................D.2
5. Limitations and Future Work........................................................................................................................................................................E
6. Broader Impacts........................................................................................................................................................................F

## Appendix A More Implementation Details

### A.1 Training Parameters and Details

In this study, we evaluate the NC-GCD framework on various datasets using a pre-trained DINO ViT-B/16 model as the backbone. The final layer of the image encoder is fine-tuned alongside a projection head for both supervised and unsupervised alignment tasks, as per previous works [^38] [^6]. The projection head is a Multi-Layer Perceptron (MLP) consisting of a 768-dimensional input layer, a 2048-dimensional hidden layer, and a 768-dimensional output layer, followed by a GeLU activation function [^16]. We empirically set periodic clustering epoch T=5 as a trade-off between stability and computational efficiency. Larger T may reduce adaptation speed, while smaller T increases computational cost.

The learning rate is set to 0.1, and other key hyperparameters such as batch size (128), temperature ($\tau_{s}=0.07$), weight decay ($1e^{-4}$), and the number of augmentations (2) follow the settings used in prior works [^38]. All experiments are conducted using an NVIDIA 3090 GPU, which provides sufficient power for the fine-grained operations required in our method.

For training, we use the SGD optimizer with a momentum of 0.9 and weight decay of $1e^{-4}$. We apply a cosine annealing learning rate scheduler to adapt the learning rate during the course of training. The method is trained for a total of 200 epochs. Additionally, data augmentation and random cropping are applied to enhance the model’s ability to generalize across different transformations.

In our model, we combine supervised and unsupervised contrastive losses and the ETF alignment loss. The unsupervised ETF alignment uses a clustering method for periodic feature grouping, while the supervised ETF alignment ensures that labeled data is aligned with their corresponding ETF prototypes. During training, we use a weighted random sampler to balance labeled and unlabeled data in each batch, allowing the model to learn from both known and novel categories effectively.

### A.2 Dataset Details and Category Splits

We evaluate our proposed framework on six diverse datasets covering general and fine-grained image classification tasks. Table A.2 provides an overview of the category distributions for each dataset, categorizing them as known and unknown categories for assessing Generalized Category Discovery (GCD) performance. Below, we summarize the details and category split protocols for each dataset.

Table 5: Distribution of labeled (known) and unlabeled (novel) categories across datasets. Labeled categories have annotated training samples and serve as known categories for supervision, while unlabeled categories represent novel categories that the model must discover without labels.

| Dataset | Labeled Categories | Unlabeled Categories |
| --- | --- | --- |
| CIFAR100 [^23] | 80 | 20 |
| ImageNet100 [^7] | 50 | 50 |
| CUB-200-2011 [^39] | 100 | 100 |
| Stanford Cars [^22] | 98 | 98 |
| FGVC-Aircraft [^28] | 50 | 50 |
| Herbarium19 [^36] | 341 | 342 |

CIFAR-100. CIFAR-100 [^23] consists of 60,000 images across 100 categories. For the GCD evaluation, we used 80 categories as known categories for training and the remaining 20 as unknown categories for testing.

ImageNet100. ImageNet100 [^7] is a subset of ImageNet with 100 categories. Following standard protocols, we randomly assign 50 categories as known categories and the other 50 as unknown categories.

CUB-200-2011. CUB-200-2011 [^39] is a fine-grained bird dataset with 200 categories. We split the dataset into 100 known and 100 unknown categories following the standard fine-grained GCD protocol [^37].

Stanford Cars. Stanford Cars [^22] is a fine-grained dataset with 196 car categories. We use 98 categories for training (known categories) and 98 for testing (unknown categories), consistent with previous fine-grained GCD work.

FGVC Aircraft. FGVC Aircraft [^28] includes 100 aircraft categories. The dataset is split into 50 known and 50 unknown categories, adhering to the SSB protocol.

Herbarium19. Herbarium19 [^36] is a large-scale fine-grained dataset with 683 plant species. Due to its high intra-category variance, we split the dataset into 341 known and 342 unknown categories to evaluate GCD on more challenging fine-grained datasets.

Category Split Protocol. For general datasets (CIFAR-100 and ImageNet100), we apply a random category split using a fixed seed to ensure reproducibility. For fine-grained datasets (CUB-200-2011, Stanford Cars, and FGVC Aircraft), we use the SSB split protocol [^37], which ensures balanced category distributions while preserving intra-category variance. For Herbarium19, the dataset is split into nearly equal known and unknown categories, testing the model’s robustness in handling datasets with significant intra-category variation.

Algorithm 1 Semantic Consistency Matcher (SCM)

Input:
- Prediction sets from the current iteration: $\{\hat{y}_{1}^{t},\dots,\hat{y}_{N}^{t}\},\quad\forall\hat{y}_{i}^{t}\in\mathcal{Y}_{t}$
- Prediction sets from the previous iteration: $\{\hat{y}_{1}^{t-1},\dots,\hat{y}_{N}^{t-1}\},\quad\forall\hat{y}_{i}^{t-1}\in\mathcal{Y}_{t-1}$
- Prediction sets of labeled samples from the current iteration: $\{\hat{y}_{1}^{tl},...,\hat{y}_{M}^{tl}\},\quad\forall\hat{y}_{i}^{t}\in\mathcal{Y}_{t}^{l}$
- Ground-truth labels sets: $\{y_{1}^{l},...,y_{M}^{l}\},\quad\forall{y}_{i}^{l}\in\mathcal{Y}^{l}$
- Number of clusters: $K$

Output:
- Updated label set: $\mathcal{Y}_{t}\leftarrow\sigma^{*}(\mathcal{Y}_{t-1})$
- Updated supervised ETF-aligned label set: $\mathcal{Y}_{t}^{ETF}\leftarrow\sigma^{l}(\mathcal{Y}^{l})$

Define permutation set: 
$$
S_{K}=\{\sigma\mid\sigma:\{1,2,\dots,K\}\rightarrow\{1,2,\dots,K\},\sigma\text{ is bi-jective mapping function}\}
$$

Step 1: Optimal Alignment for Prediction (Unsupervised)

  Find an optimal one-to-one mapping $\sigma^{*}$ between clusters at iteration $t-1$ and $t$: 
$$
\sigma^{*}=\arg\max_{\sigma\in S_{K}}\sum_{k=1}^{K}\sum_{i=1}^{N}\mathbb{I}(\hat{y}_{i}^{t}=k)\mathbb{I}(\hat{y}_{i}^{t-1}=\sigma(k))
$$

  Here, $S_{K}$ denotes the set of all possible permutations (bi-jective mappings) over $K$ clusters

  Update pseudo-labels for the current iteration: 
$$
\mathcal{Y}_{t}\leftarrow\sigma^{*}(\mathcal{Y}_{t-1})
$$

Step 2: Optimal Alignment for Supervised ETF Labels

  Find an optimal one-to-one mapping $\sigma^{l}$ between predicted labels and ground-truth labels: 
$$
\sigma^{l}=\arg\max_{\sigma\in S_{J}}\sum_{j=1}^{J}\sum_{i=1}^{M}\mathbb{I}(\hat{y}_{i}^{tl}=j)\mathbb{I}(y_{i}^{l}=\sigma(j))
$$

  Here, $S_{J}$ denotes the set of all possible permutations (bi-jective mappings) over $J$ clusters

  Update ETF-aligned labels for supervised samples: 
$$
\mathcal{Y}_{t}^{ETF}\leftarrow\sigma^{l}(\mathcal{Y}^{l})
$$

Return: Updated label set $\mathcal{Y}_{t}$ and ETF-aligned supervised label set $\mathcal{Y}_{t}^{ETF}$

### A.3 Pseudocode of SCM

Semantic Consistency Matcher (SCM) addresses two key issues in generalized category discovery tasks: the instability of pseudo-label assignments caused by periodic clustering and the misalignment between predicted labels and ground-truth labels in supervised ETF alignment. Specifically, SCM operates in two steps, as detailed in Algorithm 1.

(1) Unsupervised Scenario (Pseudo-label Alignment): Given a prediction set from the current iteration with $N$ samples, let $\{\hat{y}_{1}^{t},\dots,\hat{y}_{N}^{t}\}$, where $\hat{y}_{i}^{t}\in\mathcal{Y}_{t}$. Similarly, prediction sets from the previous iteration denote as: $\{\hat{y}_{1}^{t-1},\dots,\hat{y}_{N}^{t-1}\}$, where $\hat{y}_{i}^{t-1}\in\mathcal{Y}_{t-1}$. Both $\mathcal{Y}_{t}$ and $\mathcal{Y}_{t-1}$ contain $K$ cluster labels. SCM finds an optimal bijective mapping $\sigma^{*}$ between clusters of consecutive iterations by maximizing their agreement. $S_{K}$ denotes all possible permutations (one-to-one mappings) over $K$ clusters. The indicator function $\mathbb{I}(\cdot)$ returns 1 if the condition is satisfied and 0 otherwise. The pseudo-label set at iteration $t$ is then updated as: $\mathcal{Y}_{t}=\sigma^{*}(\mathcal{Y}_{t-1})$.

(2) Supervised Scenario: In the supervised setting, given $M$ labeled samples, $\{\hat{y}_{1}^{tl},\dots,\hat{y}_{M}^{tl}\}$ denote the predicted sets of labels samples from the current iteration $t$. Let $\{y_{1}^{l},\dots,y_{M}^{l}\}$ denote the ground-truth labels, each belonging to the true label set $\mathcal{Y}^{l}$ with cardinality $J$. SCM identifies an optimal one-to-one mapping $\sigma^{l}$ between predicted and ground-truth labels by maximizing their consistency.$S_{J}$ denotes all possible permutations over the $J$ known categories. The supervised ETF-aligned label set is then updated as: $\mathcal{Y}_{t}^{\text{ETF}}=\sigma^{l}(\mathcal{Y}^{l})$.

## Appendix B More Experiment Results and Ablation Study

Table 6: Performance comparison on CUB, Stanford Cars, FGVC Aircraft, and Herbarium 19.

<table><thead><tr><th rowspan="2">Sup ETF Alignment</th><th rowspan="2">Unsup ETF Alignment</th><th colspan="3">CUB</th><th colspan="3">Stanford Cars</th><th colspan="3">FGVC Aircraft</th><th colspan="3">Herbarium 19</th></tr><tr><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><td>✗</td><td>✗</td><td>67.7</td><td>75.7</td><td>63.9</td><td>55.1</td><td>74.3</td><td>45.8</td><td>54.3</td><td>57.4</td><td>52.8</td><td>36.5</td><td>55.0</td><td>26.5</td></tr><tr><td>✓</td><td>✗</td><td>69.6</td><td>75.8</td><td>66.5</td><td>57.0</td><td>75.1</td><td>46.7</td><td>56.1</td><td>62.9</td><td>53.2</td><td>37.6</td><td>56.7</td><td>28.1</td></tr><tr><td>✗</td><td>✓</td><td>75.7</td><td>71.3</td><td>77.8</td><td>56.0</td><td>74.7</td><td>46.9</td><td>59.8</td><td>54.7</td><td>62.3</td><td>47.2</td><td>60.0</td><td>40.3</td></tr><tr><td>✓</td><td>✓</td><td>74.8</td><td>76.8</td><td>73.8</td><td>59.9</td><td>77.8</td><td>51.2</td><td>60.0</td><td>57.6</td><td>61.2</td><td>46.4</td><td>58.4</td><td>40.7</td></tr><tr><td colspan="2">Improv. over baseline</td><td>+7.1</td><td>+1.1</td><td>+9.9</td><td>+4.8</td><td>+3.5</td><td>+5.4</td><td>+5.7</td><td>+0.2</td><td>+8.4</td><td>+9.9</td><td>+3.4</td><td>+14.2</td></tr></tbody></table>

Table 7: Performance comparison on CIFAR100, ImageNet100, and Six Datasets Average results.

<table><thead><tr><th rowspan="2">Sup ETF Alignment</th><th rowspan="2">Unsup ETF Alignment</th><th colspan="3">CIFAR100</th><th colspan="3">ImageNet100</th><th colspan="3">Six Datasets Avg</th></tr><tr><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><td>✗</td><td>✗</td><td>82.0</td><td>85.4</td><td>75.2</td><td>84.4</td><td>94.0</td><td>80.8</td><td>63.3</td><td>73.7</td><td>57.5</td></tr><tr><td>✓</td><td>✗</td><td>82.3</td><td>85.7</td><td>75.7</td><td>85.0</td><td>95.3</td><td>79.3</td><td>64.6</td><td>75.2</td><td>58.3</td></tr><tr><td>✗</td><td>✓</td><td>82.0</td><td>84.1</td><td>77.8</td><td>87.6</td><td>94.5</td><td>84.1</td><td>68.0</td><td>73.2</td><td>64.9</td></tr><tr><td>✓</td><td>✓</td><td>82.7</td><td>85.5</td><td>77.3</td><td>88.4</td><td>94.1</td><td>85.5</td><td>68.7</td><td>74.9</td><td>64.9</td></tr><tr><td colspan="2">Improv. over baseline</td><td>+0.7</td><td>+0.1</td><td>+2.1</td><td>+4.0</td><td>+0.1</td><td>+4.7</td><td>+5.4</td><td>+1.2</td><td>+7.4</td></tr></tbody></table>

### B.1 Detailed Experiment Results of the Effectiveness of ETF Alignment

In the main manuscript, we have demonstrated the effectiveness of the proposed ETF Alignment strategy on CUB, Herbarium 19, and ImageNet100 datasets. To further validate its generalization capability, we conduct additional experiments on Stanford Cars, FGVC Aircraft, and CIFAR100, and report the comprehensive results in Tables 6 and 7.

From Table 6, we observe that applying either supervised or unsupervised ETF Alignment leads to noticeable performance improvements across different datasets. Specifically, the combination of both supervised and unsupervised ETF Alignment consistently achieves the best results, especially on the novel categories, which demonstrates the effectiveness of our method in handling new categories under a generalized category discovery scenario.

Table 7 further reports the results on CIFAR100 and ImageNet100, as well as the average performance over all six datasets. Our method achieves significant improvements in both overall accuracy and novel category accuracy, confirming its strong generalization ability and robustness across various datasets and domains.

### B.2 Main Results with Different Backbones

To evaluate the generalization ability of NC-GCD across different pretrained representations, we conduct experiments using three representative backbones: DINOv1, DINOv2, and CLIP. The results on the CUB-200 dataset are summarized in Table 8.

Our method consistently achieves the best performance across all backbones. With DINOv1 and DINOv2 as backbones, NC-GCD slightly outperforms SelEx by up to +1.5% in Old accuracy and +1.0% in New accuracy, demonstrating stable and balanced gains. It should be noted that SelEx, GCD, and SimGCD did not originally adopt CLIP as their backbone—we thus reproduced these methods with CLIP as the backbone. When using the CLIP backbone, our method’s improvements become more significant, achieving +4.3% in All accuracy and +9.8% in Old accuracy. This result indicates that our consistent supervised–unsupervised alignment can effectively leverage semantic priors from large-scale vision–language pretraining.

Averaged over all backbones, NC-GCD reaches 80.4%, 80.5%, and 80.3% on All, Old, and New categories, respectively, outperforming all existing methods. These results verify that the proposed framework maintains strong backbone-agnostic consistency: by fixing Equiangular Tight Frame (ETF) prototypes, NC-GCD enforces a unified geometric structure regardless of the feature distribution of the encoder. Moreover, the large gain under CLIP shows that our geometric alignment is complementary to semantic alignment, resulting in a more structured and discriminative representation space for both known and novel categories. NC-GCD exhibits robust generalization across diverse backbones, demonstrating its universality and adaptability for generalized category discovery tasks.

Table 8: Comparison across different backbones (DINOv1, DINOv2, CLIP) and average on CUB200. The best results are in bold, the second best are underlined.

<table><thead><tr><th>Method</th><th colspan="3">DINOv1</th><th colspan="3">DINOv2</th><th colspan="3">CLIP</th><th colspan="3">AVG</th></tr><tr><th></th><th>ALL</th><th>Old</th><th>New</th><th>ALL</th><th>Old</th><th>New</th><th>ALL</th><th>Old</th><th>New</th><th>ALL</th><th>Old</th><th>New</th></tr></thead><tbody><tr><th>GCD (CVPR 22)</th><td>51.3</td><td>56.6</td><td>48.7</td><td>71.9</td><td>71.2</td><td>72.3</td><td>51.1</td><td>56.2</td><td>48.6</td><td>58.1</td><td>61.3</td><td>56.5</td></tr><tr><th>SimGCD (ICCV 23)</th><td>60.3</td><td>65.6</td><td>57.7</td><td>74.9</td><td>78.5</td><td>73.1</td><td>69.6</td><td>75.8</td><td>66.5</td><td>68.3</td><td>73.3</td><td>65.8</td></tr><tr><th>ProtoGCD (TPAMI-25)</th><td>63.5</td><td>68.5</td><td>60.5</td><td>75.7</td><td>81.5</td><td>72.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>RLCD (ICML 25)</th><td>70.0</td><td>79.1</td><td>65.4</td><td>78.7</td><td>79.5</td><td>78.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>SelEx (ECCV 24)</th><td>73.6</td><td>75.3</td><td>72.8</td><td>87.4</td><td>85.1</td><td>88.5</td><td>74.2</td><td>69.5</td><td>76.5</td><td>78.4</td><td>76.6</td><td>79.3</td></tr><tr><th>Ours (NC-GCD)</th><td>74.8</td><td>76.8</td><td>73.8</td><td>87.9</td><td>85.3</td><td>89.2</td><td>78.5</td><td>79.3</td><td>78.0</td><td>80.4</td><td>80.5</td><td>80.3</td></tr><tr><th>Improvement over SelEx</th><td>+1.2</td><td>+1.5</td><td>+1.0</td><td>+0.5</td><td>+0.2</td><td>+0.7</td><td>+4.3</td><td>+9.8</td><td>+1.5</td><td>+2.0</td><td>+3.9</td><td>+1.0</td></tr></tbody></table>

Table 9: Performance on CUB-200 with different ETF loss rates.

<table><thead><tr><th rowspan="3">ETF loss rate <math><semantics><mi>β</mi> <annotation>\beta</annotation></semantics></math></th><th colspan="6">CUB-200</th></tr><tr><th colspan="3">Best ACC</th><th colspan="3">Final ACC</th></tr><tr><th>ALL</th><th>Old</th><th>Novel</th><th>ALL</th><th>Old</th><th>Novel</th></tr></thead><tbody><tr><th>0.5</th><td>71.57</td><td>75.85</td><td>69.44</td><td>71.44</td><td>75.85</td><td>69.24</td></tr><tr><th>1</th><td>76.67</td><td>73.72</td><td>77.31</td><td>76.33</td><td>74.18</td><td>77.41</td></tr><tr><th>2</th><td>77.45</td><td>74.85</td><td>78.75</td><td>71.42</td><td>66.11</td><td>74.07</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/Ab-ETF-rate-cub.png.jpg)

Figure 5: Visualization of the influence of the ETF loss coefficient β \\beta on CUB-200.

### B.3 Analysis of ETF Loss Coefficient β\\beta

The ETF loss coefficient $\beta$ plays a crucial role in balancing the contributions of the ETF alignment loss and the representation learning loss in our model. As shown in Table 9 and Fig. 5, varying $\beta$ significantly influences both the "Best Accuracy" (Best ACC) and "Final Accuracy" (Final ACC) during training. "Best Accuracy" (Best ACC) refers to the highest accuracy achieved by the model at any point during the training process, typically before the model converges. It represents the model’s peak performance in classifying both known and novel categories during training. "Final Accuracy" (Final ACC) refers to the accuracy achieved at the last epoch, which indicates the model’s final performance after all epochs. This measure is typically considered the most reliable indicator of how well the model generalizes to unseen data.

In Fig. 5, we observe the accuracy curves across various values of $\beta$. When $\beta=1$ (Fig. 5b), the model demonstrates the best overall performance with the highest Best ACC of 76.67% and a solid Final ACC of 76.33%, indicating an optimal balance between ETF alignment and representation learning. The model shows excellent performance in both known and novel categories, with consistent improvements across all category types. The curves in Fig. 6b stabilize, reflecting the model’s ability to learn stable features across both known and new categories without overfitting.

When $\beta$ is reduced to 0.5 (Fig. 5a), we see a noticeable decrease in both Best ACC (71.57%) and Final ACC (71.44%), especially for novel categories, where the accuracy drops to 69.44%. The model’s performance on novel categories is notably weaker compared to the scenario when $\beta=1$, as reflected in the flatter curves for new categories in Fig. 6a. The lower $\beta$ value reduces the contribution of ETF alignment, impairing the model’s ability to effectively separate categories geometrically, which results in poorer category discovery and less effective feature alignment.

On the other hand, increasing $\beta$ to 2 (Fig. 5c) leads to a higher Best ACC for novel categories (78.75%) but a significant drop in Final ACC for old categories, which falls to 66.11%. While the model achieves better performance on novel categories during early training, it eventually overfits to the new categories, as seen in the accuracy curves for new categories. This causes a performance imbalance, with the model neglecting the stability of learned representations for known categories. As a result, the Final ACC for old categories is significantly lower, leading to an overall drop in model performance.

In conclusion, the results show that $\beta=1$ strikes the optimal balance, achieving the highest and most stable performance across both known and novel categories. The Best ACC indicates the model’s peak performance, while the Final ACC reflects its robustness in the final phase of learning. As our evaluation focuses on Final ACC, the results indicate that $\beta=1$ is the best choice for stable and high performance in generalized category discovery tasks, ensuring effective feature alignment and balanced learning across categories.

Table 10: Impact of estimation bias in category number $K$ on CUB-200 and CIFAR-100.

<table><thead><tr><th rowspan="2">Bias in <math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th colspan="3">CUB-200</th><th colspan="3">CIFAR-100</th></tr><tr><th>All (%)</th><th>Old (%)</th><th>Novel (%)</th><th>All (%)</th><th>Old (%)</th><th>Novel (%)</th></tr></thead><tbody><tr><th>-20%</th><td>70.3</td><td>72.1</td><td>69.4</td><td>79.8</td><td>83.7</td><td>72.5</td></tr><tr><th>-10%</th><td>72.5</td><td>73.5</td><td>72.0</td><td>81.2</td><td>84.3</td><td>75.2</td></tr><tr><th>0%</th><td>74.8</td><td>76.8</td><td>73.8</td><td>82.7</td><td>85.5</td><td>77.3</td></tr><tr><th>+10%</th><td>74.2</td><td>75.2</td><td>73.7</td><td>82.6</td><td>85.7</td><td>76.8</td></tr><tr><th>+20%</th><td>73.5</td><td>74.5</td><td>73.0</td><td>81.4</td><td>84.9</td><td>74.8</td></tr></tbody></table>

### B.4 Impact of Incorrect Estimation of Category Number KK

In the NC-GCD framework, the pre-assigned ETF prototypes rely on the estimated number of categories $K$ to construct the Simplex ETF structure. However, accurately determining the true number of novel categories is often difficult in real-world scenarios. To investigate the model’s robustness to estimation errors, we systematically evaluate NC-GCD under different levels of estimation bias, specifically with deviations of $\pm 10\%$ and $\pm 20\%$ from the ground-truth value, using the CUB-200 and CIFAR-100 datasets.

As shown in Table 10, underestimating $K$ leads to a significant decline in performance, particularly for novel categories. When $K$ is underestimated by 20%, the novel category accuracy on CUB-200 drops to 69.4%, and All category Accuracy (All ACC) decreases to 70.3%. In contrast, overestimating $K$ by 20% results in a much smaller drop, with All ACC still maintaining 73.5%. A similar trend is observed on CIFAR-100, where underestimating $K$ by 20% causes the novel category accuracy to fall to 72.5% and All ACC to 79.8%, while overestimating $K$ by 20% still preserves relatively high novel accuracy of 74.8% and All ACC of 81.4%.

These results confirm that underestimating $K$ has a more detrimental impact on novel category discovery due to insufficient prototype diversity, whereas the model is more tolerant to overestimation. This observation supports our conservative $K$ estimation strategy, where slightly overestimating the number of categories is preferred to mitigate the negative effects of prototype under-representation.

Overall, these findings demonstrate that accurate estimation of $K$ is crucial for optimal performance, particularly in novel category discovery. Underestimating $K$ significantly hampers the model’s ability to generalize to novel categories, while overestimating $K$ introduces redundancy without severe performance degradation. This conclusion further validates our conservative estimation strategy adopted in Section B.5, where a slightly larger $K$ is preferred to ensure sufficient prototype capacity. In future work, we will explore adaptive prototype adjustment techniques to dynamically refine $K$ during training and further enhance model robustness in open-world settings.

### B.5 Performance under Estimated Category Number KK

In real-world scenarios, the exact number of novel categories is often unknown. Instead of assuming access to the ground-truth $K$, our NC-GCD framework is capable of estimating the number of categories directly from data through clustering-based methods. This section evaluates how well the model performs under such automatic estimation compared to other state-of-the-art approaches.

We conduct an initial CMS-style [^6] representation learning phase for 30 epochs and estimate the number of categories every 5 epochs based on the current feature space. Following our earlier analysis on the impact of ETF prototype number deviations, we adopt a conservative estimation strategy by selecting the maximum estimated $K$ across all estimation points. This approach effectively reduces the negative impact of underestimating $K$, which, as previously shown, leads to significant performance degradation on novel categories, while the model remains more tolerant to overestimation.

As shown in Table 11, our NC-GCD achieves the highest All category Accuracy (All ACC) across all evaluated datasets, despite moderate estimation biases in $K$. On challenging fine-grained datasets such as CUB and Stanford Cars, our method significantly outperforms other baselines. Specifically, NC-GCD achieves 70.3% All ACC on CUB, surpassing CMS by +5.9%, and 54.0% on Stanford Cars, improving over CMS by +2.3%. On Herbarium19, even with a slight underestimation of $K$, our method still leads by a considerable margin, achieving 42.3% All ACC compared to the next best 37.4% by CMS.

For more generic datasets like CIFAR100 and ImageNet100, where the estimated $K$ values are closer to the ground truth, NC-GCD continues to deliver the best performance, reaching 80.5% and 85.7% All ACC, respectively. These results demonstrate that our method is not only effective in fine-grained scenarios but also robust across diverse open-world classification tasks.

Although the estimated $K$ may deviate from the true value, especially in complex fine-grained settings, NC-GCD maintains superior accuracy and shows strong resilience to estimation errors. In future work, we plan to explore advanced model selection and adaptive prototype adjustment techniques to further improve the estimation accuracy of $K$ and enhance the overall performance of generalized category discovery.

Table 11: Comparison of category number estimation ($K$) and all category accuracy (All ACC) across different methods.

<table><thead><tr><th rowspan="2">Method</th><th colspan="2">CIFAR100</th><th colspan="2">ImageNet100</th><th colspan="2">CUB</th><th colspan="2">Stanford Cars</th><th colspan="2">FGVC Aircraft</th><th colspan="2">Herbarium19</th></tr><tr><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th><th><math><semantics><mi>K</mi> <annotation>K</annotation></semantics></math></th><th>All ACC</th></tr></thead><tbody><tr><th>Ground Truth</th><td>100</td><td>-</td><td>100</td><td>-</td><td>200</td><td>-</td><td>196</td><td>-</td><td>100</td><td>-</td><td>683</td><td>-</td></tr><tr><th>GCD <sup><a href="#fn:38">38</a></sup></th><td>100</td><td>70.8</td><td>109</td><td>77.9</td><td>230</td><td>51.1</td><td>230</td><td>39.1</td><td>-</td><td>-</td><td>520</td><td>37.2</td></tr><tr><th>GPC <sup><a href="#fn:47">47</a></sup></th><td>100</td><td>75.4</td><td>103</td><td>75.3</td><td>201</td><td>52.0</td><td>201</td><td>38.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>PIM <sup><a href="#fn:5">5</a></sup></th><td>95</td><td>75.6</td><td>102</td><td>83.0</td><td>227</td><td>62.0</td><td>169</td><td>55.1</td><td>-</td><td>-</td><td>563</td><td>42.0</td></tr><tr><th>CMS <sup><a href="#fn:6">6</a></sup></th><td>97</td><td>79.6</td><td>116</td><td>81.3</td><td>170</td><td>64.4</td><td>156</td><td>51.7</td><td>98</td><td>55.2</td><td>666</td><td>37.4</td></tr><tr><th>Ours (NC-GCD)</th><td>96</td><td>80.5</td><td>109</td><td>85.7</td><td>182</td><td>70.3</td><td>169</td><td>54.0</td><td>105</td><td>55.4</td><td>568</td><td>42.3</td></tr></tbody></table>

### B.6 Analysis of Clustering Frequency TT

We further analyze the sensitivity of our NC-GCD framework to the clustering frequency, controlled by the parameter $T$, which determines how often clustering and prototype updates occur. As shown in Table B.6, setting $T=5$ achieves the best overall performance, with the highest All accuracy of 74.80% and the highest Novel accuracy of 73.8%. This indicates that performing clustering every five epochs strikes an effective balance between feature stability and timely prototype updates, leading to improved novel category discovery.

Table 12: Impact of $T$ on CUB-200.

<table><thead><tr><th rowspan="2">Clustering Period <math><semantics><mi>T</mi> <annotation>T</annotation></semantics></math></th><th colspan="3">Accuracy</th></tr><tr><th>All</th><th>Old</th><th>Novel</th></tr></thead><tbody><tr><th>1</th><td>73.7</td><td>75.2</td><td>72.9</td></tr><tr><th>5</th><td>74.8</td><td>76.8</td><td>73.8</td></tr><tr><th>10</th><td>74.2</td><td>77.5</td><td>72.6</td></tr><tr><th>20</th><td>72.9</td><td>74.5</td><td>71.9</td></tr></tbody></table>

Interestingly, when $T=10$, the accuracy for Old categories peaks at 77.5%, suggesting that less frequent clustering better preserves the learned representations of known categories. However, this comes at the cost of reduced Novel accuracy, highlighting a trade-off between stability for known categories and adaptability for novel categories.

When the clustering frequency is too high ($T=1$), the model suffers from unstable prototype updates, negatively affecting both Old and Novel accuracies. Conversely, setting $T=20$ significantly reduces the model’s ability to adapt to novel categories due to infrequent updates, leading to the lowest overall performance.

## Appendix C Parameter and SCM Efficiency Analysis

We analyze the computational efficiency of our method from two perspectives: (1) model parameter efficiency and (2) the computational cost introduced by the Semantic Consistency Matcher (SCM).

### C.1 Parameter Analysis

We analyze the computational efficiency of our method by comparing the total number of parameters, the trainable parameters, and the average accuracy (Avg ACC) achieved by our approach, SimGCD, and CMS.

As shown in Table 13, our method achieves competitive performance with an average accuracy of 68.7%, significantly outperforming SimGCD and CMS, which achieve accuracies of 62.57% and 64.08%, respectively. Our method has a similar total parameter count (92.89M) compared to SimGCD (92.89M), but unlike SimGCD, where the classifier is trained alongside the rest of the model, we pre-assign ETF prototypes and do not require training a classifier. This results in the same number of trainable parameters (20.34M) as CMS, which does not use a classifier either.

This comparison demonstrates that our approach does not introduce additional trainable parameters compared to methods like CMS, which do not rely on classifiers but still achieve significantly higher accuracy. The efficiency of our model is evident in the fact that it improves performance without requiring extra learnable parameters for the classifier, as seen in the results. In conclusion, our method achieves high performance with highly efficient parameterization, making it an attractive solution for generalized category discovery tasks, as it strikes a balance between computational cost and accuracy.

Table 13: Comparison of parameters and accuracy.

| Method | Total parameter | Trainable parameter | Avg Acc |
| --- | --- | --- | --- |
| SimGCD (ICCV 2023) | 92.89M | 20.74M | 62.57 |
| CMS (CVPR 2024) | 92.49M | 20.34M | 64.08 |
| NC-GCD (Ours) | 92.89M | 20.34M | 68.70 |

Table 14: Computation time comparison (per iteration) on CUB-200 (3090 GPU).

| Method | Feature Extraction (s) | Clustering (s) | SCM Matching (ms) | Class Center Computation (s) |
| --- | --- | --- | --- | --- |
| CMS (CVPR 2024) | 17.2846 | 4.2050 | – | – |
| NC-GCD (Ours) | 17.2846 | 4.2050 | 0.138 | 0.2828 |

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/vis-cub-Allclass.jpg)

Figure 6: Visualization of all categories of CUB-200.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/vis-cub-10class-a.jpg)

Figure 7: Visualization of 10 random categories of CUB-200.

### C.2 SCM Efficiency

The Semantic Consistency Matcher (SCM) effectively improves the stability of clustering assignments while introducing minimal computational overhead. As reported in Table 14, SCM Matching takes only 0.138 ms per execution, and the additional time for category center computation is just 0.2828 s. Moreover, SCM is executed only once every 5 epochs, which further reduces its impact on the overall training time.

All experiments are conducted on a single NVIDIA 3090 GPU, and even under this setting, the additional overhead introduced by SCM remains negligible.

Despite this minimal overhead, SCM plays a critical role in improving label consistency and enhancing model performance. Notably, our NC-GCD framework achieves significant performance gains with almost no additional computational cost, demonstrating an excellent trade-off between accuracy and efficiency. These results highlight the practical scalability and real-world applicability of NC-GCD for generalized category discovery tasks.

## Appendix D More Visualization Results

### D.1 Comparison with State of the Art Methods

To further demonstrate the effectiveness of our NC-GCD framework, we present additional visualizations of the feature space learned by our method. The t-SNE visualizations of all categories and 10 random categories from the CUB-200 dataset are shown in Figures 6 and 7, respectively.

In Fig. 6, we visualize the embeddings of all categories in the CUB-200 dataset. Fig. 6 (a) shows the t-SNE visualization of the embeddings extracted using DINOv1, where significant overlap between categories is observed. This overlap indicates that the model struggles to distinguish between many of the categories. Fig. 6 (b) presents the embeddings extracted by CMS, which also show notable clusters but still suffer from some confusion and overlap. In contrast, Fig. 6 (c) shows the embeddings from our NC-GCD method, where the categories are more clearly separated. This demonstrates our model’s ability to learn more distinct and structured feature representations for all categories, confirming the efficacy of our pre-assigned ETF prototypes and alignment strategy in improving class separability.

Fig. 7 provides a zoomed-in view with t-SNE visualizations of 10 randomly selected categories from the CUB-200 dataset. Fig. 7 (a) again shows the embeddings from DINOv1, where the categories are still widely scattered and not well separated. Fig. 7 (b) shows CMS embeddings, where the clustering has improved but still lacks the fine-grained separation seen in (c). In Fig. 7 (c), our method successfully groups the categories with minimal overlap, and the embeddings show compact, well-separated clusters. This highlights our method’s ability to preserve class separability even with a limited subset of categories, further demonstrating the robustness of NC-GCD in handling both known and novel categories.

These visualizations underscore the power of our NC-GCD framework in ensuring clear category separation and reducing category confusion. The effectiveness of the ETF alignment strategy is evident in both the overall category separability and the detailed random category embeddings, where our method significantly outperforms both DINOv1 and CMS in organizing the feature space.

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/vis-cub-epoch-all-class.jpg)

Figure 8: Visualization of our training process on CUB-200 (All categories).

![Refer to caption](https://arxiv.org/html/2507.04725v3/img/vis-cub-epoch-3-class.jpg)

Figure 9: Visualization of our training process on CUB-200 (3 random categories).

### D.2 Training Process

To better understand how our NC-GCD framework organizes feature representations over time, we provide t-SNE visualizations of the training process on the CUB-200 dataset. These visualizations demonstrate how our method progressively refines feature distributions, ultimately achieving a well-structured and highly separable feature space.

Fig. 8 illustrates the evolution of feature representations for all categories across different training epochs. At epoch 0, the features are highly entangled, with significant category confusion due to the lack of geometric constraints. As training progresses, the feature clusters gradually become more distinct, aligning with their respective ETF prototypes. By epoch 40, the initial separation between categories emerges, and by epoch 100, the majority of categories are well-structured. By epoch 200, the features exhibit a near-complete collapse to a simplex ETF structure, aligning closely with the Neural Collapse (NC) theory [^30], where features within each class concentrate around their mean while inter-class distances remain maximized.

Fig. 9 further highlights this phenomenon by showing three randomly selected categories. Initially, these categories exhibit substantial overlap, but as training progresses, their feature distributions become increasingly compact while maintaining distinct separation. Eventually, each category collapses onto a single point, which is consistent with the expected NC behavior. This confirms that our method successfully enforces a geometric structure that maximizes category separability while maintaining within-category compactness.

These visualizations validate the effectiveness of our approach. By leveraging Neural Collapse principles and ETF alignment, NC-GCD ensures a consistent and optimal feature arrangement throughout training. The final collapse of features to a well-defined simplex ETF structure highlights the theoretical soundness of our method and its strong alignment with NC theory, making it highly effective for Generalized Category Discovery.

## Appendix E Limitations and Future Work

While our NC-GCD framework demonstrates strong performance across various benchmarks, there remain opportunities for further improvement.

First, the scalability of the fixed ETF prototype structure has not been thoroughly explored. Although the current design works well on the evaluated datasets, its effectiveness in large-scale or highly complex scenarios remains to be validated. Future work will investigate how to extend the ETF framework to better handle diverse and large-scale category distributions.

Second, the estimation of the category number $K$ relies on clustering-based heuristics, which generally perform well but may be sensitive to feature quality in some challenging scenarios. Incorporating more robust estimation techniques could help improve performance stability, especially in cases with highly ambiguous data distributions.

## Appendix F Broader Impacts

This work explores a novel framework for Generalized Category Discovery (GCD), enhancing the ability of AI systems to autonomously recognize and differentiate new concepts in open-world environments. By reducing the dependence on large-scale labeled datasets, our method lowers the barriers to applying AI technologies in domains where data annotation is expensive or difficult to obtain [^38] [^13].

The proposed framework encourages more efficient use of unlabeled data, offering practical solutions for knowledge discovery in fields such as environmental monitoring, healthcare diagnostics, and industrial quality control. Furthermore, it has the potential to accelerate the development of adaptive intelligent systems capable of continuously learning from new, unseen data without extensive human supervision. We hope this research will contribute to broader advancements in learning with limited supervision and inspire further innovations in building more flexible and scalable AI systems [^11].

[^1]: A. Bendale and T. Boult (2015) Towards open world recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1893–1902. Cited by: §1.

[^2]: K. Cao, M. Brbic, and J. Leskovec (2021) Open-world semi-supervised learning. arXiv preprint arXiv:2102.03526. Cited by: §2.1.

[^3]: M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin (2021) Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650–9660. Cited by: §5.1.

[^4]: F. J. Cendra, B. Zhao, and K. Han (2024) Promptccd: learning gaussian mixture prompt pool for continual category discovery. In European conference on computer vision, pp. 188–205. Cited by: §2.1.

[^5]: F. Chiaroni, J. Dolz, Z. I. Masud, A. Mitiche, and I. Ben Ayed (2023) Parametric information maximization for generalized category discovery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1729–1739. Cited by: Table 11, §1, §2.1, §5.2.

[^6]: S. Choi, D. Kang, and M. Cho (2024) Contrastive mean-shift learning for generalized category discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23094–23104. Cited by: §A.1, §B.5, Table 11, §1, §2.1, §4.4, §5.1, §5.2.

[^7]: J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei (2009) ImageNet: a large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255. Cited by: §A.2, Table 5, §5.1.

[^8]: A. Dosovitskiy (2020) An image is worth 16x16 words: transformers for image recognition at scale. arXiv preprint arXiv:2010.11929. Cited by: §5.1.

[^9]: C. Fang, H. He, Q. Long, and W. J. Su (2021) Exploring deep neural networks via layer-peeled model: minority collapse in imbalanced training. Proceedings of the National Academy of Sciences 118 (43). Cited by: §2.2.

[^10]: E. Fini, E. Sangineto, S. Lathuilière, Z. Zhong, M. Nabi, and E. Ricci (2021) A unified objective for novel class discovery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9284–9292. Cited by: §2.1.

[^11]: Y. Gong and G. Wang (2024) Preface: brain-inspired ai research. Science China. Technological Sciences 67 (8), pp. 2281–2281. Cited by: Appendix F.

[^12]: F. Graf, C. Hofer, M. Niethammer, and R. Kwitt (2021) Dissecting supervised contrastive learning. In Proceedings of the International Conference on Machine Learning, pp. 3821–3830. Cited by: §2.2.

[^13]: J. Han, C. Ding, Y. He, S. Dong, Q. Wang, X. Gao, and Y. Gong (2025) Learn by reasoning: analogical weight generation for few-shot class-incremental learning. arXiv preprint arXiv:2503.21258. Cited by: Appendix F.

[^14]: K. Han, S. Rebuffi, S. Ehrhardt, A. Vedaldi, and A. Zisserman (2020) Automatically discovering and learning new visual categories with ranking statistics. arXiv preprint arXiv:2002.05714. Cited by: §2.1.

[^15]: X. Han, V. Papyan, and D. L. Donoho (2021) Neural collapse under mse loss: proximity to and dynamics on the central path. arXiv preprint arXiv:2106.02073. Cited by: §2.2.

[^16]: D. Hendrycks and K. Gimpel (2016) Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415. Cited by: §A.1, §5.1.

[^17]: C. Huang, L. Xie, Y. Yang, W. Wang, B. Lin, and D. Cai (2023) Neural collapse inspired federated learning with non-iid data. arXiv preprint arXiv:2303.16066. Cited by: §2.2.

[^18]: W. Ji, Y. Lu, Y. Zhang, Z. Deng, and W. J. Su (2021) An unconstrained layer-peeled perspective on neural collapse. arXiv preprint arXiv:2110.02796. Cited by: §2.2.

[^19]: J. Jiang, J. Zhou, P. Wang, Q. Qu, D. Mixon, C. You, and Z. Zhu (2023) Generalized neural collapse for a large number of classes. arXiv preprint arXiv:2310.05351. Cited by: §2.2.

[^20]: K. Joseph, S. Khan, F. S. Khan, and V. N. Balasubramanian (2021) Towards open world object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5830–5840. Cited by: §1.

[^21]: H. Kim, S. Suh, D. Kim, D. Jeong, H. Cho, and J. Kim (2023) Proxy anchor-based unsupervised learning for continuous generalized category discovery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 16688–16697. Cited by: §2.1.

[^22]: J. Krause, M. Stark, J. Deng, and L. Fei-Fei (2013) 3D object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision workshops, pp. 554–561. Cited by: §A.2, Table 5, §5.1.

[^23]: A. Krizhevsky and G. Hinton (2009) Learning multiple layers of features from tiny images. Technical report University of Toronto. Cited by: §A.2, Table 5, §5.1.

[^24]: Y. Li, Y. Wang, W. Wang, D. Lin, B. Li, and K. Yap (2025) Open world object detection: a survey. IEEE Transactions on Circuits and Systems for Video Technology 35 (2), pp. 988–1008. Cited by: §1.

[^25]: D. Liu, Z. Tan, L. Zhao, Z. Zhang, X. Fang, and W. Huang (2025) Generalized category discovery via reciprocal learning and class-wise distribution regularization. In Proceedings of the International Conference on Machine Learning, Cited by: §2.1.

[^26]: S. Ma, F. Zhu, X. Zhang, and C. Liu (2025) ProtoGCD: unified and unbiased prototype learning for generalized category discovery. IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (7), pp. 6022–6038. Cited by: §2.1.

[^27]: S. Ma, F. Zhu, Z. Zhong, X. Zhang, and C. Liu (2024) Active generalized category discovery. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16890–16900. Cited by: §1.

[^28]: S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi (2013) Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151. Cited by: §A.2, Table 5, §5.1.

[^29]: D. G. Mixon, H. Parshall, and J. Pi (2022) Neural collapse with unconstrained features. Sampling Theory, Signal Processing, and Data Analysis 20 (2), pp. 11. Cited by: §2.2.

[^30]: V. Papyan, X. Han, and D. L. Donoho (2020) Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences 117 (40), pp. 24652–24663. Cited by: §D.2, §1, §2.2, §3.2.

[^31]: K. Park, H. Lee, K. Song, and G. Park (2024) Online continuous generalized category discovery. In European Conference on Computer Vision, pp. 53–69. Cited by: §2.1.

[^32]: T. Poggio and Q. Liao (2020) Explicit regularization and implicit bias in deep network classifiers trained with the square loss. arXiv preprint arXiv:2101.00072. Cited by: §2.2.

[^33]: N. Pu, Z. Zhong, and N. Sebe (2023) Dynamic conceptual contrastive learning for generalized category discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7579–7588. Cited by: §1, §2.1, §5.2.

[^34]: S. Rastegar, M. Salehi, Y. M. Asano, H. Doughty, and C. G. Snoek (2024) SelEx: self-expertise in fine-grained generalized category discovery. In European Conference on Computer Vision, pp. 440–458. Cited by: §1, §5.2.

[^35]: G. Rypeść, D. Marczak, S. Cygert, T. Trzciński, and B. Twardowski (2024) Category adaptation meets projected distillation in generalized continual category discovery. In European Conference on Computer Vision, pp. 320–337. Cited by: §2.1.

[^36]: K. C. Tan, Y. Liu, B. Ambrose, M. Tulig, and S. Belongie (2019) The herbarium challenge 2019 dataset. arXiv preprint arXiv:1906.05372. Cited by: §A.2, Table 5, §5.1.

[^37]: S. Vaze, K. Han, A. Vedaldi, and A. Zisserman (2021) Open-set recognition: a good closed-set classifier is all you need?. In International Conference on Learning Representations, Cited by: §A.2, §A.2, §5.1.

[^38]: S. Vaze, K. Han, A. Vedaldi, and A. Zisserman (2022) Generalized category discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8997–9006. Cited by: §A.1, §A.1, Table 11, Appendix F, §1, §2.1, §5.1, §5.2.

[^39]: C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie (2011) The caltech-ucsd birds-200-2011 dataset. Technical report California Institute of Technology. Cited by: §A.2, Table 5, §5.1.

[^40]: H. Wang, S. Vaze, and K. Han (2024) SPTNet: an efficient alternative framework for generalized category discovery with spatial prompt tuning. In Proceedings of the International Conference on Learning Representations, Cited by: §5.2.

[^41]: Q. Wang, Y. He, S. Dong, X. Song, J. Han, H. Luo, and Y. Gong (2025) DualCP: rehearsal-free domain-incremental learning via dual-level concept prototype. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, pp. 21198–21206. Cited by: §2.2.

[^42]: X. Wen, B. Zhao, and X. Qi (2023) Parametric classification for generalized category discovery: a baseline study. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 16590–16600. Cited by: §2.1, §5.2.

[^43]: R. Xiao, L. Feng, K. Tang, J. Zhao, Y. Li, G. Chen, and H. Wang (2024) Targeted representation alignment for open-world semi-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 23072–23082. Cited by: §2.2, §5.2.

[^44]: Y. Yang, S. Chen, X. Li, L. Xie, Z. Lin, and D. Tao (2022) Inducing neural collapse in imbalanced learning: do we really need a learnable classifier at the end of deep neural network?. Advances in neural information processing systems 35, pp. 37991–38002. Cited by: §2.2, §4.2.

[^45]: Y. Yang, H. Yuan, X. Li, Z. Lin, P. H. S. Torr, and D. Tao (2023) Neural collapse inspired feature-classifier alignment for few-shot class-incremental learning. In International Conference on Learning Representations, Cited by: §2.2.

[^46]: S. Zhang, S. Khan, Z. Shen, M. Naseer, G. Chen, and F. S. Khan (2023) Promptcal: contrastive affinity learning via auxiliary prompts for generalized novel category discovery. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3479–3488. Cited by: §1, §2.1, §5.2.

[^47]: B. Zhao, X. Wen, and K. Han (2023) Learning semi-supervised gaussian mixture models for generalized category discovery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 16623–16633. Cited by: Table 11, §2.1, §5.2.

[^48]: Z. Zhong, J. Cui, Y. Yang, X. Wu, X. Qi, X. Zhang, and J. Jia (2023) Understanding imbalanced semantic segmentation through neural collapse. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: §2.2.

[^49]: D. Zhou, F. Wang, H. Ye, L. Ma, S. Pu, and D. Zhan (2022) Forward compatible few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9046–9056. Cited by: §2.2.