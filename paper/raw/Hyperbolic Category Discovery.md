---
title: "Hyperbolic Category Discovery"
source: "https://arxiv.org/html/2504.06120v1"
author:
published:
created: 2026-05-04
description:
tags:
  - "clippings"
---
Yuanpei Liu <sup>*</sup>   Zhenqi He <sup>*</sup>   Kai Han <sup>†</sup>  
Visual AI Lab, The University of Hong Kong  
{ypliu0,zhenqi\_he}@connect.hku.hk   kaihanx@hku.hk

###### Abstract

Generalized Category Discovery (GCD) is an intriguing open-world problem that has garnered increasing attention. Given a dataset that includes both labelled and unlabelled images, GCD aims to categorize all images in the unlabelled subset, regardless of whether they belong to known or unknown classes. In GCD, the common practice typically involves applying a spherical projection operator at the end of the self-supervised pretrained backbone, operating within Euclidean or spherical space. However, both of these spaces have been shown to be suboptimal for encoding samples that possesses hierarchical structures. In contrast, hyperbolic space exhibits exponential volume growth relative to radius, making it inherently strong at capturing the hierarchical structure of samples from both seen and unseen categories. Therefore, we propose to tackle the category discovery challenge in the hyperbolic space. We introduce HypCD, a simple Hyperbolic framework for learning hierarchy-aware representations and classifiers for generalized Category Discovery. HypCD first transforms the Euclidean embedding space of the backbone network into hyperbolic space, facilitating subsequent representation and classification learning by considering both hyperbolic distance and the angle between samples. This approach is particularly helpful for knowledge transfer from known to unknown categories in GCD. We thoroughly evaluate HypCD on public GCD benchmarks, by applying it to various baseline and state-of-the-art methods, consistently achieving significant improvements. Project page: [https://visual-ai.github.io/hypcd/](https://visual-ai.github.io/hypcd/)

<sup>1</sup> <sup>2</sup>

## 1 Introduction

Recently, category discovery – initially explored as novel category discovery (NCD) [^24] and subsequently extended to generalized category discovery (GCD) [^53] – has emerged as an intriguing open-world problem, gaining increasing attention. GCD addresses the challenges posed by partially labelled datasets, where the unlabelled subset may contain instances from both seen and unseen classes. The goal is to leverage knowledge from labelled data to effectively categorize the unlabelled data. Based on the way to predict category index, existing GCD methods can be broadly classified into two types: non-parametric methods [^53] [^27] [^45] [^46] and parametric methods [^61] [^57] [^37]. Non-parametric methods predict category index based on feature clustering while parametric methods utilize a parametric classifier.

![Refer to caption](https://arxiv.org/html/2504.06120v1/x1.png)

Figure 1: (a) Spherical-based vs. Hyperbolic-based methods, where hyperbolic space better accommodates variations in scale and improves connections between samples. (b) Average ACC comparison of our method and previous SOTA across ‘All’, ‘Old’, and ‘New’ categories on the SSB 54 benchmark using DINO 7.

As shown in the GCD literature [^53], object parts are effective for knowledge transfer from seen to unseen categories, which is crucial for novel category discovery. Methods have been developed to explicitly learn better local features by learning pixel-level prompts around local image regions [^57] or utilizing part-level features [^65]. However, these methods consider object parts as rigid image patches of the same size, without considering the hierarchical nature of the object parts and the scale discrepancy of the same parts in different images, thus unavoidably restricting the performance for GCD (see Fig. 1(a)), in which the objects often have distinct poses, scales and appearance. To address this problem, one possible solution is to learn image embeddings possessing hierarchical constraints or following tree-like structures. This has been proven to be effective in many tasks. For example, in image retrieval and clustering, the hierarchy constraint may arise from whole-fragment relation [^29] [^31]. Intuitively, in category discovery, which can be regarded as a transfer clustering task [^24], we hypothesize that an embedding space that captures the hierarchical relations of object parts can also facilitate the discovery of new categories. Indeed, the hierarchical relations have been studied in GCD, such as [^27] [^45] [^59] [^46] [^41]. However, they study from a substantially different perspective: inter-category hierarchy (see Fig. 2(a)). This only considers the hierarchy of different semantic classes from coarse to fine levels.

![Refer to caption](https://arxiv.org/html/2504.06120v1/x2.png)

Figure 2: Hierarchical relations in GCD. (a) Inter-category relationships within the Stanford-Cars dataset 33. (b) Intra-category relationships within CUB 56 dataset.

Additionally, these methods require the relationships and number of levels in the hierarchy to be predefined, resulting in a lack of flexibility and scalability. Moreover, these methods are unable to capture more complex hierarchies, such as the compositional parts of an object (see Fig. 2(b)). This is particularly the case because existing methods [^53] [^61] [^57] [^46], no matter whether they consider the hierarchy or not, learn the image embeddings in a spherical space. This follows the common practice of applying a spherical projection operator at the end of the self-supervised feature backbone [^7] [^40]. Consequently, all subsequent operations, including distance calculations, are performed under either Euclidean or spherical geometry, resulting in limited awareness of hierarchical object parts.

In this work, we study the overlooked perspective in category discovery: *instead of learning in the Euclidean or spherical space, we advocate a space that captures the hierarchical structure of each data point.* In spherical space, both the radius and volume are constant, whereas in Euclidean space, the volume grows polynomially with respect to the radius. Both of these spaces have been shown to be suboptimal for encoding samples that possess hierarchical structures [^31] [^15] [^10]. In contrast, hyperbolic space possesses a distinctive property where its volume grows exponentially relative to the radius. This characteristic makes hyperbolic space particularly suitable for embedding tree-like data, enhancing its representational power. Learning representations in hyperbolic space has proven to be effective in various computer vision tasks, including object recognition [^15], object detection [^32], semantic segmentation [^60], and anomaly detection [^35]. Inspired by these successes, we aim to realize the idea of learning hierarchy-aware representations to facilitate knowledge transfer in category discovery, thereby unleashing the potential of hyperbolic representations.

To achieve this goal, we propose a simple yet effective framework, HypCD, to properly learn the hierarchy-aware representation and classifier for category discovery through the lens of hyperbolic geometry. In this framework, we adapt our framework to popular parametric [^61] and non-parametric [^53] GCD baselines as well as the state-of-the-art (SOTA) method SelEx [^46], obtaining substantial improvements for them, establishing the new SOTA (see Fig. 1(b)). Firstly, starting from the self-supervised backbone pretrained in Euclidean space, we propose to map the Euclidean representation to a constrained Poincaré ball through feature clipping and exponential mapping. Secondly, we implement the hyperbolic representation learning and build a hyperbolic classifier on the Poincaré ball, considering both angle and distance between samples in hyperbolic space. Thirdly, to assign labels to unlabelled data after training, for non-parametric methods, we apply semi-supervised k-means following the common practice; for parametric methods, we employ a hyperbolic classifier to make predictions. Despite its simplicity, our framework achieves significant performance improvements with two different pretrained weights (DINO [^7] and DINOv2 [^40]) on the public GCD datasets, including the coarse-grained classification datasets CIFAR-10 [^34], CIFAR-100 [^34], and ImageNet-100 [^13], as well as the fine-grained SSB [^52] benchmark.

In summary, we make the following contributions in this paper: (i) We identify the existing GCD methods’ common shortcoming in encoding the hierarchical structure and propose to incorporate the hyperbolic geometry into the embedding space to address this limitation; (ii) We propose a simple yet effective framework, called HypCD, for incorporating the hyperbolic geometry in representation learning and classification for category discovery; and (iii) Through extensive experiments on public GCD benchmarks by applying HypCD to baseline and SOTA methods, our method consistently demonstrates effectiveness and superiority.

## 2 Related Work

#### Category Discovery.

Novel category discovery (NCD) is initially introduced in [^24] to establish a realistic framework for transferring knowledge from seen categories to cluster unseen categories, by considering it as a transfer clustering problem. Many subsequent methods have been proposed to advance the field [^25] [^26] [^28] [^65] [^67] [^17]. Generalized category discovery [^53] (GCD) relaxes the assumption in NCD by considering unlabelled data containing samples from both known and unknown classes [^53]. Further investigations [^6] [^27] [^43] [^30] [^8] [^58] [^37] have explored a variety of strategies to address the challenges posed by GCD. One notable approach, SimGCD [^61], proposes to learn a parametric classifier enhanced by mean entropy regularization, thereby improving performance. In another vein, GPC [^66] employs Gaussian mixture models to jointly learn robust representations while simultaneously estimating the number of unknown categories. SPTNet [^57] presents a spatial prompt tuning method that enables models to concentrate more effectively on specific object parts, thus enhancing knowledge transfer in GCD. Most recently, SelEx [^46] has been proposed, leveraging hierarchical semi-supervised $k$ -means to achieve SOTA results on fine-grained datasets. Additionally, various efforts are focused on addressing category discovery from multiple perspectives. For instance, [^28] emphasizes multi-modal category discovery; [^64] and [^8] explore a continual setting; [^44] studies category discovery in a federated setting; and [^58] examines GCD in the presence of domain shifts.

Hyperbolic Geometry. Hyperbolic space, defined as a non-Euclidean manifold with exponential volume growth in relation to its radius, is inherently aligned with the embedding of tree-like and hierarchical data structures in visual recognition tasks. Significant advancements in this area include [^31], which presents a hyperbolic image embedding technique by projecting model outputs into hyperbolic space, and [^15], which integrates hyperbolic geometry into various vision transformer architectures, showcasing performance that surpasses their Euclidean counterparts. Hyperbolic methods have also been developed on diverse tasks such as image classification [^31] [^15] [^23], action recognition [^18], few-shot learning [^20] and object segmentation [^21] [^62]. Moreover, recent developments have introduced hyperbolic geometry for neural networks including fully connected layers [^47], convolutional neural networks [^3], graph neural networks [^36] [^9], and attention network [^22], thereby facilitating a deeper integration of hyperbolic geometry into deep learning regime.

## 3 Method

In this section, we first introduce the task in Sec. 3.1, then move to a review of baselines in Sec. 3.2. Afterwards, the geometry mapping and training details of our framework are described in Sec. 3.3 and Sec. 3.4. Lastly, the label assignment details are outlined in Sec. 3.5.

### 3.1 Problem Statement

GCD aims to learn a model capable of accurately classifying unlabelled samples from known categories while simultaneously clustering those from unknown categories. Consider an unlabelled dataset denoted as $\mathbf{D}_{u}=\{(\mathbf{x}^{u}_{i},{y}^{u}_{i})\}\in\mathbf{X}\times\mathbf{%
Y}_{u}$ and a labelled dataset represented as $\mathbf{D}_{l}=\{(\mathbf{x}^{l}_{i},{y}^{l}_{i})\}\in\mathbf{X}\times\mathbf{%
Y}_{l}$, where $\mathbf{Y}_{u}$ and $\mathbf{Y}_{l}$ denote the respective label sets. The unlabelled dataset comprises samples from both known and unknown categories, i.e., specifically $\mathbf{Y}_{l}\subset\mathbf{Y}_{u}$. Let the number of labelled categories be denoted by $M=|\mathbf{Y}_{l}|$. We assume that the total number of categories, $K=|\mathbf{Y}_{l}\cup\mathbf{Y}_{u}|$, is known, as posited in prior works [^26] [^61] [^55]. In scenarios where this information is unavailable, alternative methods such as those proposed in [^24] [^53] can be employed to yield a reliable estimation.

### 3.2 Review of Baselines

Non-parametric Baseline. [^53] formalizes the GCD task and proposes a non-parametric baseline. The approach involves finetuning the pre-trained DINO [^7] model [^14] to enhance the learned representation. The loss function comprises a supervised contrastive loss, which operates on the labelled samples, and a self-supervised contrastive loss, which operates on all the samples. Specifically, given two randomly augmented views $\mathbf{x}_{i}$ and $\mathbf{x}_{i}^{\prime}$ for the same image in a mini-batch ${B}$, the self-supervised contrastive loss is:

$$
\textstyle\mathcal{L}_{rep}^{u}=\frac{1}{|{B}|}\sum_{i\in{B}}-\text{log}\frac{%
\exp(\mathbf{z}_{i}\cdot\mathbf{z}_{i}^{\prime}/\tau_{r})}{\sum\nolimits_{j}^{%
j\neq i}\exp(\mathbf{z}_{i}\cdot\mathbf{z}_{j}^{\prime}/\tau_{r})},
$$

where the feature $\mathbf{z}_{i}=\rho_{r}(\phi(\mathbf{x}_{i}))$ is a $\ell_{2}$ -normalized vector and $\mathbf{z}_{i}^{\prime}$ represents feature from another view $\mathbf{x}_{i}^{\prime}$. Here, $\phi$ refers to the backbone network, $\rho_{r}$ denotes the projection head, and $\tau_{r}$ stands as the temperature parameter used for scaling the features. The supervised contrastive loss for labelled samples is:

$$
\textstyle\mathcal{L}_{rep}^{s}=\frac{1}{|{B}_{l}|}\sum_{i\in{B}_{l}}\frac{1}{%
|{N}_{i}|}\sum\limits_{q\in{N}_{i}}-\text{log}\frac{\exp(\mathbf{z}_{i}\cdot%
\mathbf{z}_{q}/\tau_{r})}{\sum\nolimits_{j}^{j\neq i}\exp(\mathbf{z}_{i}\cdot%
\mathbf{z}_{j}/\tau_{r})},
$$

where ${N}_{i}$ is the index set for all other images in the labelled mini-batch ${B}_{l}\subset{B}$ having the same label as $\mathbf{x}_{i}$. The overall representation learning loss is then: $\mathcal{L}_{rep}=(1-\lambda_{b})\mathcal{L}_{rep}^{u}+\lambda_{b}\mathcal{L}_%
{rep}^{s}$, where $\lambda_{b}$ is a balance factor.

Parametric Baseline. [^61] introduces a robust parametric GCD baseline, which has been widely adopted in the field ever since [^55] [^57]. This method employs a parametric classifier implemented in a self-distillation framework [^7]. The classifier is randomly initialized with $K$ normalized category prototypes $\mathbf{C}=\{\mathbf{c}_{1},...,\mathbf{c}_{K}\}$. For a randomly augmented view $\mathbf{x}_{i}$ and its corresponding normalized hidden feature vector $\mathbf{h}_{i}=\phi(\mathbf{x}_{i})/||\phi(\mathbf{x}_{i})||$, the output probability for the $k$ -th category is given by:

$$
\textstyle{\mathbf{p}_{i}}^{(k)}=\frac{\exp(\mathbf{h}_{i}\cdot\mathbf{c}_{k}/%
\tau_{s})}{\sum\nolimits_{j=1}^{K}\exp(\mathbf{h}_{i}\cdot\mathbf{c}_{j}/\tau_%
{s})},
$$

where $\tau_{s}$ is the scaling temperature for the ‘student’ view. The soft label $\mathbf{q}_{i}$ is generated by the ‘teacher’ view with a sharper temperature $\tau_{t}$ based on another augmented view in a similar manner. The self-distillation loss for the two views is then computed using the cross-entropy loss function: $\ell_{ce}(\mathbf{q}^{\prime},\mathbf{p})=-\sum\nolimits_{j=1}^{K}\mathbf{q}^{%
\prime(j)}\text{log}~{}\mathbf{p}^{(j)}$. The unsupervised loss is then computed by aggregating contributions from all samples in the mini-batch $B$ as follows:

$$
\textstyle\mathcal{L}_{cls}^{u}=\frac{1}{|{B}|}\sum_{i\in{B}}\ell_{ce}(\mathbf%
{q}^{\prime}_{i},\mathbf{p}_{i})-\xi\mathcal{H}(\overline{\mathbf{p}}),
$$

where $\overline{\mathbf{p}}=\frac{1}{2|{B}|}\sum\nolimits_{i\in{B}}(\mathbf{p}_{i}+%
\mathbf{p}^{\prime}_{i})$ denotes the mean prediction across the mini-batch. The mean entropy is defined as: $\mathcal{H}(\overline{\mathbf{p}})=-\sum\nolimits_{j=1}^{K}\overline{\mathbf{p%
}}^{(j)}\text{log}~{}\overline{\mathbf{p}}^{(j)}$, weighted by $\xi$.

For the labelled samples, the supervised classification loss is written as $\mathcal{L}_{cls}^{s}=\frac{1}{|{B}_{l}|}\sum\nolimits_{i\in{B}_{l}}\ell_{ce}(%
\mathbf{p}_{i},\mathbf{y}_{i})$, where $\mathbf{y}_{i}$ represents the one-hot vector corresponding to the ground-truth label $y_{i}$. The overall objective is $\mathcal{L}_{cls}=(1-\lambda_{b})\mathcal{L}_{cls}^{u}+\lambda_{b}\mathcal{L}_%
{cls}^{s}$. Integrating this with the representation learning loss $\mathcal{L}_{rep}$ adopted from [^53], the comprehensive training objective is expressed as: $\mathcal{L}_{gcd}=\mathcal{L}_{cls}+\mathcal{L}_{rep}.$ Through training with $\mathcal{L}_{gcd}$ on both $\mathbf{D}_{l}$ and $\mathbf{D}_{u}$, the classifier is empowered to directly predict labels for the unlabelled samples after the training process concludes.

![Refer to caption](https://arxiv.org/html/2504.06120v1/x3.png)

Figure 3: Overall pipeline of our HypCD framework for parametric and non-parametric GCD baselines. (a) Hyperbolic representation learning. (b) Hyperbolic classifier. (c) Non-parametric label assignment. (d) Parametric label assignment.

### 3.3 Hyperbolic Space for Category Discovery

As previously discussed, object parts are critical for facilitating knowledge transfer from labelled categories to unseen ones in GCD. Each sample inherently contains object parts that reside within a hierarchical structure. Moreover, existing GCD methods [^45] [^46] emphasize the inter-category hierarchy to enhance the clustering performance of unlabelled samples in Euclidean or spherical spaces. However, the geometry of representation space limits their ability to effectively capture other kinds of hierarchy [^31]. In contrast, hyperbolic space, characterized by its property of exponential volume growth with respect to the radius [^15], emerges as a more suitable space for GCD.

Hyperbolic space $\mathbb{H}^{n}$ is defined as an $n$ -dimensional Riemannian manifold exhibiting constant negative curvature, and it encompasses several analytic models [^5]. Following previous literature [^31] [^15], we employ the Poincaré ball [^39] model. In this model, the hyperbolic space is represented as an $n$ -dimensional ball $\mathbb{D}_{c}^{n}=\left\{\mathbf{a}\in\mathbb{R}^{n}\ \big{|}\ c\|\mathbf{a}%
\|^{2}<1\right\}$ with curvature value $-c^{2}$, where $c$ is the non-negative curvature parameter. The manifold is equipped with the Riemannian metric $g^{\mathbb{D}}=\lambda_{c}^{2}\,g^{\mathbb{E}}$ where $\lambda_{c}(\mathbf{a})=\frac{2}{1-c\|\mathbf{a}\|^{2}}$ is the conformal factor and $g^{\mathbb{E}}$ is the identity metric $\mathbf{I}_{n}$ in Euclidean space. In this way, the local distances are scaled by the factor $\lambda_{c}$ approaching infinity near the boundary of the ball. This gives rise to the exponential expansion property of hyperbolic spaces, unlike the polynomial expansion in Euclidean space. However, hyperbolic space is not vector space and thus operations such as addition can not be directly conducted. To address this problem, we leverage the gyrovector formalism [^50]. For a pair of points $\mathbf{a},\mathbf{b}\in\mathbb{D}_{c}^{n}$, their Möbius addition is defined as:

$$
\textstyle\mathbf{a}\oplus_{c}\mathbf{b}=\frac{(1+2c\langle\mathbf{a},\mathbf{%
b}\rangle+c\|\mathbf{b}\|^{2})\mathbf{a}+(1-c\|\mathbf{a}\|^{2})\mathbf{b}}{1+%
2c\langle\mathbf{a},\mathbf{b}\rangle+c^{2}\|\mathbf{a}\|^{2}\|\mathbf{b}\|^{2%
}}.
$$

The hyperbolic distance between them is then:

$$
\textstyle\mathcal{D}_{\mathbb{H}}(\mathbf{a},\mathbf{b})=\frac{2}{\sqrt{c}}%
\mathrm{arctanh}(\sqrt{c}\|-\mathbf{a}\oplus_{c}\mathbf{b}\|)
$$

When $c\rightarrow 0$, the hyperbolic distance (Eq. 6) closes to the Euclidean distance $\lim_{c\to 0}\mathcal{D}_{\mathbb{H}}(\mathbf{a},\mathbf{b})=2\|\mathbf{a}-%
\mathbf{b}\|.$

### 3.4 HypCD

As illustrated in Fig. 3, we propose a unified framework, HypCD, for category discovery in hyperbolic space, incorporating both parametric [^61] and non-parametric GCD approaches. Given two randomly augmented views, we initially obtain the respective Euclidean feature vectors $\mathbf{z}_{i}$ and $\mathbf{z}_{i}^{\prime}$ through a self-supervised pretrained backbone [^7] [^40]. Subsequently, the feature embeddings are mapped into hyperbolic space $\mathbb{H}^{n}$ using exponential mapping, facilitating representation learning within this exponentially growing space to more effectively capture and utilize the hierarchical relationships inherent in the training data.

The exponential mapping [^31] serves as a bijective projection between Euclidean space $\mathbb{E}^{n}$ and hyperbolic space $\mathbb{H}^{n}$. The projection of tangent vector $\mathbf{z}$ from $\mathbb{E}^{n}$ to $\mathbb{H}^{n}$ is formulated as:

$$
\exp_{\mathbf{\mathbf{o}}}^{c}(\mathbf{z})=\mathbf{o}\oplus_{c}\bigg{(}\tanh%
\bigg{(}\sqrt{c}\frac{\lambda_{\mathbf{o}}^{c}\|\mathbf{z}\|}{2}\bigg{)}\frac{%
\mathbf{z}}{\sqrt{c}\|\mathbf{z}\|}\bigg{)},
$$

where $\oplus_{c}$ is the Möbius addition, as introduced in Eq. 5 and $\mathbf{o}$ represents the base point of the mapping. To address the issue of gradient vanishing [^23] near the boundary of the Poincaré ball during training, we implement a feature clipping operation in $\mathbb{E}^{n}$ prior to the exponential mapping. The operation is defined as: $\mathcal{C}(\mathbf{z})=\min\{1,\frac{r}{||\mathbf{z}||}\}\cdot\mathbf{z}$, where $r$ denotes the clipping value. For the feature vector $\mathbf{z}_{i}$ in $\mathbb{E}^{n}$, the corresponding mapped feature in $\mathbb{H}^{n}$ is expressed as $\mathcal{M}(\mathbf{z}_{i})=\exp_{\mathbf{\mathbf{o}}}^{c}(\mathcal{C}(\mathbf%
{z}_{i}))$. The same operation will also be applied to the other feature vector $\mathbf{z}_{i}^{\prime}$.

As described in Sec.3.2, both parametric [^53] and non-parametric [^61] baselines utilize the same representation learning method. In our framework, we implement a consistent representation learning solution in hyperbolic space for them (Fig.3(a)). For parametric approaches, a hyperbolic parametric classifier is employed (Fig. 3(b)). We will introduce these components in detail subsequently.

Hyperbolic Representation Learning. Following prior attempts [^53] [^61] [^46], we incorporate both self-supervised and supervised contrastive learning into our framework. However, our approach uniquely operates within hyperbolic space. Furthermore, unlike previous GCD methods that exclusively utilize cosine distance [^53] [^61] [^57] (angle-based) or Euclidean distance [^46] (distance-based) for calculating pairwise similarity, we propose a hybrid approach that combines both distance-based and angle-based losses. Such integration has been shown to be more effective for model optimization in hyperbolic space [^18]. First, the unified form of self-supervised contrastive loss can be defined as:

$$
\textstyle\mathcal{L}^{u}=\frac{1}{|{B}|}\sum\limits_{i\in{B}}-\text{log}\frac%
{\exp(\mathcal{S}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{i}^{%
\prime}))/\tau_{r})}{\sum\nolimits_{j}^{j\neq i}\exp(\mathcal{S}(\mathcal{M}(%
\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{j}^{\prime}))/\tau_{r})}.
$$

Similarly, the supervised contrastive loss is unified as:

$$
\textstyle\mathcal{L}^{s}=\frac{1}{|{B}_{l}|}\!\!\sum\limits_{i\in{B}_{l}}\!\!%
\frac{1}{|{N}_{i}|}\!\!\!\sum\limits_{q\in{N}_{i}}\!\!\!-\text{log}\frac{\exp(%
\mathcal{S}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{q}))/\tau_{r})%
}{\sum\limits_{j\neq i}\exp(\mathcal{S}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M%
}(\mathbf{z}_{j}))/\tau_{r})},
$$

where $\mathcal{S}$ denotes the similarity function, which can be either distance-based or angle-based. For distance-based contrastive loss, we utilize $\mathcal{S}_{d}=-\mathcal{D}_{\mathbb{H}}$ as the similarity function, which is formally computed using negative Euclidean distance in prior methods [^46]. For angle-based contrastive loss, we employ the cosine similarity, formulated as:

$$
\textstyle\mathcal{S}_{a}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{%
i}^{\prime}))=\frac{\mathcal{M}(\mathbf{z}_{i})\cdot\mathcal{M}(\mathbf{z}_{i}%
^{\prime})}{||\mathcal{M}(\mathbf{z}_{i})||\cdot||\mathcal{M}(\mathbf{z}_{i}^{%
\prime})||}.
$$

Since hyperbolic space is conformal with Euclidean space, cosine similarity remains equivalent in both $\mathbb{E}^{n}$ and $\mathbb{H}^{n}$.

The final supervised and self-supervised hyperbolic contrastive loss is composed of both distance-based and angle-based losses:

$$
\textstyle\begin{split}&\mathcal{L}_{hrep}^{s}=\alpha_{d}\mathcal{L}_{dis}^{s}%
+(1-\alpha_{d})\mathcal{L}_{ang}^{s},\\
&\mathcal{L}_{hrep}^{u}=\alpha_{d}\mathcal{L}_{dis}^{u}+(1-\alpha_{d})\mathcal%
{L}_{ang}^{u},\end{split}
$$

where $\mathcal{L}_{hrep}^{s}$ and $\mathcal{L}_{hrep}^{u}$ represent the supervised and self-supervised hyperbolic contrastive loss, respectively. The terms $\mathcal{L}_{dis}$ and $\mathcal{L}_{ang}$ correspond to distance-based and angle-based contrastive loss, respectively, obtained by substituting $\mathcal{S}$ with $\mathcal{S}_{d}$ and $\mathcal{S}_{a}$. Additionally, $\alpha_{d}$ is the loss weight of distance-based loss. The overall training objective for hyperbolic representation learning is:

$$
{\textstyle\mathcal{L}_{rep}^{\mathbb{H}}=(1-\lambda_{b}^{\mathbb{H}})\mathcal%
{L}_{hrep}^{u}+\lambda_{b}^{\mathbb{H}}\mathcal{L}_{hrep}^{s},}
$$

where $\lambda_{b}^{\mathbb{H}}$ serves as the balancing factor between the supervised and unsupervised losses.

Hyperbolic Classifier. To enhance the parametric baseline with hyperbolic geometry, we replace the conventional Euclidean classification head—traditionally reliant on a multilayer perceptron (MLP) in Euclidean space—with its hyperbolic counterpart, the hyperbolic feed forward network (HypFFN). The hyperbolic linear layer [^19] exhibits greater alignment with the baseline [^61], and we experimentally find that it outperforms the hyperbolic multinomial logistic regression layer. Consider the last linear layer of the MLP; similar to its Euclidean counterpart, the hyperbolic linear layer is parameterized by a weight matrix $\mathbf{w}\in\mathbb{R}^{I\times K}$ and a bias vector $\mathbf{s}\in\mathbb{R}^{1\times K}$, where $I$ denotes the input feature dimension. Given the hyperbolic feature $\mathbf{z}^{\mathbb{H}}_{i}=\mathcal{M}(\mathbf{z}_{i})\in\mathbb{R}^{1\times I}$, the linear layer operates as HypLinear $(\mathbf{z}^{\mathbb{H}}_{i},\mathbf{w},\mathbf{s})=\texttt{Proj}[(\mathbf{w}%
\otimes_{c}\mathbf{z}^{\mathbb{H}}_{i})\oplus_{c}\mathbf{s}]$, where $\oplus_{c}$ follows Eq. 5. The Möbius matrix-vector multiplication $\mathbf{v}_{i}=\mathbf{w}\otimes_{c}\mathbf{z}^{\mathbb{H}}_{i}$ is defined as:

$$
\textstyle\frac{1}{\sqrt{c}}\tanh\left(\frac{\|\mathbf{z}^{\mathbb{H}}_{i}%
\mathbf{w}\|_{2}}{\|\mathbf{z}^{\mathbb{H}}_{i}\|_{2}}\tanh^{-1}(\sqrt{c}\|%
\mathbf{z}^{\mathbb{H}}_{i}\|_{2})\right)\frac{\mathbf{z}^{\mathbb{H}}_{i}%
\mathbf{w}}{\|\mathbf{z}^{\mathbb{H}}_{i}\mathbf{w}\|_{2}}.
$$

To ensure numerical stability [^19], a safe projection is operated on the result manifold and represented as:

$$
\textstyle\texttt{Proj}(\mathbf{v}_{i})\!=\!\!\\
\begin{cases}\frac{\mathbf{v}_{i}}{\|\mathbf{v}_{i}\|_{2}}\times\frac{1-10^{-3%
}}{\sqrt{c}},&\!\!\!\!\!\frac{1-10^{-3}}{\sqrt{c}}<\|\mathbf{v}_{i}\|_{2}\\
\mathbf{v}_{i},&\text{otherwise}\end{cases}.
$$

This integration allows our hyperbolic classifier to be seamlessly incorporated into the baseline [^61] by substituting the original MLP with HypFFN. For each point in $\mathbb{H}^{n}$, the tangent space at that point serves as a Euclidean subspace, enabling straightforward adaptation of Euclidean operations within this space [^10]. Consequently, the cross-entropy loss for the hyperbolic classifier can be expressed as: $\ell_{ce}^{\mathbb{H}}=\ell_{ce}(\texttt{HypFFN}(\mathbf{z}^{\mathbb{H}}_{i}),%
\mathbf{y}_{i})$. Additionally, we can define the hyperbolic counterpart $\mathcal{H}^{\mathbb{H}}$ for the mean entropy $\mathcal{H}$. By substituting the original $\ell_{ce}$ and $\mathcal{H}$ with our derived $\ell_{ce}^{\mathbb{H}}$ and $\mathcal{H}^{\mathbb{H}}$, respectively, we can readily compute the final hyperbolic classifier loss $\mathcal{L}^{\mathbb{H}}_{cls}$, as detailed in Sec. 3.2.

### 3.5 Label Assignment

Existing approaches typically employ either a parametric classification head or non-parametric methods, such as semi-supervised k-means [^53], for label assignment. In this paper, we do not independently assess these two methods; rather, we integrate both within the HypCD framework as shown in Fig. 3(c) and (d). For non-parametric approaches, including [^53] and the recent SelEx [^46], we retain the original label assignment strategy by applying semi-supervised k-means clustering directly to feature representations extracted by $\phi$ in $\mathbb{E}^{n}$. Our empirical results indicate that training in hyperbolic space allows for the transfer of hierarchical structure encoding from $\mathbb{H}^{n}$ to $\mathbb{E}^{n}$. Moreover, we find that the operations of k-means in $\mathbb{E}^{n}$ are significantly more efficient while maintaining comparable performance. For the parametric baseline exemplified by SimGCD [^61], we utilize the hyperbolic classification head to conduct classification within hyperbolic space using the trained hyperbolic classifier HypFFN. Both design choices are theoretically supported by the property of hyperbolic geometry of encoding hierarchical structures, facilitating a more intuitive and effective representation and classifier for GCD.

## 4 Experiment

Table 1: Comparison of GCD methods on the SSB [^54] benchmark, CIFAR-10 [^34], CIFAR-100 [^34] and ImageNet-100 [^13] datasets. Results are reported in ACC across the ‘All’, ‘Old’ and ‘New’ categories.

<table><tbody><tr><th></th><th></th><td colspan="3">CUB <sup><a href="#fn:56">56</a></sup></td><td colspan="3">Stanford-Cars <sup><a href="#fn:33">33</a></sup></td><td colspan="3">FGVC-Aircraft <sup><a href="#fn:38">38</a></sup></td><td colspan="3">CIFAR-10 <sup><a href="#fn:34">34</a></sup></td><td colspan="3">CIFAR-100 <sup><a href="#fn:34">34</a></sup></td><td colspan="3">ImageNet-100 <sup><a href="#fn:13">13</a></sup></td></tr><tr><th></th><th>Method</th><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><th rowspan="19"><em>DINO</em></th><th>ORCA <sup><a href="#fn:6">6</a></sup></th><td>36.3</td><td>43.8</td><td>32.6</td><td>31.6</td><td>32.0</td><td>31.4</td><td>31.9</td><td>42.2</td><td>26.9</td><td>69.0</td><td>77.4</td><td>52.0</td><td>73.5</td><td>92.6</td><td>63.9</td><td>81.8</td><td>86.2</td><td>79.6</td></tr><tr><th>XCon <sup><a href="#fn:16">16</a></sup></th><td>52.1</td><td>54.3</td><td>51.0</td><td>40.5</td><td>58.8</td><td>31.7</td><td>47.7</td><td>44.4</td><td>49.4</td><td>96.0</td><td>97.3</td><td>95.4</td><td>74.2</td><td>81.2</td><td>60.3</td><td>77.6</td><td>93.5</td><td>69.7</td></tr><tr><th>OpenCon <sup><a href="#fn:48">48</a></sup></th><td>54.7</td><td>63.8</td><td>54.7</td><td>49.1</td><td>78.6</td><td>32.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>84.0</td><td>93.8</td><td>81.2</td></tr><tr><th>PromptCAL <sup><a href="#fn:63">63</a></sup></th><td>62.9</td><td>64.4</td><td>62.1</td><td>50.2</td><td>70.1</td><td>40.6</td><td>52.2</td><td>52.2</td><td>52.3</td><td>97.9</td><td>96.6</td><td>98.5</td><td>81.2</td><td>84.2</td><td>75.3</td><td>83.1</td><td>92.7</td><td>78.3</td></tr><tr><th>DCCL <sup><a href="#fn:43">43</a></sup></th><td>63.5</td><td>60.8</td><td>64.9</td><td>43.1</td><td>55.7</td><td>36.2</td><td>-</td><td>-</td><td>-</td><td>96.3</td><td>96.5</td><td>96.9</td><td>75.3</td><td>76.8</td><td>70.2</td><td>80.5</td><td>90.5</td><td>76.2</td></tr><tr><th>GPC <sup><a href="#fn:66">66</a></sup></th><td>52.0</td><td>55.5</td><td>47.5</td><td>38.2</td><td>58.9</td><td>27.4</td><td>43.3</td><td>40.7</td><td>44.8</td><td>90.6</td><td>97.6</td><td>87.0</td><td>75.4</td><td>84.6</td><td>60.1</td><td>75.3</td><td>93.4</td><td>66.7</td></tr><tr><th>PIM <sup><a href="#fn:11">11</a></sup></th><td>62.7</td><td>75.7</td><td>56.2</td><td>43.1</td><td>66.9</td><td>31.6</td><td>-</td><td>-</td><td>-</td><td>94.7</td><td>97.4</td><td>93.3</td><td>78.3</td><td>84.2</td><td>66.5</td><td>83.1</td><td>95.3</td><td>77.0</td></tr><tr><th><math><semantics><mi>μ</mi> <ci>𝜇</ci> <annotation>\mu</annotation> <annotation>italic_μ</annotation></semantics></math> GCD <sup><a href="#fn:55">55</a></sup></th><td>65.7</td><td>68.0</td><td>64.6</td><td>56.5</td><td>68.1</td><td>50.9</td><td>53.8</td><td>55.4</td><td>53.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>InfoSieve <sup><a href="#fn:45">45</a></sup></th><td>69.4</td><td>77.9</td><td>65.2</td><td>55.7</td><td>74.8</td><td>46.4</td><td>56.3</td><td>63.7</td><td>52.5</td><td>94.8</td><td>97.7</td><td>93.4</td><td>78.3</td><td>82.2</td><td>70.5</td><td>80.5</td><td>93.8</td><td>73.8</td></tr><tr><th>CiPR <sup><a href="#fn:27">27</a></sup></th><td>57.1</td><td>58.7</td><td>55.6</td><td>47.0</td><td>61.5</td><td>40.1</td><td>-</td><td>-</td><td>-</td><td>97.7</td><td>97.5</td><td>97.7</td><td>81.5</td><td>82.4</td><td>79.7</td><td>80.5</td><td>84.9</td><td>78.3</td></tr><tr><th>SPTNet <sup><a href="#fn:57">57</a></sup></th><td>65.8</td><td>68.8</td><td>65.1</td><td>59.0</td><td>79.2</td><td>49.3</td><td>59.3</td><td>61.8</td><td>58.1</td><td>97.3</td><td>95.0</td><td>98.6</td><td>81.3</td><td>84.3</td><td>75.6</td><td>85.4</td><td>93.2</td><td>81.4</td></tr><tr><th>CMS <sup><a href="#fn:12">12</a></sup></th><td>68.2</td><td>76.5</td><td>64.0</td><td>56.9</td><td>76.1</td><td>47.6</td><td>56.0</td><td>63.4</td><td>52.3</td><td>-</td><td>-</td><td>-</td><td>82.3</td><td>85.7</td><td>75.5</td><td>84.7</td><td>95.6</td><td>79.2</td></tr><tr><th>AMEND <sup><a href="#fn:2">2</a></sup></th><td>64.9</td><td>75.6</td><td>59.6</td><td>52.8</td><td>61.8</td><td>48.3</td><td>56.4</td><td>73.3</td><td>48.2</td><td>96.8</td><td>94.6</td><td>97.8</td><td>81.0</td><td>79.9</td><td>83.3</td><td>83.2</td><td>92.9</td><td>78.3</td></tr><tr><th>GCD <sup><a href="#fn:53">53</a></sup></th><td>51.3</td><td>56.6</td><td>48.7</td><td>39.0</td><td>57.6</td><td>29.9</td><td>45.0</td><td>41.1</td><td>46.9</td><td>91.5</td><td>97.9</td><td>88.2</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td></tr><tr><th>Hyp-GCD</th><td>61.0</td><td>67.0</td><td>58.0</td><td>50.8</td><td>60.9</td><td>45.8</td><td>48.2</td><td>43.6</td><td>50.5</td><td>92.9</td><td>97.5</td><td>90.6</td><td>74.0</td><td>80.0</td><td>62.0</td><td>80.4</td><td>92.5</td><td>74.4</td></tr><tr><th>SimGCD <sup><a href="#fn:61">61</a></sup></th><td>60.3</td><td>65.6</td><td>57.7</td><td>53.8</td><td>71.9</td><td>45.0</td><td>54.2</td><td>59.1</td><td>51.8</td><td>97.1</td><td>95.1</td><td>98.1</td><td>80.1</td><td>81.2</td><td>77.8</td><td>83.0</td><td>93.1</td><td>77.9</td></tr><tr><th>Hyp-SimGCD</th><td>64.8</td><td>65.8</td><td>64.2</td><td>62.8</td><td>73.4</td><td>57.7</td><td>58.7</td><td>58.9</td><td>58.5</td><td>96.8</td><td>95.9</td><td>97.2</td><td>82.4</td><td>83.1</td><td>81.2</td><td>86.5</td><td>93.7</td><td>83.0</td></tr><tr><th>SelEx <sup><a href="#fn:46">46</a></sup></th><td>73.6</td><td>75.3</td><td>72.8</td><td>58.5</td><td>75.6</td><td>50.3</td><td>57.1</td><td>64.7</td><td>53.3</td><td>95.9</td><td>98.1</td><td>94.8</td><td>82.3</td><td>85.3</td><td>76.3</td><td>83.1</td><td>93.6</td><td>77.8</td></tr><tr><th>Hyp-SelEx</th><td>79.8</td><td>75.8</td><td>81.8</td><td>62.9</td><td>80.0</td><td>54.7</td><td>65.9</td><td>67.3</td><td>65.1</td><td>96.7</td><td>97.6</td><td>96.3</td><td>82.4</td><td>85.1</td><td>77.0</td><td>86.8</td><td>94.6</td><td>82.8</td></tr><tr><th rowspan="9"><em>DINOv2</em></th><th><math><semantics><mi>μ</mi> <ci>𝜇</ci> <annotation>\mu</annotation> <annotation>italic_μ</annotation></semantics></math> GCD <sup><a href="#fn:55">55</a></sup></th><td>74.0</td><td>75.9</td><td>73.1</td><td>76.1</td><td>91.0</td><td>68.9</td><td>66.3</td><td>68.7</td><td>65.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>CiPR <sup><a href="#fn:27">27</a></sup></th><td>78.3</td><td>73.4</td><td>80.8</td><td>66.7</td><td>77.0</td><td>61.8</td><td>-</td><td>-</td><td>-</td><td>99.0</td><td>98.7</td><td>99.2</td><td>90.3</td><td>89.0</td><td>93.1</td><td>88.2</td><td>87.6</td><td>88.5</td></tr><tr><th>SPTNet <sup><a href="#fn:57">57</a></sup></th><td>76.3</td><td>79.5</td><td>74.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.1</td><td>96.1</td><td>87.1</td></tr><tr><th>GCD <sup><a href="#fn:53">53</a></sup></th><td>71.9</td><td>71.2</td><td>72.3</td><td>65.7</td><td>67.8</td><td>64.7</td><td>55.4</td><td>47.9</td><td>59.2</td><td>97.8</td><td>99.0</td><td>97.1</td><td>79.6</td><td>84.5</td><td>69.9</td><td>78.5</td><td>89.5</td><td>73.0</td></tr><tr><th>Hyp-GCD</th><td>75.6</td><td>75.1</td><td>75.9</td><td>72.8</td><td>80.4</td><td>69.1</td><td>62.7</td><td>70.0</td><td>59.0</td><td>97.5</td><td>98.9</td><td>96.8</td><td>84.5</td><td>87.5</td><td>78.5</td><td>82.9</td><td>92.4</td><td>78.2</td></tr><tr><th>SimGCD <sup><a href="#fn:61">61</a></sup></th><td>71.5</td><td>78.1</td><td>68.3</td><td>71.5</td><td>81.9</td><td>66.6</td><td>63.9</td><td>69.9</td><td>60.9</td><td>98.7</td><td>96.7</td><td>99.7</td><td>88.5</td><td>89.2</td><td>87.2</td><td>89.9</td><td>95.5</td><td>87.1</td></tr><tr><th>Hyp-SimGCD</th><td>77.6</td><td>77.9</td><td>77.4</td><td>82.5</td><td>85.8</td><td>81.0</td><td>76.4</td><td>70.3</td><td>79.4</td><td>98.9</td><td>97.7</td><td>99.5</td><td>91.5</td><td>90.0</td><td>94.6</td><td>91.9</td><td>96.2</td><td>89.8</td></tr><tr><th>SelEx <sup><a href="#fn:46">46</a></sup></th><td>87.4</td><td>85.1</td><td>88.5</td><td>82.2</td><td>93.7</td><td>76.7</td><td>79.8</td><td>82.3</td><td>78.6</td><td>98.5*</td><td>98.8*</td><td>98.5*</td><td>87.7*</td><td>90.8*</td><td>81.5*</td><td>90.9*</td><td>96.2*</td><td>88.3*</td></tr><tr><th>Hyp-SelEx</th><td>90.7</td><td>85.3</td><td>93.4</td><td>83.8</td><td>93.3</td><td>79.2</td><td>83.4</td><td>82.0</td><td>84.1</td><td>98.6</td><td>98.1</td><td>98.9</td><td>88.6</td><td>91.5</td><td>82.8</td><td>92.3</td><td>96.4</td><td>90.2</td></tr></tbody></table>

\*results from our implementation.

### 4.1 Setups and Implementations

Datasets. We thoroughly evaluate our method across diverse benchmarks, including the generic image recognition datasets CIFAR-10 and CIFAR-100 [^34], as well as ImageNet-100 [^13]. Additionally, we assess our approach on the Semantic Shift Benchmark (SSB) [^54], which includes fine-grained datasets such as CUB [^56], Stanford-Cars [^33], and FGVC-Aircraft [^38]. For each dataset, we adhere to the data split scheme detailed in [^53]. The method involves sampling a subset of all classes as the known (‘Old’) classes $\mathbf{Y}_{l}$. Subsequently, $50\%$ of the images from these known classes are utilized to construct $\mathbf{D}_{l}$, while the remaining images are designated as the unlabelled data $\mathbf{D}_{u}$.

Evaluation Metrics. We evaluate the performance using the clustering accuracy (ACC) as defined in the literature [^53]. The ACC on $\mathbf{D}_{u}$ is computed as follows, given the ground truth $y_{i}$ and the predicted labels $\hat{y}_{i}$: $\textstyle\textit{ACC}=\frac{1}{|\mathbf{D}_{u}|}\sum\nolimits_{i=1}^{|\mathbf%
{D}_{u}|}\mathbbm{1}(y_{i}=h(\hat{y}_{i}))$, where $h$ denotes the optimal permutation that aligns the predicted cluster assignments with the ground-truth labels. The ACC values for the ‘All’, ‘Old’ and ‘New’ classes are reported separately.

Implementation Details. We evaluate HypCD against the non-parametric baseline GCD [^53], the parametric baseline SimGCD [^61], and the SOTA method SelEx [^46], utilizing both DINO [^7] and DINOv2 [^40] pretrained weights. Detailed information regarding SelEx can be found in the supplementary materials. For GCD [^53], the output dimension of the projection head $\rho_{r}$ is $256$. In the case of SimGCD [^61], the feature dimension from backbone $\phi$ is $768$. $\rho_{r}$ and the final block of $\phi$ are optimized using the SGD optimizer, with an initial learning rate of $0.1$, which is decayed to $0.001$ over time according to a cosine annealing schedule. The HypFFN is optimized using the Riemannian Adam optimizer [^4], with a constant learning rate of $0.01$. All models are trained for $200$ epochs using a batch size of $128$. The curvature parameter $c$ is set to $0.05$ for the fine-grained datasets and $0.01$ for the generic datasets. Following baselines, the balancing factor $\lambda_{b}^{\mathbb{H}}$ is set to $0.35$. By default, the loss weight $\alpha_{d}$ increases linearly from $00$ to $1.0$.

### 4.2 Quantitative Comparison

We compare our method with recent GCD methods (including ORCA [^6], GCD [^53], XCon [^16], OpenCon [^48], PromptCAL [^63], DCCL [^43], GPC [^66], CiPR [^27], SimGCD [^61], $\mu$ GCD [^55], InfoSieve [^45], SPTNet [^57], CMS [^12], AMEND [^2] and SelEx [^46]) and report the results in Tab. 1. The evaluation encompasses performance on the SSB benchmark [^54] and generic datasets [^34] [^13]. The hyperbolic methods applying our HypCD framework are indicated by the ‘Hyp-’ prefix.

![Refer to caption](https://arxiv.org/html/2504.06120v1/x4.png)

Figure 4: Comparison of baseline and hyperbolic counterparts on the SSB. Left: ‘All’ ACC (higher is better). Right: Discrepancy between ‘Old’ and ‘New’ (smaller is better).

Results on SSB. The performance of the GCD methods on the SSB benchmark, utilizing both DINO [^7] and DINOv2 [^40] pretrained weights, is summarized in the left section of Tab. 1. Besides, we provide a comparative analysis between the three baseline methods and their hyperbolic counterparts with DINO backbone in Fig. 4. Our hyperbolic methods consistently outperform their Euclidean counterparts, with particularly strong results observed when utilizing the DINOv2 backbone. Among the evaluated methods, Hyp-SelEx achieves the highest average accuracy (ACC) across all datasets, notably excelling on the CUB dataset, where it records an accuracy of 79.8% for the ‘All’ classes with DINO and 90.7% with DINOv2, establishing it as the leading approach. More strikingly, on the Stanford-Cars dataset, our Hyp-GCD method outperforms the baseline by 11.8%, 13.3% and 15.9% in terms of ACC for the ‘All’, ‘Old’ and ‘New’ categories, respectively. Fig. 4 (left) illustrates the average ACC on ‘All’ categories across the three datasets in the SSB benchmark, indicating that our hyperbolic methods surpass the baseline by a margin of at least 6.0%. Furthermore, as shown in Fig. 4 (right), hyperbolic methods exhibit a consistently smaller ACC gap between ‘Old’ and ‘New’ classes, highlighting the effectiveness of HypCD in enhancing knowledge transfer from known to unseen categories. Additionally, DINOv2 outperforms DINO across all methods, underscoring its ability to capture complex data representations more effectively.

Results on Generic Datasets. In the right section of Tab. 1, we present the results on three widely used generic datasets: CIFAR-10 [^34], CIFAR-100 [^34], and ImageNet-100 [^13]. Our methods demonstrate consistent improvements across all cases, regardless of the backbone employed. Notably, these enhancements are especially significant on CIFAR-100 and ImageNet-100, which present greater challenges compared to CIFAR-10, where performance is nearly saturated. For CIFAR-100, Hyp-SimGCD and Hyp-SelEx achieve the highest accuracy of 82.4% for the ‘All’ categories using DINO, while Hyp-SimGCD ranks first with an accuracy of 91.5% on this metric when utilizing DINOv2, significantly surpassing baseline methods and the previous SOTA. Results on ImageNet-100 further validate the effectiveness of hyperbolic methods; Hyp-SelEx achieves the highest performance across ‘All’, ‘Old’, and ‘New’ categories with both DINO and DINOv2, outperforming the baseline by a margin of up to 3.7%.

Table 2: Experimental results using different $c$, $r$ and $\alpha_{d}^{\text{max}}$ values in Hyp-SimGCD with DINO [^7] pre-trained backbone. Results on the CUB [^56] and CIFAR-100 [^34] datasets are reported.

<table><thead><tr><th></th><th colspan="3">Stanford-Cars <sup><a href="#fn:33">33</a></sup></th><th colspan="3">CIFAR-100 <sup><a href="#fn:34">34</a></sup></th></tr><tr><th>parameter</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><th><math><semantics><mi>c</mi> <ci>𝑐</ci> <annotation>c</annotation> <annotation>italic_c</annotation></semantics></math> = 0.01</th><td>61.4</td><td>74.4</td><td>55.1</td><td>82.4</td><td>83.1</td><td>81.2</td></tr><tr><th><math><semantics><mi>c</mi> <ci>𝑐</ci> <annotation>c</annotation> <annotation>italic_c</annotation></semantics></math> = 0.05</th><td>62.8</td><td>73.4</td><td>57.7</td><td>81.6</td><td>84.0</td><td>76.7</td></tr><tr><th><math><semantics><mi>c</mi> <ci>𝑐</ci> <annotation>c</annotation> <annotation>italic_c</annotation></semantics></math> = 0.1</th><td>62.3</td><td>75.1</td><td>56.1</td><td>81.1</td><td>82.3</td><td>78.8</td></tr><tr><th><math><semantics><mi>r</mi> <ci>𝑟</ci> <annotation>r</annotation> <annotation>italic_r</annotation></semantics></math> = 1.0</th><td>60.0</td><td>72.9</td><td>53.7</td><td>82.4</td><td>83.1</td><td>81.2</td></tr><tr><th><math><semantics><mi>r</mi> <ci>𝑟</ci> <annotation>r</annotation> <annotation>italic_r</annotation></semantics></math> = 1.5</th><td>61.2</td><td>75.7</td><td>54.2</td><td>81.2</td><td>82.4</td><td>78.8</td></tr><tr><th><math><semantics><mi>r</mi> <ci>𝑟</ci> <annotation>r</annotation> <annotation>italic_r</annotation></semantics></math> = 2.3</th><td>62.8</td><td>73.4</td><td>57.7</td><td>80.1</td><td>81.1</td><td>78.3</td></tr><tr><th><math><semantics><msubsup><mi>α</mi> <mi>d</mi> <mtext>max</mtext></msubsup> <apply><csymbol>superscript</csymbol> <apply><csymbol>subscript</csymbol> <ci>𝛼</ci> <ci>𝑑</ci></apply> <ci><mtext>max</mtext></ci></apply> <annotation>\alpha_{d}^{\text{max}}</annotation> <annotation>italic_α start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT max end_POSTSUPERSCRIPT</annotation></semantics></math> = 0.1</th><td>59.6</td><td>77.5</td><td>51.0</td><td>81.3</td><td>82.4</td><td>79.1</td></tr><tr><th><math><semantics><msubsup><mi>α</mi> <mi>d</mi> <mtext>max</mtext></msubsup> <apply><csymbol>superscript</csymbol> <apply><csymbol>subscript</csymbol> <ci>𝛼</ci> <ci>𝑑</ci></apply> <ci><mtext>max</mtext></ci></apply> <annotation>\alpha_{d}^{\text{max}}</annotation> <annotation>italic_α start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT max end_POSTSUPERSCRIPT</annotation></semantics></math> = 0.5</th><td>62.0</td><td>77.2</td><td>54.6</td><td>82.4</td><td>83.1</td><td>81.2</td></tr><tr><th><math><semantics><msubsup><mi>α</mi> <mi>d</mi> <mtext>max</mtext></msubsup> <apply><csymbol>superscript</csymbol> <apply><csymbol>subscript</csymbol> <ci>𝛼</ci> <ci>𝑑</ci></apply> <ci><mtext>max</mtext></ci></apply> <annotation>\alpha_{d}^{\text{max}}</annotation> <annotation>italic_α start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT max end_POSTSUPERSCRIPT</annotation></semantics></math> = 1.0</th><td>62.8</td><td>73.4</td><td>57.7</td><td>78.9</td><td>83.5</td><td>69.7</td></tr></tbody></table>

### 4.3 Impact of Hyperparameters

Manifold Curvature. Building on previous studies [^31] [^1] that explore the application of hyperbolic geometry across various tasks, the curvature parameter $c$ (as discussed in Sec. 3.3) is a crucial factor influencing performance and may yield different optimal values across datasets and methods. Intuitively, as the value of $c$ approaches $00$, the radius tends toward infinity, causing the Poincaré ball to flatten and resemble Euclidean space; conversely, larger values of $c$ correspond to a steeper configuration. The widely accepted range for $c$ is between $0.01$ and $0.3$ [^15], with larger values exceeding this range resulting in performance degradation. In our experiments, we evaluate different curvature values of $0.01$, $0.05$, and $0.1$ using Hyp-SimGCD, as presented in Tab. 2. Our findings indicate that the optimal curvature values differ between generic and fine-grained datasets. For fine-grained datasets such as CUB [^56], the optimal value is $0.05$, while for generic datasets like CIFAR-100, a value of $0.01$ proves to be more effective.

Clipping Value. As articulated in [^23], feature clipping has emerged as a standard technique for training hyperbolic neural networks. In our framework, we also observe that it plays a crucial role in category discovery performance. In line with the methodology outlined in [^23], we investigate a range of clipping values, specifically $1.0$, $1.5$, and $2.3$. The results shown in the second row of Tab. 2 demonstrate that optimal clipping values vary between fine-grained and generic datasets. For fine-grained datasets like CUB [^56], the optimal clipping value is determined to be $2.3$. Conversely, for generic datasets like CIFAR-100 [^34], a clipping value of $1.0$ is shown to be more effective.

Loss Weight. As detailed in Sec. 3.4, we implement a hybrid contrastive loss that combines both distance-based and angle-based components, which is essential for effective optimization in hyperbolic space. A loss weight, denoted as $\alpha_{d}$, is introduced to regulate the balance between these two types of losses and linearly increasing from $00$. In the initial stages, the model prioritizes optimizing the angle between sample points and progressively shifts focus toward optimizing the hyperbolic distance. Consistent with the observations for curvature and clipping values, the optimal max value $\alpha_{d}^{\text{max}}$, varies considerably between coarse-grained and fine-grained datasets. For fine-grained datasets such as CUB [^56], an optimal value of $1.0$ is observed, whereas for more generic datasets like CIFAR-100 [^34], a value of $0.5$ is found to yield better performance.

### 4.4 Qualitative Comparison

![Refer to caption](https://arxiv.org/html/2504.06120v1/x5.png)

Figure 5: T-SNE 51 comparison between SimGCD 61 and our Hyp-SimGCD using 40 randomly sampled instances from 10 randomly selected categories of the Stanford-Cars dataset 33.

In Fig. 5, we present a t-SNE [^51] visualization of features extracted from the backbone, represented as $\mathbf{z}_{i}=\phi(\mathbf{x}_{i})$. This visualization compares SimGCD with our Hyp-SimGCD. On the left side of the figure, the clusters generated by SimGCD appear dispersed. Data points from Class 42, highlighted in pink, are spread across multiple areas, indicating significant overlap and a lack of compactness. In contrast, Hyp-SimGCD creates more distinct and tightly clustered groups, concentrating the data points of Class 42 in a more confined area. This comparison implies that Hyp-SimGCD enhances both intra-class compactness and inter-class separation through our hyperbolic representation and classifier learning method. Importantly, even within the original Euclidean space of the backbone network, Hyp-SimGCD exhibits robust clustering performance, which arises from the properties of hyperbolic space in encoding hierarchical structures.

## 5 Conclusion

In this paper, we investigate a previously overlooked perspective in GCD by utilizing a representation space that captures the hierarchical structure of each sample, instead of the conventional Euclidean or spherical spaces. Our approach leverages the distinctive properties of hyperbolic space, where the volume increases exponentially with radius. This characteristic makes hyperbolic space especially suitable for modelling data possessing hierarchical structures, thereby enhancing representational capacity for category discovery. We propose a simple yet effective framework, HypCD, for integrating hyperbolic geometry into GCD methods. Through extensive experiments with parametric and non-parametric GCD baselines and the SOTA method, our framework consistently demonstrates superior performance on public benchmarks, underscoring the effectiveness of hyperbolic space for category discovery.

#### Acknowledgements.

This work is supported by National Natural Science Foundation of China (Grant No. 62306251), Hong Kong Research Grant Council - Early Career Scheme (Grant No. 27208022) and General Research Fund (Grant No. 17211024), and HKU Seed Fund for Basic Research.

## References

Supplementary Material

We provide additional details and experimental results in this supplementary material, which is organized as follows:

- §6 More Experimental Details
- §7 More Quantitative Results
- §8 More Qualitative Results

## 6 More Experimental Details

### 6.1 Dataset Statistics

For each dataset, we adhere to the data splitting scheme described in [^53]. In this scheme, $50\%$ of the classes will be sampled as ‘Old’, with the exception of CIFAR-100, which samples $80\%$ of the classes. Following this, $50\%$ of the images from known classes are used to create the labelled dataset $\mathbf{D}_{l}$, while the remaining images are allocated to the unlabelled dataset $\mathbf{D}_{u}$. The statistics for all the datasets utilized in this work are summarized in Tab. 3.

Table 3: Overview of the dataset, including the classes in the labelled and unlabelled sets ($M=|\mathbf{Y}_{l}|$, $K=|\mathbf{Y}_{l}\cup\mathbf{Y}_{u}|$) and counts of images ($|\mathbf{D}_{l}|$, $|\mathbf{D}_{u}|$). ‘FG’ denotes fine-grained.

| Dataset | FG | $\|\mathbf{D}_{l}\|$ | $M$ | $\|\mathbf{D}_{u}\|$ | $K$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 [^34] | ✗ | 12.5K | 5 | 37.5K | 10 |
| CIFAR-100 [^34] | ✗ | 20.0K | 80 | 30.0K | 100 |
| ImageNet-100 [^13] | ✗ | 31.9K | 50 | 95.3K | 100 |
| CUB [^56] | ✓ | 1.5K | 100 | 4.5K | 200 |
| Stanford-Cars [^33] | ✓ | 2.0K | 98 | 6.1K | 196 |
| FGVC-Aircraft [^38] | ✓ | 1.7K | 50 | 5.0K | 100 |
| Herbarium19 [^49] | ✓ | 8.9K | 341 | 25.4K | 683 |
| Oxford-Pet [^42] | ✓ | 0.9K | 19 | 2.7K | 37 |

### 6.2 Additional Implementation Details

Consistent with prior studies [^46] [^61] [^53], we employ the ViT-B architecture [^14] with pretrained weights from either DINO [^7] or DINOv2 [^40] as our backbone network. For our proposed hyperbolic methods, we adhere to nearly all hyperparameter settings established in [^46] [^61] [^53] to facilitate fair comparisons with their respective baselines. The specific details are summarized as follows: For Hyp-SimGCD and Hyp-GCD, only the last block of the backbone is fine-tuned across all datasets. In contrast, Hyp-SelEx implements dataset-specific fine-tuning: the last two blocks are fine-tuned for CUB [^56], FGVC-Aircraft [^38], and all generic datasets, while the last three blocks are fine-tuned for Stanford-Cars [^33]. Regarding method-specific hyperparameters, for Hyp-SimGCD, we set the weight $\xi$, which controls the weight of mean entropy loss, to $1.0$ for all the datasets. For Hyp-SelEx, we follow [^46] in setting $\alpha$, which regulates label smoothing, to $0.5$ for FGVC-Aircraft [^38], $1.0$ for CUB [^56] and Stanford-Cars [^33], and $0.1$ for generic datasets. Additionally, the proposed parameter $\alpha_{d}$, which balances distance-based and angle-based losses, linearly increases from $00$ to its maximum value during training according to the formula: $\alpha_{d}=\frac{{e}*\alpha_{d}^{\text{max}}}{200}$, where $e$ is the current training epoch. Specifically, we set $\alpha_{d}^{\text{max}}$ to $1$ for fine-grained and $0.5$ for generic datasets.

### 6.3 Details of Hyp-SelEx

[^46] proposes a hierarchical non-parametric method, SelEx, to address fine-grained GCD through a novel concept of self-expertise. It begins by constructing hierarchical pseudo-labeling via a balanced semi-supervised k-means algorithm to initialize clusters for known categories and then iteratively refines them by incorporating an equal number of random samples for unseen categories to balance cluster distribution. Following it, supervised self-expertise leverages weakly-supervised pseudo labels to group samples by capturing abstract-level similarity, whereas unsupervised self-expertise focuses on distinguishing semantically similar hard negative samples within the same clusters to sharpen fine-grained categorization.

Its representation learning objective composes of unsupervised self-expertise loss $\mathcal{L}_{\text{\text{USE}}}$ and supervised self-expertise loss $\mathcal{L}_{\text{SSE}}$. The unsupervised self-expertise loss, defined as $\mathcal{L}_{\text{USE}}=\ell_{ce}(\mathbf{p},\hat{\mathbf{t}})$, calculates the binary cross entropy loss between the logits $\mathbf{p}$ and an adjusted target $\hat{\mathbf{t}}$, where $\mathbf{p}$ is calculated based on Euclidean distance, unlike prior GCD [^53] approach that utilizes cosine similarity. [^46] introduces an adjusted target matrix $\mathbf{\hat{t}}$ to recalibrate targets based on semantic similarity between samples. Specifically, $\mathbf{\hat{t}}=\alpha\mathbf{t}+(1-\alpha)\mathbf{I}$, where $\mathbf{t}$ can be calculated using $\mathbf{t}=[\sum_{k=1}^{\lg K}\frac{\mathbbm{1}(\hat{y}_{i}^{k}\neq\hat{y}_{j}%
^{k})}{2^{k}}]$ based on pseudo label $\hat{y}_{i}^{k}$ and $\hat{y}_{j}^{k}$ from hierarchical level $k$. $\alpha$ is the hyperparameter to control the label smoothing by identity metric $\mathbf{I}$. Then, the hierarchical supervised self-expertise loss can be denoted as:

$$
\mathcal{L}_{\text{SSE}}=\frac{1}{2}\left(\sum_{k=0}^{\lg K}\frac{\mathcal{L}_%
{s}^{k}|\frac{\mathbf{d}}{2^{k}}}{2^{k}}\right),
$$

where $\mathcal{L}_{s}^{k}|\frac{\mathbf{d}}{2^{k}}$ represents the supervised representation loss applied exclusively to the segment $\frac{\mathbf{d}}{2^{k}}$ of the embedding vector $\mathbf{d}$, corresponding to each level of the hierarchy. The final representation loss is given by $\mathcal{L}_{rep}=(1-\lambda_{b})\mathcal{L}_{\text{USE}}+\lambda_{b}\mathcal{%
L}_{\text{SSE}}$. To combine SelEx with hyperbolic embeddings, we extend the hierarchical representation learning used in [^46] into the hyperbolic space, utilizing the methodology introduced in the main paper.

Following the above pace, our Hyp-SelEx utilizes hyperbolic supervised and unsupervised self-expertise, denoted as $\mathcal{L}_{\text{SSE}}^{\mathbb{H}}$ and $\mathcal{L}_{\text{USE}}^{\mathbb{H}}$, respectively. Given two randomly augmented views $\mathbf{x}_{i}$ and $\mathbf{x}_{i}^{\prime}$ for the same image in a mini-batch $B$, $\mathbf{z}_{i}$ and $\mathbf{z}_{i}^{\prime}$ represent the feature extracted from backbone network $\phi$ and projector $\rho_{r}$ of these two views in the Euclidean space, represented as $\mathbf{z}_{i}=\rho_{r}(\phi(\mathbf{x}_{i}))$. As introduced in Sec.3.4 of the main paper, we employ a hybrid of distance-based and angle-based loss functions, and hence the unsupervised self-expertise loss is represented as:

$$
\mathcal{L}_{\text{USE}}^{\mathbb{H}}=\alpha_{d}\ell_{ce}(\mathbf{p}_{\text{%
dis}},\mathbf{\hat{t}})+(1-\alpha_{d})\ell_{ce}(\mathbf{p}_{\text{ang}},%
\mathbf{\hat{t}}),
$$

where $\mathbf{p}_{\text{dis}}$ is the logit calculated based on the negative hyperbolic distance, expressed as $\mathcal{S}_{d}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{i}^{\prime%
}))$, and $\mathbf{p}_{\text{ang}}$ is the logit calculated based on the original distance metrics, expressed as $\mathcal{S}_{a}(\mathcal{M}(\mathbf{z}_{i}),\mathcal{M}(\mathbf{z}_{i}^{\prime%
}))$. Similarly, the hyperbolic supervised self-expertise loss is defined as:

$$
\textstyle\mathcal{L}_{\text{SSE}}^{\mathbb{H}}=\frac{1}{2}\left(\sum_{k=0}^{%
\lg K}\frac{\alpha_{d}(\mathcal{L}_{dis}^{k}|\frac{\mathbf{d}}{2^{k}})+(1-%
\alpha_{d})(\mathcal{L}_{ang}^{k}|\frac{\mathbf{d}}{2^{k}})}{2^{k}}\right),
$$

where $\mathcal{L}_{dis}^{k}|\frac{\mathbf{d}}{2^{k}}$ and $\mathcal{L}_{ang}^{k}|\frac{\mathbf{d}}{2^{k}}$ denote the hyperbolic supervised distance-based and angle-based losses applied exclusively to the segment $\frac{\mathbf{d}}{2^{k}}$. The final training objective of Hyp-SelEx is formulated as:

$$
\mathcal{L}_{rep}^{\mathbb{H}}=(1-\lambda_{b}^{\mathbb{H}})\mathcal{L}_{\text{%
USE}}^{\mathbb{H}}+\lambda_{b}^{\mathbb{H}}\mathcal{L}_{\text{SSE}}^{\mathbb{H%
}}.
$$

Table 4: Results with the estimated number of categories, all methods use the DINO [^7] pretrained weights.

<table><tbody><tr><th></th><td colspan="3">CUB <sup><a href="#fn:56">56</a></sup></td><td colspan="3">Stanford-Cars <sup><a href="#fn:33">33</a></sup></td></tr><tr><th>Method</th><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><th>GCD <sup><a href="#fn:53">53</a></sup></th><td>47.1</td><td>55.1</td><td>44.8</td><td>35.0</td><td>56.0</td><td>24.8</td></tr><tr><th>SimGCD <sup><a href="#fn:61">61</a></sup></th><td>61.5</td><td>66.4</td><td>59.1</td><td>49.1</td><td>65.1</td><td>41.3</td></tr><tr><th><math><semantics><mi>μ</mi> <ci>𝜇</ci> <annotation>\mu</annotation> <annotation>italic_μ</annotation></semantics></math> GCD <sup><a href="#fn:55">55</a></sup></th><td>62.0</td><td>60.3</td><td>62.8</td><td>56.3</td><td>66.8</td><td>51.1</td></tr><tr><th>SelEx <sup><a href="#fn:46">46</a></sup></th><td>72.0</td><td>72.3</td><td>71.9</td><td>58.7</td><td>75.3</td><td>50.8</td></tr><tr><th>Hyp-GCD</th><td>60.2</td><td>64.6</td><td>58.0</td><td>48.1</td><td>60.2</td><td>42.2</td></tr><tr><th>Hyp-SimGCD</th><td>64.7</td><td>66.6</td><td>63.8</td><td>60.3</td><td>73.5</td><td>53.9</td></tr><tr><th>Hyp-SelEx</th><td>79.6</td><td>75.8</td><td>81.6</td><td>62.1</td><td>76.2</td><td>55.3</td></tr></tbody></table>

## 7 More Quantitative Results

### 7.1 GCD With Unknown Category Numbers

In line with the majority of the literature [^53] [^61] [^57] [^46], our primary experiments presented in the main paper utilize the ground-truth category numbers. This section reports results based on estimated category numbers obtained from an off-the-shelf method [^53], illustrating the performance of our approach when ground-truth category numbers are unavailable. For the CUB dataset, we estimate $K=231$, while for Stanford-Cars, we estimate $K=230$. In contrast, the actual ground-truth counts are $K=200$ and $K=196$, respectively. We compare our methods with SimGCD [^61], $\mu$ GCD [^55], and GCD [^53] in Table 4. Despite a discrepancy of approximately 15% between the ground-truth and estimated category numbers for both CUB [^56] and Stanford-Cars [^33], our hyperbolic methods exhibit only a marginal decline in performance.

Table 5: Experimental results using different embedding dimensions on Hyp-GCD with DINO [^7] pre-trained backbone. Results on the CUB [^56] and Stanford-Cars [^33] datasets are reported.

<table><thead><tr><th></th><th colspan="3">CUB <sup><a href="#fn:56">56</a></sup></th><th colspan="3">Stanford-Cars <sup><a href="#fn:33">33</a></sup></th></tr><tr><th>dimension</th><th>All</th><th>Old</th><th>New</th><th>All</th><th>Old</th><th>New</th></tr></thead><tbody><tr><th>64</th><td>57.6</td><td>63.6</td><td>54.6</td><td>47.2</td><td>56.7</td><td>42.6</td></tr><tr><th>128</th><td>59.5</td><td>65.0</td><td>56.7</td><td>48.2</td><td>60.0</td><td>42.5</td></tr><tr><th>256</th><td>61.0</td><td>67.0</td><td>58.0</td><td>50.8</td><td>60.9</td><td>45.8</td></tr><tr><th>512</th><td>61.2</td><td>65.3</td><td>59.1</td><td>50.3</td><td>59.5</td><td>45.9</td></tr></tbody></table>

### 7.2 Embedding Dimension

In our framework, the parametric method Hyp-SimGCD employs the original $768$ -dimensional embeddings from the pretrained ViT-B backbone. For the non-parametric methods, Hyp-GCD and Hyp-SelEx, we project the features from the pretrained backbone into a new spherical space using an MLP projection network, followed by an exponential mapping into hyperbolic space. In the baseline methods, GCD and SelEx, the final embedding dimension is set to $65,536$. However, our empirical findings indicate that a significantly lower dimension can yield satisfactory performance with our hyperbolic method, Hyp-GCD. As shown in Tab. 5, embeddings of $256$ dimensions yield promising results for Hyp-GCD. This suggests that the intrinsic properties of hyperbolic space facilitate more expressive representations at lower dimensions (*e.g*., $256$ or $512$), effectively capturing hierarchical structures and complex relationships among data points. For Hyp-SelEx, we have chosen a dimension of $8,092$, which is also significantly lower than that of the baseline methods.

Table 6: Comparison with recent GCD methods on Herbarium19 [^49] and Oxford-Pet [^42].

<table><tbody><tr><th></th><td colspan="3">Oxford-Pet <sup><a href="#fn:42">42</a></sup></td><td colspan="3">Herbarium19 <sup><a href="#fn:49">49</a></sup></td></tr><tr><th>Method</th><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><th>ORCA <sup><a href="#fn:6">6</a></sup></th><td>-</td><td>-</td><td>-</td><td>24.6</td><td>26.5</td><td>23.7</td></tr><tr><th>GCD <sup><a href="#fn:53">53</a></sup></th><td>80.2</td><td>85.1</td><td>77.6</td><td>35.4</td><td>51.0</td><td>27.0</td></tr><tr><th>XCon <sup><a href="#fn:16">16</a></sup></th><td>86.7</td><td>91.5</td><td>84.1</td><td>-</td><td>-</td><td>-</td></tr><tr><th>OpenCon <sup><a href="#fn:48">48</a></sup></th><td>-</td><td>-</td><td>-</td><td>39.3</td><td>58.9</td><td>28.6</td></tr><tr><th>DCCL <sup><a href="#fn:43">43</a></sup></th><td>88.1</td><td>88.2</td><td>88.0</td><td>-</td><td>-</td><td>-</td></tr><tr><th>SimGCD <sup><a href="#fn:61">61</a></sup></th><td>91.7</td><td>83.6</td><td>96.0</td><td>44.0</td><td>58.0</td><td>36.4</td></tr><tr><th><math><semantics><mi>μ</mi> <ci>𝜇</ci> <annotation>\mu</annotation> <annotation>italic_μ</annotation></semantics></math> GCD <sup><a href="#fn:55">55</a></sup></th><td>-</td><td>-</td><td>-</td><td>45.8</td><td>61.9</td><td>37.2</td></tr><tr><th>InfoSieve <sup><a href="#fn:45">45</a></sup></th><td>90.7</td><td>95.2</td><td>88.4</td><td>40.3</td><td>59.0</td><td>30.2</td></tr><tr><th>SelEx <sup><a href="#fn:46">46</a></sup></th><td>92.5</td><td>91.9</td><td>92.8</td><td>39.6</td><td>54.9</td><td>31.3</td></tr><tr><th>Hyp-GCD</th><td>86.7</td><td>85.5</td><td>87.4</td><td>38.6</td><td>43.1</td><td>36.2</td></tr><tr><th>Hyp-SimGCD</th><td>92.2</td><td>85.7</td><td>95.7</td><td>45.1</td><td>60.1</td><td>36.9</td></tr><tr><th>Hyp-SelEx</th><td>92.7</td><td>91.5</td><td>93.3</td><td>40.5</td><td>49.0</td><td>36.0</td></tr></tbody></table>

### 7.3 Results on Additional Datasets

To further evaluate the proposed method, we conduct assessments on two additional fine-grained datasets: Oxford-Pet [^42] and Herbarium19 [^49]. The Oxford-Pet dataset poses a significant challenge due to its variety of cat and dog species, alongside limited data availability. In contrast, Herbarium19 is a botanical research dataset that encompasses a wide range of plant types, characterized by its long-tailed distribution and fine-grained categorization. The results of our experiments on these two datasets are summarized in Tab. 6. Our Hyp-SelEx method achieves the highest accuracy across all categories in the Oxford-Pet dataset. Furthermore, on Herbarium19, Hyp-SelEx secures the second-best performance on all three evaluation metrics.

![Refer to caption](https://arxiv.org/html/2504.06120v1/x6.png)

Figure 6: Visualization of attention maps of GCD 53 and our Hyp-GCD.

## 8 More Qualitative Results

Fig. 6 displays the attention maps of GCD [^53] and Hyp-GCD, generated from the final transformer block of the DINO backbone [^7]. These attention maps are applied across three fine-grained datasets within the SSB benchmark [^54]. In this block, a multi-head self-attention layer utilizing $12$ attention heads processes the input features, resulting in $12$ attention maps at a resolution of $14\times 14$. Following the methodology detailed in [^7], we compute the mean value of these attention maps and subsequently upsample them to the original image resolution for visualization. The results indicate that our method significantly enhances focus on semantically relevant regions within the image, effectively capturing fine-grained details that are crucial for distinguishing between categories. In contrast, the baseline approach yields more diffuse and less targeted attention maps, often insufficiently highlighting critical areas, particularly concerning unseen categories. These findings emphasize the robustness and generalization capability of our method in identifying meaningful visual regions, even for novel categories, thereby demonstrating its superiority over the baseline approach.

[^1]: Mina Ghadimi Atigh, Julian Schoep, Erman Acar, Nanne Van Noord, and Pascal Mettes. Hyperbolic image segmentation. In *CVPR*, 2022.

[^2]: Anwesha Banerjee, Liyana Sahir Kallooriyakath, and Soma Biswas. Amend: Adaptive margin and expanded neighborhood for efficient generalized category discovery. In *WACV*, 2024.

[^3]: Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. Fully hyperbolic convolutional neural networks for computer vision. In *ICLR*, 2024.

[^4]: Gary Becigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods. In *ICLR*, 2019.

[^5]: James W Cannon, William J Floyd, Richard Kenyon, Walter R Parry, et al. Hyperbolic geometry. *Flavors of geometry*, 31(59-115):2, 1997.

[^6]: Kaidi Cao, Maria Brbic, and Jure Leskovec. Open-world semi-supervised learning. In *ICLR*, 2022.

[^7]: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *ICCV*, 2021.

[^8]: Fernando Julio Cendra, Bingchen Zhao, and Kai Han. Promptccd: Learning gaussian mixture prompt pool for continual category discovery. In *ECCV*, 2024.

[^9]: Ines Chami, Rex Ying, Christopher Ré, and Jure Leskovec. Hyperbolic graph convolutional neural networks. In *NeurIPS*, 2019.

[^10]: Weize Chen, Xu Han, Yankai Lin, Hexu Zhao, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Fully hyperbolic neural networks. In *ACL*, 2022.

[^11]: Florent Chiaroni, Jose Dolz, Ziko Imtiaz Masud, Amar Mitiche, and Ismail Ben Ayed. Parametric information maximization for generalized category discovery. In *ICCV*, 2023.

[^12]: Sua Choi, Dahyun Kang, and Minsu Cho. Contrastive mean-shift learning for generalized category discovery. In *CVPR*, 2024.

[^13]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *CVPR*, 2009.

[^14]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.

[^15]: Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, and Ivan Oseledets. Hyperbolic vision transformers: Combining improvements in metric learning. In *CVPR*, 2022.

[^16]: Yixin Fei, Zhongkai Zhao, Siwei Yang, and Bingchen Zhao. Xcon: Learning with experts for fine-grained category discovery. In *BMVC*, 2022.

[^17]: Enrico Fini, Enver Sangineto, Stéphane Lathuiliere, Zhun Zhong, Moin Nabi, and Elisa Ricci. A unified objective for novel class discovery. In *ICCV*, 2021.

[^18]: Luca Franco, Paolo Mandica, Bharti Munjal, and Fabio Galasso. Hyperbolic self-paced learning for self-supervised skeleton-based action representations. In *ICLR*, 2023.

[^19]: Octavian Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic neural networks. In *NeurIPS*, 2018.

[^20]: Zhi Gao, Yuwei Wu, Yunde Jia, and Mehrtash Harandi. Curvature generation in curved spaces for few-shot learning. In *ICCV*, 2021.

[^21]: Mina GhadimiAtigh, Julian Schoep, Erman Acar, Nanne van Noord, and Pascal Mettes. Hyperbolic image segmentation. In *CVPR*, 2022.

[^22]: Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, and Nando de Freitas. Hyperbolic attention networks. In *ICLR*, 2019.

[^23]: Yunhui Guo, Xudong Wang, Yubei Chen, and Stella X. Yu. Clipped hyperbolic classifiers are super-hyperbolic classifiers. In *CVPR*, 2022.

[^24]: Kai Han, Andrea Vedaldi, and Andrew Zisserman. Learning to discover novel visual categories via deep transfer clustering. In *ICCV*, 2019.

[^25]: Kai Han, Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Andrea Vedaldi, and Andrew Zisserman. Automatically discovering and learning new visual categories with ranking statistics. In *ICLR*, 2020.

[^26]: Kai Han, Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Andrea Vedaldi, and Andrew Zisserman. Autonovel: Automatically discovering and learning novel visual categories. *IEEE TPAMI*, 2021.

[^27]: Shaozhe Hao, Kai Han, and Kwan-Yee K Wong. Cipr: An efficient framework with cross-instance positive relations for generalized category discovery. *TMLR*, 2024.

[^28]: Xuhui Jia, Kai Han, Yukun Zhu, and Bradley Green. Joint representation learning and novel category discovery on single-and multi-modal data. In *ICCV*, 2021.

[^29]: Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. Image retrieval using scene graphs. In *CVPR*, 2015.

[^30]: KJ Joseph, Sujoy Paul, Gaurav Aggarwal, Soma Biswas, Piyush Rai, Kai Han, and Vineeth N Balasubramanian. Novel class discovery without forgetting. In *ECCV*, 2022.

[^31]: Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In *CVPR*, 2020.

[^32]: Fanjie Kong, Yanbei Chen, Jiarui Cai, and Davide Modolo. Hyperbolic learning with synthetic captions for open-world detection. In *CVPR*, 2024.

[^33]: Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In *ICCV workshop*, 2013.

[^34]: Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[^35]: Huimin Li, Zhentao Chen, Yunhao Xu, and Junlin Hu. Hyperbolic anomaly detection. In *CVPR*, 2024.

[^36]: Qi Liu, Maximilian Nickel, and Douwe Kiela. Hyperbolic graph neural networks. In *NeurIPS*, 2019.

[^37]: Yuanpei Liu and Kai Han. Debgcd: Debiased learning with distribution guidance for generalized category discovery. In *ICLR*, 2025.

[^38]: Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. *arXiv preprint arXiv:1306.5151*, 2013.

[^39]: Maximilian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representations. In *NeurIPS*, 2017.

[^40]: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.

[^41]: Jona Otholt, Christoph Meinel, and Haojin Yang. Guided cluster aggregation: A hierarchical approach to generalized category discovery. In *WACV*, pages 2618–2627, 2024.

[^42]: Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In *CVPR*, 2012.

[^43]: Nan Pu, Zhun Zhong, and Nicu Sebe. Dynamic conceptional contrastive learning for generalized category discovery. In *CVPR*, 2023.

[^44]: Nan Pu, Wenjing Li, Xingyuan Ji, Yalan Qin, Nicu Sebe, and Zhun Zhong. Federated generalized category discovery. In *CVPR*, 2024.

[^45]: Sarah Rastegar, Hazel Doughty, and Cees Snoek. Learn to categorize or categorize to learn? self-coding for generalized category discovery. In *NeurIPS*, 2023.

[^46]: Sarah Rastegar, Mohammadreza Salehi, Yuki M Asano, Hazel Doughty, and Cees G M Snoek. Selex: Self-expertise in fine-grained generalized category discovery. In *ECCV*, 2024.

[^47]: Ryohei Shimizu, Yusuke Mukuta, and Tatsuya Harada. Hyperbolic neural networks++. In *ICLR*, 2021.

[^48]: Yiyou Sun and Yixuan Li. Opencon: Open-world contrastive learning. *TMLR*, 2022.

[^49]: Kiat Chuan Tan, Yulong Liu, Barbara Ambrose, Melissa Tulig, and Serge Belongie. The herbarium challenge 2019 dataset. *arXiv preprint arXiv:1906.05372*, 2019.

[^50]: Abraham Albert Ungar. A gyrovector space approach to hyperbolic geometry. *Synthesis Lectures on Mathematics and Statistics*, 1(1):1–194, 2008.

[^51]: Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. *JMLR*, 2008.

[^52]: Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Open-set recognition: A good closed-set classifier is all you need? In *ICLR*, 2022a.

[^53]: Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Generalized category discovery. In *CVPR*, 2022b.

[^54]: Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. The semantic shift benchmark. In *ICML workshop*, 2022c.

[^55]: Sagar Vaze, Andrea Vedaldi, and Andrew Zisserman. No representation rules them all in category discovery. In *NeurIPS*, 2023.

[^56]: Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

[^57]: Hongjun Wang, Sagar Vaze, and Kai Han. Sptnet: An efficient alternative framework for generalized category discovery with spatial prompt tuning. In *ICLR*, 2024a.

[^58]: Hongjun Wang, Sagar Vaze, and Kai Han. Hilo: A learning framework for generalized category discovery robust to domain shifts. In *ICLR*, 2025.

[^59]: Yu Wang, Zhun Zhong, Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, and Jie Chen. Discover and align taxonomic context priors for open-world semi-supervised learning. In *NeurIPS*, 2024b.

[^60]: Simon Weber, Bar Zöngür, Nikita Araslanov, and Daniel Cremers. Flattening the parent bias: Hierarchical semantic segmentation in the poincaré ball. In *CVPR*, 2024.

[^61]: Xin Wen, Bingchen Zhao, and Xiaojuan Qi. Parametric classification for generalized category discovery: A baseline study. In *ICCV*, 2023.

[^62]: Zhenzhen Weng, Mehmet Giray Ogut, Shai Limonchik, and Serena Yeung. Unsupervised discovery of the long-tail in instance segmentation using hierarchical self-supervision. In *CVPR*, 2021.

[^63]: Sheng Zhang, Salman Khan, Zhiqiang Shen, Muzammal Naseer, Guangyi Chen, and Fahad Shahbaz Khan. Promptcal: Contrastive affinity learning via auxiliary prompts for generalized novel category discovery. In *CVPR*, 2023.

[^64]: Xinwei Zhang, Jianwen Jiang, Yutong Feng, Zhi-fan Wu, Xibin Zhao, Hai Wan, Mingqian Tang, Rong Jin, and Yue Gao. Grow and merge: a unified framework for continuous categories discovery. In *NeurIPS*, 2022.

[^65]: Bingchen Zhao and Kai Han. Novel visual category discovery with dual ranking statistics and mutual knowledge distillation. In *NeurIPS*, 2021.

[^66]: Bingchen Zhao, Xin Wen, and Kai Han. Learning semi-supervised gaussian mixture models for generalized category discovery. In *ICCV*, 2023.

[^67]: Zhun Zhong, Enrico Fini, Subhankar Roy, Zhiming Luo, Elisa Ricci, and Nicu Sebe. Neighborhood contrastive learning for novel class discovery. In *CVPR*, 2021.