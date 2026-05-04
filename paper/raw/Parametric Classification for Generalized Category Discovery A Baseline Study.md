---
title: "Parametric Classification for Generalized Category Discovery: A Baseline Study"
source: "https://arxiv.org/html/2211.11727v4"
author:
published:
created: 2026-05-04
description:
tags:
  - "clippings"
---
Xin Wen <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math></sup> <sup>1</sup>   Bingchen Zhao <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="2"><semantics><mn>2</mn> <cn type="integer">2</cn> <annotation>2</annotation> <annotation>2</annotation></semantics></math></sup> <sup>1</sup>   Xiaojuan Qi <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math></sup>  
<sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math></sup> The University of Hong Kong   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="2"><semantics><mn>2</mn> <cn type="integer">2</cn> <annotation>2</annotation> <annotation>2</annotation></semantics></math></sup> University of Edinburgh  
{wenxin,xjqi}@eee.hku.hk   bingchen.zhao@ed.ac.uk

###### Abstract

Generalized Category Discovery (GCD) aims to discover novel categories in unlabelled datasets using knowledge learned from labelled samples. Previous studies argued that parametric classifiers are prone to overfitting to seen categories, and endorsed using a non-parametric classifier formed with semi-supervised $k$ -means. However, in this study, we investigate the failure of parametric classifiers, verify the effectiveness of previous design choices when high-quality supervision is available, and identify unreliable pseudo-labels as a key problem. We demonstrate that two prediction biases exist: the classifier tends to predict seen classes more often, and produces an imbalanced distribution across seen and novel categories. Based on these findings, we propose a simple yet effective parametric classification method that benefits from entropy regularisation, achieves state-of-the-art performance on multiple GCD benchmarks and shows strong robustness to unknown class numbers. We hope the investigation and proposed simple framework can serve as a strong baseline to facilitate future studies in this field. Our code is available at: [https://github.com/CVMI-Lab/SimGCD](https://github.com/CVMI-Lab/SimGCD).

![[Uncaptioned image]](https://arxiv.org/html/2211.11727v4/x1.png)

Figure 1: Left: building blocks for representation learning or classifier learning; Right: overall abstraction of current works, where ‘ $\rightarrow$ ’ separates different stages of the method. Our work builds on GCD [^43], and jointly trains a parametric classifier.

## 1 Introduction

With large-scale labelled datasets, deep learning methods can surpass humans in recognising images [^25]. However, it is not always possible to collect large-scale human annotations for training deep learning models. Therefore, there is a rich body of recognition models that focus on learning with a large number of unlabelled data. Among them, semi-supervised learning (SSL) [^33] [^5] [^38] is regarded as a promising approach, yet with the assumption that labelled instances are provided for each of the categories the model needs to classify. Generalized category discovery (GCD) [^43] is recently formalised to relax this assumption by assuming the unlabelled data can also contain similar yet distinct categories from the labelled data. The goal of GCD is to learn a model that is able to classify the already-seen categories in the labelled data, and more importantly, jointly discover the new categories in the unlabelled data and make correct classifications. Developing a strong method for this problem could help us better utilise the easily available large-scale unlabelled datasets.

Previous works [^43] [^22] [^17] [^6] approach this problem from two perspectives: learning generic feature representations to facilitate the discovery of novel categories, and generating pseudo clusters/labels for unlabelled data to guide the learning of a classifier. The former is often achieved by using self-supervised learning methods [^22] [^52] [^18] [^24] [^9] [^54] to improve the generalization ability of features to novel categories. For constructing the classifier, earlier works [^22] [^52] [^57] [^6] [^17] adopt a parametric approach that builds a learnable classifier on top of the extracted features. The classifier is jointly optimised with the backbone using labelled data and pseudo-labelled data.

However, recent research shows [^43] [^16] that parametric classifiers are prone to overfit to seen categories (see Fig. 2) and thus promote using a non-parametric classifier such as $k$ -means clustering. Albeit obtaining promising results, the non-parametric classifiers suffer from heavy computation costs on large-scale datasets due to quadratic complexity of the clustering algorithm. Besides, unlike a learnable parametric classifier, the non-parametric method loses the ability to jointly optimise the separating hyperplane of all categories in a learnable manner, potentially being sub-optimal.

This motivates us to revisit the reason that makes previous parametric classifiers fail to recognise novel classes. In a series of investigations (Sec. 3) from the view of supervision quality, we verify the effectiveness of prior design choices in feature representations and training paradigms when strong supervision is available, and conclude that the key to previous parametric classifiers’ degraded performance is unreliable pseudo labels. By diagnosing the statistics of its predictions, we identify severe prediction biases within the model, *i.e*., the bias towards predicting more ‘Old’ classes than ‘New’ classes (Fig. 5) and the bias of producing imbalanced pseudo-labels across all classes (Fig. 6).

Based on these findings, we thus present a simple parametric classification baseline for generalized category discovery (see Parametric Classification for Generalized Category Discovery: A Baseline Study and 7). The representation learning objective follows GCD [^43], and the classification objective is simply cross-entropy for labelled samples and self-distillation [^9] [^3] for unlabelled samples. Besides, an entropy regularisation term is also adopted to overcome biased predictions by enforcing the model to predict more uniformly distributed labels across all possible categories. Empirically, we indeed observe that our method produces more balanced pseudo-labels (Figs. 9 and 10) and achieves a large performance gain on multiple GCD benchmarks (Tabs. 2, 3 and 4), indicating that the two types of biases we identified are the core reason why the parametric-classifier-based approach performs poorly for GCD. Additionally, we observe that the entropy regulariser could also be used to enforce robustness towards an unknown number of categories (Figs. 11 and 12), this could further ease the deployment of parametric classifiers for GCD in real-world scenarios.

Our contributions are summarised as follows: (1) We revisit the design choices of parametric classification and conclude the key factors that make it fail for GCD. (2) Based on the analysis, we propose a simple yet effective parametric classification method. (3) Our method achieves SOTA on multiple popular GCD benchmarks, challenging the recent promotion of non-parametric classification for this task.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x2.png)

Figure 2: Performance overview. Prior parametric classification method (UNO+ 17 ) shows highly degraded performance in ‘New’ classes. The non-parametric classification work (GCD 43 ) performs better, but at the sacrifice of ‘Old’ class and high inference cost. Our method shows that parametric classification can work well on both metrics.

## 2 Related Works

#### Semi-Supervised Learning

(SSL) has been an important research topic where a number of methods have been proposed [^5] [^38] [^41]. SSL assumes that the labelled instances are available for all possible categories in the unlabelled dataset; the objective is to learn a model to perform classification using both the labelled samples as well as the large-scale available unlabelled data. One of the most effective methods for SSL is the consistency-based method, where the model is forced to learn consistent representations of two different augmentations of the same image [^38] [^5] [^41]. Furthermore, it is also shown that self-supervised representation learning is helpful for the task of SSL [^51] [^34] as it can provide a strong representation for the task.

#### Open-Set Semi-Supervised Learning

considers the case where the unlabelled data may contain outlier data points that do not belong to any of the categories in the labelled training set. The goal is to learn a classifier for the labelled categories from a noisy unlabelled dataset [^50] [^37] [^11] [^20]. As this problem only focuses on the performance of the labelled categories, the outlier from novel categories are simply rejected and no further classification is needed.

#### Generalized Category Discovery

(GCD) is a relatively new problem recently formalised in Vaze *et al*. [^43], and is also studied in a parallel line of work termed open-world semi-supervised learning [^6] [^39]. Different from the common assumption of SSL [^33], GCD does not assume the unlabelled dataset comes from the same class set as the labelled dataset, posing a greater challenge for designing an effective model. GCD can be seen as a natural extension of the novel category discovery (NCD) problem [^23] where it is assumed that the unlabelled dataset and the labelled dataset do not have any class overlap, thus baselines for NCD [^22] [^52] [^57] [^56] [^17] can be adopted for the GCD problem by extending the classification head to have more outputs [^43]. The incremental setting of GCD is also explored [^53] [^36]. It is pointed out in [^43] that a non-parametric classifier formed using semi-supervised $k$ -means can outperform strong parametric classification baselines from NCD [^22] [^17] because it can alleviate the overfitting to seen categories in the labelled set. In this paper, we revisit this claim and show that parametric classifiers can reach stronger performance than non-parametric classifiers.

#### Deep Clustering

aims at learning a set of semantic prototypes from unlabelled images with deep neural networks. Considering that no label information is available, the focus is on how to obtain reliable pseudo-labels. While early attempts rely on hard labels produced by $k$ -means [^7], there has been a shift towards soft labels produced by optimal transport [^2] [^8], and more recently sharpened predictions from an exponential moving average-updated teacher model [^9] [^3]. Deep clustering has shown strong potential for unsupervised representation learning [^7] [^2] [^8] [^9] [^3], unsupervised semantic segmentation [^12] [^49], semi-supervised learning [^4], and novel category discovery [^17]. In this work, we study the techniques that make strong parametric classifiers for GCD with inspirations from deep clustering.

## 3 On the Failure of Parametric Classification

In order to explore the reason that makes previous parametric classifiers fail to recognise ‘New’ classes for generalized category discovery, this section presents preliminary studies to reveal the role of two major components: representation learning (Sec. 3.2) and pseudo-label quality on unseen classes (Sec. 3.3). These have led to conflicting choices of previous works, but why? We show a unified viewpoint (Figs. 3 and 4), and emphasise that taking pseudo-label quality into account is important for selecting the suitable design choice. This then led to our diagnosis of what makes the degenerated pseudo-labels (Sec. 3.4), and motivated our de-biased pseudo-labelling strategy.

### 3.1 Investigation Setting

#### Generalized category discovery.

Given an unlabelled dataset $\mathcal{D}^{u}=\mathopen{}\mathclose{{}\left\{(\boldsymbol{x}_{i}^{u},{y}_{i}%
^{u})}\right\}\in\mathcal{X}\times\mathcal{Y}_{u}$ where $\mathcal{Y}_{u}$ is the label space of the unlabelled samples, the goal of GCD is to learn a model to categorise the samples in $\mathcal{D}^{u}$ using the knowledge from a labelled dataset $\mathcal{D}^{l}=\mathopen{}\mathclose{{}\left\{(\boldsymbol{x}_{i}^{l},{y}_{i}%
^{l})}\right\}\in\mathcal{X}\times\mathcal{Y}_{l}$ where $\mathcal{Y}_{l}$ is the label space of labelled samples and $\mathcal{Y}_{l}\subset\mathcal{Y}_{u}$. We denote the number of categories in $\mathcal{Y}_{u}$ as $K_{u}=|\mathcal{Y}_{u}|$, it is common to assume the number of categories is known a-priori [^22] [^52] [^57] [^17], or can be estimated using off-the-shelf methods [^23] [^43].

#### Representation learning.

For representation learning, we follow GCD [^43], which applies supervised contrastive learning [^27] on labelled samples, and self-supervised contrastive learning [^10] on all samples (detailed in Sec. 4.1).

#### Classifier.

We follow UNO [^17] to adopt a prototypical classifier. Take $f(\boldsymbol{x})$ as the feature vector of an image $\boldsymbol{x}$ extracted using from the backbone $f$, the procedure for producing logits is $\boldsymbol{l}=\frac{1}{\tau}(\boldsymbol{w}/||\boldsymbol{w}||)^{\top}(f(%
\boldsymbol{x})/||f(\boldsymbol{x})||)$. Here $\tau$ is the temperature value that scales up the norm of $\boldsymbol{l}$ and facilitates optimisation of the cross-entropy loss [^45].

#### Training settings.

We train with varying supervision qualities. The minimal supervision setting utilises only the labels in $\mathcal{D}^{l}$, while the oracle supervision setting assumes all samples are labelled (both $\mathcal{D}^{l}$ and $\mathcal{D}^{u}$). Besides, we study two practical settings that adopt pseudo labels for unlabelled samples in $\mathcal{D}^{u}$: self-label that predicts pseudo-labels with the Sinkhorn Knopp algorithm following [^17], and self-distil, which depicts another pseudo-labelling strategy as in Fig. 7 and will be introduced in detail in Sec. 4.2. For all settings, we only employ a cross-entropy loss on the (pseudo-)labelled samples on hand for classification. Note that unless otherwise stated, this is done on decoupled features, thus representation learning is unaffected.

### 3.2 Which Representation to Build Your Classifier?

#### Motivation.

Following the trend of deep clustering that focuses on self-supervised representation learning [^8], previous parametric classification work UNO [^17] fed the classifier with representations taken from the projector. While in GCD [^43], significantly stronger performance is achieved with a non-parametric classifier built upon representations taken from the backbone. We revisit this choice as follows.

#### Setting.

Consider $f$ as the feature backbone, and $g$ is a multi-layer perceptron (MLP) projection head. Given an input image $\boldsymbol{x}_{i}$, the representation from the backbone can be written as $f(\boldsymbol{x}_{i})$, and that from the projector is $g(f(\boldsymbol{x}_{i}))$.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x3.png)

Figure 3: Results with different representations. We build the classifier on post-backbone or post-projector representations, and train with varying supervision quality. Results on ‘Old’ class consistently benefit from the post-backbone representations regardless of the supervision quality, while unleashing its potential on ‘New’ class requires stronger pseudo labels.

#### Result & discussion.

As in Fig. 3, the post-backbone feature space has a clearly higher upper bound for learning prototypical classifiers than the post-projector feature space. Using a projector in self-supervised learning lets the projector focus on solving pretext tasks and allows the backbone to keep as much information as possible (which facilitates downstream tasks) [^13]. But when good classification performance is all you need, our results suggest that the classification objective should build on post-backbone representations directly. The features post the projector might focus more on solving the pretext task and not be necessarily useful for the classification objective. Note that high-quality pseudo labels are necessary to unleash the post-backbone representations’ potential to recognise novel categories.

### 3.3 Decoupled or Joint Representation Learning?

#### Motivation.

Previous parametric classification methods, *e.g*., UNO [^17], commonly tune the representations jointly with the classification objective. On the contrary, in the two-stage non-parametric method GCD [^43] where the performance in ‘New’ classes is notably higher, classification/clustering is fully decoupled from representation learning, and the representations can be viewed as unaltered by classification. In this part, we study whether the joint learning strategy contributes to previous parametric classifiers’ degraded performance in recognising ‘New’ classes.

#### Setting.

Consider $f(\boldsymbol{x})$ as the representation fed to the classifier, decoupled training, as the previous settings adopted, indicates $f(\boldsymbol{x})$ is decoupled when computing the logits $\boldsymbol{l}$, thus the classification objective won’t supervise representation learning. While for joint training, the representations are jointly optimised by classification.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x4.png)

Figure 4: Results with different training paradigms. Decouple denotes the classifier adopts decoupled features, while joint indicates the classification objective can affect representation learning. Joint training is helpful when high-quality supervision is available, otherwise, it could lead to degraded representations.

#### Result & discussion.

The results are illustrated in Fig. 4. When adopting the self-labelling strategy, there is a sharp drop in ‘Old’ class performance on both datasets, while for the ‘New’ classes, it can improve by 13 points on CIFAR100, and drop by a small margin on CUB. In contrast, when a stronger pseudo-labelling strategy (self-distillation) or even oracle labels are utilised, we observe consistent gains from joint training. This means that the joint training strategy does not necessarily result in UNO [^17] ’s low performance in ‘New’ classes; on the contrary, it can even boost ‘New’ class performance by a notable margin. Our overall explanation is that UNO’s framework could not make reliable pseudo-labels, thus restricting its ability to benefit from joint training. The joint training strategy is not to blame and is, in fact, helpful. When switching to a more advanced pseudo-labelling paradigm that produces higher-quality pseudo-labels, the help from joint training can be even more significant.

### 3.4 The Devil Is in the Biased Predictions

#### Motivation.

In Secs. 3.2 and 3.3, we verified the effectiveness of two design choices when high-quality pseudo labels are available, and concluded the key to previous work’s degraded performance is unreliable pseudo labels. We then further diagnose the statistics of its predictions as follows.

#### Setting.

We categorise the model’s errors into four types: “True Old”, “False New”, “False Old”, and “True New” according to the relationship between predicted and ground-truth class. *E.g*., “True New” refers to predicting a ‘New’ class sample to another ‘New’ class, while “False Old” indicates predicting a ‘New’ class sample as some ‘Old’ class.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x5.png)

Figure 5: Prediction bias between ‘Old’/‘New’ classes. We simplify the results to binary classification and categorise errors in ‘All’ ACC into four types. Both works, especially UNO+, are prone to make “False Old” predictions, and many samples corresponding to ‘New’ classes are misclassified as an ‘Old’ class.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x6.png)

Figure 6: Prediction bias across ‘Old’/‘New’ classes. We show the per-class prediction distributions. Both works, especially UNO+, are prone to make biased predictions. Across all classes, the predictions are unexpectedly biased towards the head classes.

#### Result & discussion.

We observe two types of prediction bias. In Fig. 5, both works, especially UNO+ [^17], are prone to make “False Old” predictions. In other words, their predictions are biased towards ‘Old’ classes. Besides, the “True New” errors are also notable, indicating that misclassification within ‘New’ classes is also common. We then depict the predictions’ overall distribution across ‘Old’/‘New’ classes in Fig. 6, and both works show highly biased predictions. This double-bias phenomenon then motivated the prediction entropy regularisation design in our method.

## 4 Method

In this section, we present the whole picture of this simple yet effective method (see Fig. 7), a one-stage framework that builds on GCD [^43], and jointly trains a parametric classifier with self-distillation and entropy regularisation. And in Sec. 5.3, we discuss the step-by-step changes that lead a simple baseline to our solution.

### 4.1 Representation Learning

Our representation learning objective follows GCD [^43], which is supervised contrastive learning [^27] on labelled samples, and self-supervised contrastive learning [^10] on all samples. Formally, given two views (random augmentations) $\boldsymbol{x}_{i}$, and $\boldsymbol{x}_{i}^{\prime}$ of the same image in a mini-batch $B$, the self-supervised contrastive loss is written as:

$$
\mathcal{L}^{u}_{\text{rep}}=\frac{1}{|B|}\sum_{i\in B}-\log\frac{\exp%
\mathopen{}\mathclose{{}\left(\boldsymbol{z}_{i}^{\top}\boldsymbol{z}_{i}^{%
\prime}/\tau_{u}}\right)}{\sum_{i}^{i\neq n}\exp\mathopen{}\mathclose{{}\left(%
\boldsymbol{z}_{i}^{\top}\boldsymbol{z}_{n}^{\prime}/\tau_{u}}\right)}\,,
$$

where the feature $\boldsymbol{z}_{i}=g\mathopen{}\mathclose{{}\left(f\mathopen{}\mathclose{{}%
\left(\boldsymbol{x}_{i}}\right)}\right)$ and is $\ell_{2}$ -normalised, $f,g$ denote the backbone and the projection head, and $\tau_{u}$ is a temperature value. The supervised contrastive loss is similar, and the major difference is that positive samples are matched by their labels, formally written as:

$$
\mathcal{L}^{s}_{\text{rep}}=\frac{1}{|B^{l}|}\sum_{i\in B^{l}}\frac{1}{|%
\mathcal{N}_{i}|}\sum_{q\in\mathcal{N}_{i}}-\log\frac{\exp\mathopen{}%
\mathclose{{}\left(\boldsymbol{z}_{i}^{\top}\boldsymbol{z}_{q}^{\prime}/\tau_{%
c}}\right)}{\sum_{i}^{i\neq n}\exp\mathopen{}\mathclose{{}\left(\boldsymbol{z}%
_{i}^{\top}\boldsymbol{z}_{n}^{\prime}/\tau_{c}}\right)}\,,
$$

where $\mathcal{N}_{i}$ indexes all other images in the same batch that hold the same label as $\boldsymbol{x}_{i}$. The overall representation learning loss is balanced with $\lambda$: $\mathcal{L}_{\text{rep}}=(1-\lambda)\mathcal{L}^{u}_{\text{rep}}+\lambda%
\mathcal{L}^{s}_{\text{rep}}$, where $B^{l}$ corresponds to the labelled subset of $B$.

### 4.2 Parametric Classification

Our parametric classification paradigm follows the self-distillation [^9] [^3] fashion. Formally, with $K=|\mathcal{Y}_{l}\cup\mathcal{Y}_{u}|$ denoting the total number of categories, we randomly initialise a set of prototypes $\mathcal{C}=\{\boldsymbol{c}_{1},\dots,\boldsymbol{c}_{K}\}$, each standing for one category. During training, we calculate the soft label for each augmented view $\boldsymbol{x}_{i}$ by softmax on cosine similarity between the hidden feature $\boldsymbol{h}_{i}=f(\boldsymbol{x}_{i})$ and the prototypes $\mathcal{C}$ scaled by $1/\tau_{s}$:

$$
\boldsymbol{p}_{i}^{(k)}=\frac{\exp\mathopen{}\mathclose{{}\left(\frac{1}{\tau%
_{s}}(\boldsymbol{h}_{i}/||\boldsymbol{h}_{i}||_{2})^{\top}(\boldsymbol{c}_{k}%
/||\boldsymbol{c}_{k}||_{2})}\right)}{\sum_{k^{\prime}}\exp\mathopen{}%
\mathclose{{}\left(\frac{1}{\tau_{s}}(\boldsymbol{h}_{i}/||\boldsymbol{h}_{i}|%
|_{2})^{\top}(\boldsymbol{c}_{k^{\prime}}/||\boldsymbol{c}_{k^{\prime}}||_{2})%
}\right)}\,,
$$

and the soft pseudo-label $\boldsymbol{q}_{i}^{\prime}$ is produced by another view $\boldsymbol{x}_{i}$ with a sharper temperature $\tau_{t}$ in a similar fashion. The classification objectives are then simply cross-entropy loss $\ell(\boldsymbol{q}^{\prime},\boldsymbol{p})=-\sum_{k}{\boldsymbol{q}^{\prime(%
k)}}\log{\boldsymbol{p}}^{(k)}$ between the predictions and pseudo-labels or ground-truth labels:

$$
\mathcal{L}^{u}_{\text{cls}}=\frac{1}{|B|}\sum_{i\in B}\ell(\boldsymbol{q}_{i}%
^{\prime},\boldsymbol{p}_{i})-\varepsilon H(\overline{\boldsymbol{p}}),%
\mathcal{L}^{s}_{\text{cls}}=\frac{1}{|B^{l}|}\sum_{i\in B^{l}}\ell(%
\boldsymbol{y}_{i},\boldsymbol{p}_{i}),
$$

where $\boldsymbol{y}_{i}$ denote the one-hot label of $\boldsymbol{x}_{i}$. We also adopt a mean-entropy maximisation regulariser [^3] for the unsupervised objective. Here $\overline{\boldsymbol{p}}=\frac{1}{2|B|}\sum_{i\in B}\mathopen{}\mathclose{{}%
\left(\boldsymbol{p}_{i}+\boldsymbol{p}_{i}^{\prime}}\right)$ denotes the mean prediction of a batch, and the entropy $H(\overline{\boldsymbol{p}})=-\sum_{k}\overline{\boldsymbol{p}}^{(k)}\log%
\overline{\boldsymbol{p}}^{(k)}$. Then the classification objective is $\mathcal{L}_{\text{cls}}=(1-\lambda)\mathcal{L}^{u}_{\text{cls}}+\lambda%
\mathcal{L}^{s}_{\text{cls}}$, and the overall objective is simply $\mathcal{L}_{\text{rep}}+\mathcal{L}_{\text{cls}}$.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x7.png)

Figure 7: The overall framework of our method. For unlabelled samples, the pseudo-labels are from sharpened predictions of another random augmented view. And for labelled samples, we simply adopt the ground truth. Details for representation learning and the mean-entropy-maximisation regulariser are omitted for simplicity, and please refer to the text. (Also see Parametric Classification for Generalized Category Discovery: A Baseline Study for a high-level comparison with previous works)

#### Discussions.

Please note that this work doesn’t aim to promote new methods but to examine existing solutions, provide insights into their failures and build a simple yet strong baseline solution. The paradigm of producing pseudo-labels from sharpened predictions of another augmented view appears to resemble consistency-based methods [^38] [^5] [^41] in the SSL community. However, despite differences in augmentation strategies and soft/hard pseudo-labels, our approach jointly performs category discovery and self-training style learning, while the SSL methods purely focus on bootstrapping itself with unlabelled data, and does not discover novel categories. Besides, entropy regularisation is also explored in deep clustering to avoid trivial solution [^3]. In contrast, our method shows its help in overcoming the prediction bias between and within seen/novel classes (Figs. 9 and 10), and enforcing robustness to unknown numbers of categories (Fig. 11).

## 5 Experiments

### 5.1 Experimental Setup

#### Datasets.

We validate the effectiveness of our method on the generic image recognition benchmark (including CIFAR10/100 [^29] and ImageNet-100 [^14]), the recently proposed Semantic Shift Benchmark [^44] (SSB, including CUB [^48], Stanford Cars [^28], and FGVC-Aircraft [^31]), and the harder Herbarium 19 [^40] and ImageNet-1K [^14]. For each dataset, we follow [^43] to sample a subset of all classes as the labelled (‘Old’) classes $\mathcal{Y}_{l}$; 50% of the images from these labelled classes are used to construct $\mathcal{D}^{l}$, and the remaining images are regarded as the unlabelled data $\mathcal{D}^{u}$. See Tab. 1 for statistics of the datasets we evaluate on.

<table><tbody><tr><td></td><td></td><td colspan="2">Labelled</td><td colspan="2">Unlabelled</td></tr><tr><td>Dataset</td><td>Balance</td><td>#Image</td><td>#Class</td><td>#Image</td><td>#Class</td></tr><tr><td>CIFAR10 <sup><a href="#fn:29">29</a></sup></td><td>✓</td><td>12.5K</td><td>5</td><td>37.5K</td><td>10</td></tr><tr><td>CIFAR100 <sup><a href="#fn:29">29</a></sup></td><td>✓</td><td>20.0K</td><td>80</td><td>30.0K</td><td>100</td></tr><tr><td>ImageNet-100 <sup><a href="#fn:14">14</a></sup></td><td>✓</td><td>31.9K</td><td>50</td><td>95.3K</td><td>100</td></tr><tr><td>CUB <sup><a href="#fn:48">48</a></sup></td><td>✓</td><td>1.5K</td><td>100</td><td>4.5K</td><td>200</td></tr><tr><td>Stanford Cars <sup><a href="#fn:28">28</a></sup></td><td>✓</td><td>2.0K</td><td>98</td><td>6.1K</td><td>196</td></tr><tr><td>FGVC-Aircraft <sup><a href="#fn:31">31</a></sup></td><td>✓</td><td>1.7K</td><td>50</td><td>5.0K</td><td>100</td></tr><tr><td>Herbarium 19 <sup><a href="#fn:40">40</a></sup></td><td>✗</td><td>8.9K</td><td>341</td><td>25.4K</td><td>683</td></tr><tr><td>ImageNet-1K <sup><a href="#fn:14">14</a></sup></td><td>✓</td><td>321K</td><td>500</td><td>960K</td><td>1000</td></tr></tbody></table>

Table 1: Statistics of the datasets we evaluate on.

#### Evaluation protocol.

We evaluate the model performance with clustering accuracy (ACC) following standard practice [^43]. During evaluation, given the ground truth $y^{*}$ and the predicted labels $\hat{y}$, the ACC is calculated as $\text{ACC}=\frac{1}{M}\sum_{i=1}^{M}\mathds{1}(y^{*}_{i}=p(\hat{y}_{i}))$ where $M=|\mathcal{D}^{u}|$, and $p$ is the optimal permutation that matches the predicted cluster assignments to the ground truth class labels.

#### Implementation details.

Following GCD [^43], we train all methods with a ViT-B/16 backbone [^15] pre-trained with DINO [^9]. We use the output of \[CLS\] token with a dimension of 768 as the feature for an image, and only fine-tune the last block of the backbone. We train with a batch size of 128 for 200 epochs with an initial learning rate of 0.1 decayed with a cosine schedule on each dataset. Aligning with [^43], the balancing factor $\lambda$ is set to 0.35, and the temperature values $\tau_{u}$, $\tau_{c}$ as 0.07, 1.0, respectively. For the classification objective, we set $\tau_{s}$ to 0.1, and $\tau_{t}$ is initialised to 0.07, then warmed up to 0.04 with a cosine schedule in the starting 30 epochs. All experiments are done with an NVIDIA GeForce RTX 3090 GPU.

### 5.2 Comparison With the State of the Arts

<table><tbody><tr><td></td><td colspan="3">CUB</td><td colspan="3">Stanford Cars</td><td colspan="3">FGVC-Aircraft</td></tr><tr><td>Methods</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation> <annotation>italic_k</annotation></semantics></math> -means <sup><a href="#fn:30">30</a></sup></td><td>34.3</td><td>38.9</td><td>32.1</td><td>12.8</td><td>10.6</td><td>13.8</td><td>16.0</td><td>14.4</td><td>16.8</td></tr><tr><td>RS+ <sup><a href="#fn:22">22</a></sup></td><td>33.3</td><td>51.6</td><td>24.2</td><td>28.3</td><td>61.8</td><td>12.1</td><td>26.9</td><td>36.4</td><td>22.2</td></tr><tr><td>UNO+ <sup><a href="#fn:17">17</a></sup></td><td>35.1</td><td>49.0</td><td>28.1</td><td>35.5</td><td>70.5</td><td>18.6</td><td>40.3</td><td>56.4</td><td>32.2</td></tr><tr><td>ORCA <sup><a href="#fn:6">6</a></sup></td><td>35.3</td><td>45.6</td><td>30.2</td><td>23.5</td><td>50.1</td><td>10.7</td><td>22.0</td><td>31.8</td><td>17.1</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>51.3</td><td>56.6</td><td>48.7</td><td>39.0</td><td>57.6</td><td>29.9</td><td>45.0</td><td>41.1</td><td>46.9</td></tr><tr><td>SimGCD</td><td>60.3</td><td>65.6</td><td>57.7</td><td>53.8</td><td>71.9</td><td>45.0</td><td>54.2</td><td>59.1</td><td>51.8</td></tr><tr><td><math><semantics><mi>Δ</mi> <ci>Δ</ci> <annotation>\Delta</annotation> <annotation>roman_Δ</annotation></semantics></math></td><td>+9.0</td><td>+9.0</td><td>+9.0</td><td>+14.8</td><td>+14.3</td><td>+15.1</td><td>+9.2</td><td>+18.0</td><td>+4.9</td></tr></tbody></table>

Table 2: Results on the Semantic Shift Benchmark [^44].

<table><tbody><tr><td></td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td><td colspan="3">ImageNet-100</td></tr><tr><td>Methods</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation> <annotation>italic_k</annotation></semantics></math> -means <sup><a href="#fn:30">30</a></sup></td><td>83.6</td><td>85.7</td><td>82.5</td><td>52.0</td><td>52.2</td><td>50.8</td><td>72.7</td><td>75.5</td><td>71.3</td></tr><tr><td>RS+ <sup><a href="#fn:22">22</a></sup></td><td>46.8</td><td>19.2</td><td>60.5</td><td>58.2</td><td>77.6</td><td>19.3</td><td>37.1</td><td>61.6</td><td>24.8</td></tr><tr><td>UNO+ <sup><a href="#fn:17">17</a></sup></td><td>68.6</td><td>98.3</td><td>53.8</td><td>69.5</td><td>80.6</td><td>47.2</td><td>70.3</td><td>95.0</td><td>57.9</td></tr><tr><td>ORCA <sup><a href="#fn:6">6</a></sup></td><td>81.8</td><td>86.2</td><td>79.6</td><td>69.0</td><td>77.4</td><td>52.0</td><td>73.5</td><td>92.6</td><td>63.9</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>91.5</td><td>97.9</td><td>88.2</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td></tr><tr><td>SimGCD</td><td>97.1</td><td>95.1</td><td>98.1</td><td>80.1</td><td>81.2</td><td>77.8</td><td>83.0</td><td>93.1</td><td>77.9</td></tr><tr><td><math><semantics><mi>Δ</mi> <ci>Δ</ci> <annotation>\Delta</annotation> <annotation>roman_Δ</annotation></semantics></math></td><td>+5.6</td><td>-2.8</td><td>+9.9</td><td>+7.1</td><td>+5.0</td><td>+11.3</td><td>+8.9</td><td>+3.3</td><td>+11.6</td></tr></tbody></table>

Table 3: Results on generic image recognition datasets.

<table><tbody><tr><td></td><td colspan="3">Herbarium 19</td><td colspan="3">ImageNet-1K</td></tr><tr><td>Methods</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation> <annotation>italic_k</annotation></semantics></math> -means <sup><a href="#fn:30">30</a></sup></td><td>13.0</td><td>12.2</td><td>13.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RS+ <sup><a href="#fn:22">22</a></sup></td><td>27.9</td><td>55.8</td><td>12.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UNO+ <sup><a href="#fn:17">17</a></sup></td><td>28.3</td><td>53.7</td><td>14.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ORCA <sup><a href="#fn:6">6</a></sup></td><td>20.9</td><td>30.9</td><td>15.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>35.4</td><td>51.0</td><td>27.0</td><td>52.5</td><td>72.5</td><td>42.2</td></tr><tr><td>SimGCD</td><td>44.0</td><td>58.0</td><td>36.4</td><td>57.1</td><td>77.3</td><td>46.9</td></tr><tr><td><math><semantics><mi>Δ</mi> <ci>Δ</ci> <annotation>\Delta</annotation> <annotation>roman_Δ</annotation></semantics></math></td><td>+8.6</td><td>+7.0</td><td>+9.4</td><td>+4.6</td><td>+4.8</td><td>+4.7</td></tr></tbody></table>

Table 4: Results on more challenging datasets.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x8.png)

Figure 8: Step-by-step differences from GCD 43 to SimGCD. (SL: self-labelling, BR: post-backbone representation, SD: self-distillation, TW: teacher temperature warmup, JT: joint training)

We compare with state-of-the-art methods in generalized category discovery (ORCA [^6] and GCD [^43]), strong baselines derived from novel category discovery (RS+ [^22] and UNO+ [^17]), and $k$ -means [^30] on DINO [^9] features. On both the fine-grained SSB benchmark (Tab. 2) and generic image recognition datasets (Tab. 3), our method achieves notable improvements in recognising ‘New’ classes (across the instances in $\mathcal{D}^{u}$ that belong to classes in $\mathcal{Y}_{u}\text{\textbackslash}\mathcal{Y}_{l}$), outperforming the SOTAs by around 10%. The results in old classes are also competing against the best-performing baselines. Given that the ability to discover ‘New’ classes is a more desirable ability, the results are quite encouraging.

In Tab. 4, we also report the results on Herbarium 19 [^40], a naturally long-tailed fine-grained dataset that is closer to the real-world application of generalized category discovery, and ImageNet-1K [^14], a large-scale generic classification dataset. Still, our method shows consistent improvements in all metrics.

| Methods | CF100 | CUB | Herb19 | IN-100 | IN-1K |
| --- | --- | --- | --- | --- | --- |
| GCD [^43] | 7.5m | 9m | 2.5h | 36m | 7.7h |
| SimGCD | 1m | 18s | 3.5m | 9.5m | 0.6h |

Table 5: Inference time over the unlabelled split.

In Tab. 5, we compare the inference time with GCD [^43], one iconic non-parametric classification method. Let the number of all samples and unlabelled samples be $N$ and $N_{u}$, the number of classes $K$, feature dimension $d$, and the number of $k$ -means iterations to be $t$, the time complexity of GCD is $\mathcal{O}(N^{2}d+NKdt)$ (including $k$ -means++ initialisation), while our method only requires a nearest-neighbour prototype search for each instance, with time complexity $\mathcal{O}(N_{u}Kd)$. All methods adopt GPU implementations.

### 5.3 Ablation Study

In Fig. 8, we ablate the key components that bring the baseline method step-by-step to a new SOTA.

#### Baseline.

We start from GCD [^43], a non-parametric classification framework. We keep its representation learning objectives unchanged, and first impose the UNO [^17] -style self-labelling classification objectives (+SL) to it, thus transforming it into a parametric classifier. The classifier is built on the projector, and detached from representation learning. Results on ‘Old’ classes generally improve, while results on ‘New’ classes see a sharp drop. This is expected due to UNO’s strong bias toward ‘Old’ classes (Fig. 5).

#### Improving the representations.

As suggested in Sec. 3.2, we build the classifier on the backbone (+BR). This further makes notable improvements in ‘Old’ classes, while changes in ‘New’ classes vary across datasets. This indicates that the pseudo labels’ quality is insufficient to benefit from the post-backbone representations (Fig. 3).

#### Improving the pseudo labels.

We start by replacing the self-labelling strategy with our self-distillation paradigm. As shown in column (+SD), we achieve consistent improvements across all datasets by a large margin (*e.g*., 26% in CIFAR100, 13% in CUB) in ‘New’ classes. We then further adopt a teacher temperature warmup strategy (+TW) to lower the confidence of the pseudo-labels at an earlier stage. The intuition is that at the beginning, both the classifier and the representation are not well fitted to the target data, thus the pseudo-labels are not quite reliable. This is shown to be helpful for fine-grained classification datasets, while for generic classification datasets, which are similar to the pre-training data (ImageNet), the unreliable pseudo label is not a problem, thus lowering the confidence does not show help. For simplicity, we keep the training strategy consistent.

#### Jointly training the representation.

Previous settings adopt a decoupled training strategy for consistent representations with GCD [^43] and fair comparison. Finally, as confirmed in Sec. 3.3, we jointly supervise the representation with the classification objective (+JT). This results in a consistent improvement in ‘New’ classes for all datasets. Changes in ‘Old’ classes are mostly neutral or positive, with a notable drop in CIFAR100. Our intuition is that the original representations are already good enough for ‘Old’ classes in this dataset, and some incorrect pseudo labels lead to sight degradation in this case.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x9.png)

Figure 9: Effect of entropy regularisation on four types of classification errors. Appropriate entropy regularisation helps overcome the bias between ‘Old’/‘New’ classes (see “False New” and “False Old”, lower is better).

![Refer to caption](https://arxiv.org/html/2211.11727v4/x10.png)

Figure 10: Per-class prediction distributions with different entropy regularisation weights. Proper entropy regularisation helps overcome the bias across ‘Old’/‘New’ classes, and approach the GT class distribution.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x11.png)

Figure 11: Results with different numbers of categories. Stronger entropy regularisation effectively enforces the model’s robustness to unknown numbers of categories, but over-regularisation may limit the ability to recognise ‘New’ classes under ground-truth class numbers.

### 5.4 Analyses And Discussions

#### Entropy regularisation helps overcome prediction bias.

We verify the effectiveness of entropy regularisation in overcoming prediction bias by diagnosing the model’s classification errors and class-wise prediction distributions. Fig. 9 shows that this term consistently helps reduce “False New” and “False Old” errors, which refer to predicting an ‘Old’ class sample to a ‘New’ class, and vice-versa. Besides, Fig. 10 shows proper entropy regularisation helps overcome the imbalanced pseudo labels across all classes, and approach the ground truth (GT) class distribution.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x12.png)

Figure 12: Per-class prediction distributions with different numbers of categories. Our method effectively identifies the criterion for ‘New’ classes, thus keeping the number of active prototypes close to the ground-truth class number.

#### Entropy regularisation enforces robustness to unknown numbers of categories.

The main text assumed the category number is known a-priori following prior works [^22] [^52] [^57] [^17], which is impractical [^55]. In Fig. 11, we present the results with different numbers of categories on five representative datasets. A category number lower than the ground truth significantly limits the ability to discover ‘New’ categories, and the model tends to focus more on the ‘Old’ classes. On the other hand, increasing the category number results in less harm to the generic image recognition datasets and can even be helpful for some datasets. When a stronger entropy penalty is imposed, the model shows strong robustness to the category number. Interestingly, further analysis in Fig. 12 shows the network prefers to keep the number of active prototypes low and close to the real category number. This finding is inspiring and could ease the deployment of GCD in real-world scenarios.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x13.png)

Figure 13: Prediction analysis against GCD 43. Left: Based on identical representations, the non-parametric classifier (semi-supervised k 𝑘 italic\_k -means) adopted by GCD produces highly imbalanced predictions, while our method better fits the true distribution; Right: our method significantly improves GCD’s tail classes.

#### What makes for the significant improvements over GCD given identical representations?

One interesting message from Fig. 8 is that, even with the same representations (col. +TW), we can already improve GCD by a large margin. We thus study the classification predictions and the major components that lead to the performance gap. As shown in Fig. 13, the non-parametric classifier (semi-supervised $k$ -means) adopted by GCD [^43] produces highly imbalanced predictions, while our method better fits the true distribution. Further analysis (right part) shows that our method significantly improves over the tail classes of GCD.

#### How does the classification objective change the representations?

In Fig. 8, we have shown that jointly training the representations with the classification objective can lead to $\sim$ 15% boost in ‘New’ classes on CIFAR100. We study this difference by visualising the representations before and after tuning with t-SNE [^42]. As in Fig. 14, jointly tuning the feature leads to less ambiguity, larger margins, and compacter clusters. Concerning why this is not as helpful for CUB: we hypothesise that one important factor lies in how transferable the features learned in ‘Old’ classes are to ‘New’ classes. While it may be easier for a cat classifier to be adapted to dogs, things can be different for fine-grained bird recognition. Besides, the small scale of CUB, which contains only 6k images while holding a large class split (200), might also make it hard to learn transferable features.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x14.png)

Figure 14: T-SNE 42 visualisation of the representations of 10 classes randomly sampled from CIFAR100 29. Jointly supervising representation learning with a classification objective helps disambiguate ( e.g, bed & table) and forms compacter clusters.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x15.png)

Figure 15: Performance evolution throughout the model learning process. We observe a trade-off between the performance in ‘Old’ and ‘New’ categories, which is common across datasets.

#### Trade-off between ‘Old’ and ‘New’ categories.

We plot the performance evolution throughout the model learning process in Fig. 15. It can be observed that the performance on the ‘Old’ categories first climbs to the highest point at the early stage of training and then slowly degrades as the performance on the ‘New’ categories improves. We believe this demonstrates an important aspect of the design of models for the GCD problem: the performance on the ‘Old’ categories may be in odd with the performance on the ‘New’ categories, how to achieve a better trade-off between these two could be an interesting investigation for future works.

## 6 Limitations and Potential Future Works

#### Representation learning.

This paper mainly targets improving the classification ability for generalized category discovery. The representation learning, however, follows the prior work GCD [^43]. It is expectable that the quality of representation learning can be improved. For instance, generally, by using more advanced geometric and photometric data augmentations [^19], and even multiple local crops [^8]. Further, can the design of data augmentations be better aligned with the classification criteria of the target data? For another example, using a large batch size has been shown to be critical to the performance of contrastive learning-based frameworks [^10]. However, the batch size adopted by GCD [^43] is only 128, which might limit the quality of learned representations. Moreover, is the supervised contrastive learning plus self-supervised contrastive learning paradigm the ultimate answer to form the feature manifold? We believe that advances in representation learning can lead to further gains.

#### Alignment to human-defined categories.

This paper follows the common practice of previous works where human labels in seen categories implicitly define the metric for unseen ones, which can be viewed as an effort to align algorithm-discovered categories with human-defined ones. However, labels in seen categories may not be good guidance when there is a gap between seen ones and the novel categories we want to discover, *e.g*., how to use the labelled images in ImageNet to discover novel categories in CUB? For another example, when we use a very big class vocabulary (*e.g*., the full ImageNet-22K [^14]), categories could overlap with each other, and be in different granularities. Further, assigning text names to the discovered categories still requires a matching process, what if further utilising the relationship between class names, and directly predicting the novel categories in the text space? We believe the alignment between algorithm-discovered categories and human-defined categories is of high research value for future works.

#### Ethical considerations.

Current methods commonly suffer from low-data or long-tailed scenarios. Depending on the data and classification criteria of specific tasks, discrimination against minority categories or instances is possible.

## 7 Conclusion

This study investigates the reasons behind the failure of previous parametric classifiers in recognizing novel classes in GCD and uncovers that unreliable pseudo-labels, which exhibit significant biases, are the crucial factor. We propose a simple yet effective parametric classification method that addresses these issues and achieves state-of-the-art performance on multiple GCD benchmarks. Our findings provide insights into the design of robust classifiers for discovering novel categories and we hope our proposed framework will serve as a strong baseline to facilitate future studies in this field and contribute to the development of more accurate and reliable methods for category discovery.

## Acknowledgements

This work has been supported by Hong Kong Research Grant Council - Early Career Scheme (Grant No. 27209621), General Research Fund Scheme (Grant No. 17202422), and RGC Matching Fund Scheme (RMGS). Part of the described research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust. The authors acknowledge SmartMore and MEGVII for partial computing support, and Zhisheng Zhong for professional suggestions.

## References

Parametric Classification for Generalized Category Discovery: A Baseline Study  
Supplementary Material  
Xin Wen <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math> *</sup>   Bingchen Zhao <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="2"><semantics><mn>2</mn> <cn type="integer">2</cn> <annotation>2</annotation> <annotation>2</annotation></semantics></math> *</sup>   Xiaojuan Qi <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math></sup>  
<sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1"><semantics><mn>1</mn> <cn type="integer">1</cn> <annotation>1</annotation> <annotation>1</annotation></semantics></math></sup> The University of Hong Kong   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="2"><semantics><mn>2</mn> <cn type="integer">2</cn> <annotation>2</annotation> <annotation>2</annotation></semantics></math></sup> University of Edinburgh  
{wenxin,xjqi}@eee.hku.hk   bingchen.zhao@ed.ac.uk

## Appendix A Implementation Details

### A.1 Experiment Setting Details

The split of labelled (‘Old’) and unlabelled (‘New’) categories follows GCD [^43]. That is, 50% of all classes are sampled as ‘Old’ classes ($\mathcal{Y}_{l}$), and the rest are regarded as ‘New’ classes ($\mathcal{Y}_{u}\setminus\mathcal{Y}_{l}$). The exception is CIFAR100, for which 80% classes are sampled as ‘Old’, following the novel category discovery (NCD) literature. Regarding the sampling process, for generic object recognition datasets, the labelled classes are selected by their class index (the first $|\mathcal{Y}_{l}|$ ones). For the Semantic Shift Benchmark, data splits provided in [^44] are adopted. For Herbarium 19 [^40], the labelled classes are sampled randomly. Additionally, for ImageNet-1K [^14] which is not used in [^43], we follow its fashion to select the first 500 classes sorted by class id as the labelled classes. Then for all datasets, following [^43], 50% of the images from the labelled classes are randomly sampled to form the labelled dataset $\mathcal{D}^{l}$, and all remaining images are regarded as the unlabelled dataset $\mathcal{D}^{u}$. All experiments are done with a batch size of 128 on a single GPU, except for ImageNet-1K, on which we train with eight GPUs, scale the learning rate with the linear scaling rule, and keep the per-GPU batch size unchanged. The inference time on ImageNet-1K is still evaluated with one GPU.

### A.2 Re-implementing Previous Works

Results of GCD [^43] are taken from the original paper (if available), and otherwise re-implemented with the official codebase. One exception is ImageNet-1K [^14], which was not evaluated by the authors. Since naively adopting their official codebase to ImageNet-1K fails as the semi-supervised $k$ -means procedure requires too much GPU memory and cannot be done with available hardware, we drop the $k$ -mean++ initialisation [^1] which takes the most memory, and re-implement the method with faiss [^26] for speed up (otherwise the evaluation takes more than one day). The results are in the main paper, compared to our proposed strong baseline SimGCD, GCD requires significantly more time to run and more engineering efforts, and yet achieves a lower performance than SimGCD, which demonstrates the effectiveness of our proposed method. Results of UNO+ [^17] and RS+ [^21], which are adaptations of the original works to the GCD task, are directly taken from the GCD [^43] paper. Also note that unlike UNO [^17], our method does not adopt the over-clustering trick for simplicity. Results of ORCA [^6] are re-implemented with the official codebase. We align the details in dataset split and backbone (ViT-B/16 [^15] pre-trained with DINO [^9]) with GCD [^43] for a fair comparison.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x16.png)

Figure 16: Complete error analysis results of SimGCD on five representative datasets. With appropriate entropy regularisation, the bias between ‘Old’/‘New’ classes (see “False New” and “False Old” errors) are generally effectively alleviated, except in the long-tailed Herbarium 19 that the effect varies. Also notably, “True New” errors are consistently penalised to a considerable extent, confirming entropy regularisation’s ability in helping recognise and distinguish between novel categories.

### A.3 Error Analysis Details

We briefly clarify the details of obtaining the four kinds of prediction errors in the main paper: we first rank the category indexes in consecutive order, such that by index, all ‘Old’ classes are followed by all ‘New’ classes. We then compute the full confusion matrix, with each element summarising how many times images of one specific class (row index) are predicted as one class (column index). All elements are divided by the number of testing samples to account for the percentage. We then reduce the diagonal terms to zero (representing correct predictions), and thus all remaining elements represent different kinds of prediction errors (*i.e*., absolute contribution to the errors of ‘All’ ACC). Finally, we slice the confusion matrix into four sub-matrices at the boundaries between the ‘Old’ and ‘New’ classes, and add all elements in each sub-matrix together, thus obtaining the final error matrix standing for the four kinds of prediction errors. Such a way of error classification helps distinguish the prediction bias between and within seen and novel categories, and thus facilitates the design of new solutions. Note that the diagonal elements, *e.g*., ‘True Old’ predictions, do not stand for correct predictions, but for cases that incorrectly predicting samples of one specific ‘Old’ class to another wrong ‘Old’ class.

## Appendix B Extended Experiments And Discussions

### B.1 Main Results

We present the full results of SimGCD in the main paper with error bars in Tab. 6. The results are obtained from three independent runs and thus avoid randomness.

| Dataset | All | Old | New |
| --- | --- | --- | --- |
| CIFAR10 [^29] | 97.1 $\pm$ 0.0 | 95.1 $\pm$ 0.1 | 98.1 $\pm$ 0.1 |
| CIFAR100 [^29] | 80.1 $\pm$ 0.9 | 81.2 $\pm$ 0.4 | 77.8 $\pm$ 2.0 |
| ImageNet-100 [^14] | 83.0 $\pm$ 1.2 | 93.1 $\pm$ 0.2 | 77.9 $\pm$ 1.9 |
| ImageNet-1K [^14] | 57.1 $\pm$ 0.1 | 77.3 $\pm$ 0.1 | 46.9 $\pm$ 0.2 |
| CUB [^48] | 60.3 $\pm$ 0.1 | 65.6 $\pm$ 0.9 | 57.7 $\pm$ 0.4 |
| Stanford Cars [^28] | 53.8 $\pm$ 2.2 | 71.9 $\pm$ 1.7 | 45.0 $\pm$ 2.4 |
| FGVC-Aircraft [^31] | 54.2 $\pm$ 1.9 | 59.1 $\pm$ 1.2 | 51.8 $\pm$ 2.3 |
| Herbarium 19 [^40] | 44.0 $\pm$ 0.4 | 58.0 $\pm$ 0.4 | 36.4 $\pm$ 0.8 |

Table 6: Complete results of SimGCD in three independent runs.

### B.2 Unknown Category Number

In the main text, we showed that the performance of SimGCD is robust to a wide range of estimated unknown category numbers. In this section, we report the results with the number of categories estimated using an off-the-shelf method [^43] (Tab. 7) or with a roughly estimated relatively big number (two times of the ground-truth $K$), and compare with the baseline method GCD [^43].

|  | CIFAR100 | ImageNet-100 | CUB | SCars | Herb19 |
| --- | --- | --- | --- | --- | --- |
| GT $K$ | 100 | 100 | 200 | 196 | 683 |
| Est. $K$ | 100 | 109 | 231 | 230 | 520 |

Table 7: Number of categories $K$ estimated using [^43].

<table><tbody><tr><td></td><td></td><td colspan="3">CIFAR100</td><td colspan="3">ImageNet-100</td></tr><tr><td>Methods</td><td>Known <math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation> <annotation>italic_K</annotation></semantics></math></td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>✓</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td></tr><tr><td>SimGCD</td><td>✓</td><td>80.1</td><td>81.2</td><td>77.8</td><td>83.0</td><td>93.1</td><td>77.9</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>✗ (w/ Est.)</td><td>73.0</td><td>76.2</td><td>66.5</td><td>72.7</td><td>91.8</td><td>63.8</td></tr><tr><td>SimGCD</td><td>✗ (w/ Est.)</td><td>80.1</td><td>81.2</td><td>77.8</td><td>81.7</td><td>91.2</td><td>76.8</td></tr><tr><td>SimGCD</td><td>✗ (w/ <math><semantics><mrow><mn>2</mn> <mo>⁢</mo> <mi>K</mi></mrow> <apply><cn>2</cn> <ci>𝐾</ci></apply> <annotation>2K</annotation> <annotation>2 italic_K</annotation></semantics></math>)</td><td>77.7</td><td>79.5</td><td>74.0</td><td>80.9</td><td>93.4</td><td>74.8</td></tr></tbody></table>

Table 8: Results on generic image recognition datasets.

<table><tbody><tr><td></td><td></td><td colspan="3">CUB</td><td colspan="3">Stanford Cars</td></tr><tr><td>Methods</td><td>Known <math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation> <annotation>italic_K</annotation></semantics></math></td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>✓</td><td>51.3</td><td>56.6</td><td>48.7</td><td>39.0</td><td>57.6</td><td>29.9</td></tr><tr><td>SimGCD</td><td>✓</td><td>60.3</td><td>65.6</td><td>57.7</td><td>53.8</td><td>71.9</td><td>45.0</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>✗ (w/ Est.)</td><td>47.1</td><td>55.1</td><td>44.8</td><td>35.0</td><td>56.0</td><td>24.8</td></tr><tr><td>SimGCD</td><td>✗ (w/ Est.)</td><td>61.5</td><td>66.4</td><td>59.1</td><td>49.1</td><td>65.1</td><td>41.3</td></tr><tr><td>SimGCD</td><td>✗ (w/ <math><semantics><mrow><mn>2</mn> <mo>⁢</mo> <mi>K</mi></mrow> <apply><cn>2</cn> <ci>𝐾</ci></apply> <annotation>2K</annotation> <annotation>2 italic_K</annotation></semantics></math>)</td><td>63.6</td><td>68.9</td><td>61.1</td><td>48.2</td><td>64.6</td><td>40.2</td></tr></tbody></table>

Table 9: Results on the Semantic Shift Benchmark [^44].

The results on CIFAR100 [^29], ImageNet-100 [^14], CUB [^48], and Stanford Cars [^28] are available in Tabs. 8 and 9. Our method shows consistent improvements on four representative datasets when $K$ is unknown, no matter with the category number estimated with a specialised algorithm (w/ Est.), or simply with a loose estimation that is two times the ground truth (w/ $2K$, other values are also applicable since our method is robust to a wide range of estimations). This property could ease the deployment of parametric classifiers for GCD in real-world scenarios.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x17.png)

Figure 17: Complete per-class prediction distribution results of SimGCD on five representative datasets. Proper entropy regularisation helps overcome the prediction bias in both ‘Old’ classes and ‘New’ classes, and fits the ground-truth distribution. The conclusion is consistent across generic classification datasets, fine-grained classification datasets, and naturally long-tailed datasets.

### B.3 Extended Analyses

In supplementary to the main paper, we present a more complete version of the analytical experiments.

In Fig. 16, we show the error analysis results of SimGCD over five representative datasets that cover coarse-grained, fine-grained, and long-tailed classification tasks. Overall, it shows that the entropy regulariser mainly helps in overcoming two types of errors: the error of misclassification between ‘Old’/‘New’ categories, and the error of misclassification within ‘New’ categories. One exception is the long-tailed Herbarium 19 dataset, in which the models’ “False Old” errors also increased, and our intuition is that the long-tailed distribution adds to the difficulty in discriminating between ‘Old’ and ‘New’ categories. Still, the gain in distinguishing between novel categories is consistent, and we provide a further analysis via per-class prediction distributions in the next paragraph.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x18.png)

Figure 18: A closer look at the per-class distributions. Notably, although the entropy regularisation term is formulated to approach uniform distribution, it could make the models’ predictions more biased on the class-balanced ImageNet-100 dataset when the regularisation is too strong. Interestingly, it also could help fit the distribution of the long-tailed Herbarium 19 dataset.

In Fig. 17, we show the complete per-class prediction results of SimGCD to further analyse the entropy regulariser’s effect in overcoming the classification errors within ‘Old’ and ‘New’ classes, and it consistently verifies the help in alleviating the prediction bias within ‘Old’ and ‘New’ classes, and better fitting the ground-truth class distribution. In Fig. 18, we present a closer look at ImageNet-100 and Herbarium 19. The entropy regularisation term is formulated to make the model’s predictions closer to the uniform distribution. But interestingly, we empirically found that it could make the models’ predictions more biased on the class-balanced ImageNet-100 dataset when the regularisation is too strong. And when the dataset itself is long-tailed (Herbarium 19), it also could help fit the ground-truth distribution. We also note that the self-labelling strategy adopted by UNO [^17] forces the predictions in a batch to be strictly uniform, which may account for its inferior performance.

![Refer to caption](https://arxiv.org/html/2211.11727v4/x19.png)

Figure 19: Per-class prediction distributions using different category numbers on ImageNet-100 and Herbarium 19. Our method effectively identifies the criterion for ‘New’ classes, thus keeping the number of active prototypes close to the ground-truth class number. Notably, a loose category number greater than the ground truth may harm fitting the class-balanced ImageNet-100 dataset, but could help fit the distribution of the long-tailed Herbarium 19 dataset.

<table><tbody><tr><td></td><td></td><td colspan="3">CIFAR100</td><td colspan="3">ImageNet-100</td><td colspan="3">CUB</td><td colspan="3">Stanford Cars</td><td colspan="3">Herbarium 19</td></tr><tr><td>Method</td><td>Logit Adjust</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>ORCA <sup><a href="#fn:6">6</a></sup></td><td>✓</td><td>69.0</td><td>77.4</td><td>52.0</td><td>73.5</td><td>92.6</td><td>63.9</td><td>35.3</td><td>45.6</td><td>30.2</td><td>23.5</td><td>50.1</td><td>10.7</td><td>20.9</td><td>30.9</td><td>15.5</td></tr><tr><td>DebiasPL <sup><a href="#fn:47">47</a></sup></td><td>✓</td><td>60.9</td><td>69.8</td><td>43.1</td><td>43.5</td><td>59.1</td><td>35.6</td><td>38.1</td><td>44.2</td><td>35.0</td><td>31.1</td><td>49.6</td><td>22.1</td><td>30.1</td><td>39.1</td><td>25.3</td></tr><tr><td>UNO+ <sup><a href="#fn:17">17</a></sup></td><td>✗</td><td>69.5</td><td>80.6</td><td>47.2</td><td>70.3</td><td>95.0</td><td>57.9</td><td>35.1</td><td>49.0</td><td>28.1</td><td>35.5</td><td>70.5</td><td>18.6</td><td>28.3</td><td>53.7</td><td>14.7</td></tr><tr><td>GCD <sup><a href="#fn:43">43</a></sup></td><td>✗</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td><td>51.3</td><td>56.6</td><td>48.7</td><td>39.0</td><td>57.6</td><td>29.9</td><td>35.4</td><td>51.0</td><td>27.0</td></tr><tr><td>SimGCD</td><td>✗</td><td>80.1</td><td>81.2</td><td>77.8</td><td>83.0</td><td>93.1</td><td>77.9</td><td>60.3</td><td>65.6</td><td>57.7</td><td>53.8</td><td>71.9</td><td>45.0</td><td>44.0</td><td>58.0</td><td>36.4</td></tr></tbody></table>

Table 10: Comparison to imbalanced recognition-inspired methods.

In Fig. 19, we also show the per-class prediction distributions using different category numbers. The results on the class-balanced ImageNet-100 are consistent with the results on CIFAR100 and CUB in the main paper, using a loose category number greater than the ground truth may harm fitting the ground-truth class distribution, yet the model still manages to find the ground truth category number. Interestingly, we also find that for the long-tailed Herbarium 19 dataset, using a greater category number could in fact help fit the ground-truth distribution.

### B.4 Relationship to Imbalanced Recognition

Our work also shares motivation with literature in long-tailed/imbalanced recognition [^32] [^46] [^35], in which resolving the imbalance in models’ prediction is also an important issue. Technically, they commonly depend on a prior class distribution to adjust classifiers’ output, which is not accessible in GCD since labels for novel classes are unknown. One could also estimate this distribution online from predictions, which is inaccurate due to its open-world nature. We note one baseline (ORCA [^6]) compared in the paper also shares key intuition with these works (adaptive margin). We also reimplement one close work that operates on imbalanced semi-supervised learning, *i.e*., DebiasPL [^47], aligning representation learning with GCD, and show a comparison in Tab. 10. DebiasPL surpasses UNO+ on fine-grained classification in novel classes and verifies it could overcome the prediction imbalance to some extent. It also outperforms ORCA but still lags behind GCD and ours. We hypothesise manually altering logits may not be suitable for open-world settings. Instead, a more natural and general solution could be to regularise prediction statistics and let the model adjust via optimisation.

[^1]: David Arthur and Sergei Vassilvitskii. k-means++: the advantages of careful seeding. In ACM-SIAM Symposium on Discrete Algorithms, 2007.

[^2]: Yuki M. Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR, 2020.

[^3]: Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Mike Rabbat, and Nicolas Ballas. Masked siamese networks for label-efficient learning. In ECCV, 2022.

[^4]: Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. Semi-supervised learning of visual features by non-parametrically predicting view assignments with support samples. In ICCV, 2021.

[^5]: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. Mixmatch: A holistic approach to semi-supervised learning. In NeurIPS, 2019.

[^6]: Kaidi Cao, Maria Brbić, and Jure Leskovec. Open-world semi-supervised learning. In ICLR, 2022.

[^7]: Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep Clustering for Unsupervised Learning of Visual Features. In ECCV, 2018.

[^8]: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In nips, 2020.

[^9]: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

[^10]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

[^11]: Yanbei Chen, Xiatian Zhu, Wei Li, and Shaogang Gong. Semi-supervised learning under class distribution mismatch. In AAAI, 2020.

[^12]: Jang Hyun Cho, Utkarsh Mall, Kavita Bala, and Bharath Hariharan. Picie: Unsupervised semantic segmentation using invariance and equivariance in clustering. In CVPR, 2021.

[^13]: Quan Cui, Bingchen Zhao, Zhao-Min Chen, Borui Zhao, Renjie Song, Jiajun Liang, Boyan Zhou, and Osamu Yoshie. Discriminability-transferability trade-off: An information-theoretic perspective. In ECCV, 2022.

[^14]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[^15]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[^16]: Yixin Fei, Zhongkai Zhao, Siwei Yang, and Bingchen Zhao. Xcon: Learning with experts for fine-grained category discovery. In BMVC, 2022.

[^17]: Enrico Fini, Enver Sangineto, Stéphane Lathuilière, Zhun Zhong, Moin Nabi, and Elisa Ricci. A unified objective for novel class discovery. In ICCV, 2021.

[^18]: Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.

[^19]: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In NeurIPS, 2020.

[^20]: Lan-Zhe Guo, Zhen-Yu Zhang, Yuan Jiang, Yu-Feng Li, and Zhi-Hua Zhou. Safe deep semi-supervised learning for unseen-class unlabeled data. In ICML, 2020.

[^21]: Kai Han, Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Andrea Vedaldi, and Andrew Zisserman. Automatically discovering and learning new visual categories with ranking statistics. In ICLR, 2020.

[^22]: Kai Han, Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Andrea Vedaldi, and Andrew Zisserman. Autonovel: Automatically discovering and learning novel visual categories. IEEE TPAMI, 2021.

[^23]: Kai Han, Andrea Vedaldi, and Andrew Zisserman. Learning to discover novel visual categories via deep transfer clustering. In ICCV, 2019.

[^24]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[^25]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[^26]: Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 2019.

[^27]: Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In NeurIPS, 2020.

[^28]: Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In ICCV Workshops, 2013.

[^29]: Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report, 2009.

[^30]: James MacQueen. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1967.

[^31]: Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[^32]: Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In ICLR, 2021.

[^33]: Avital Oliver, Augustus Odena, Colin Raffel, Ekin D Cubuk, and Ian J Goodfellow. Realistic evaluation of deep semi-supervised learning algorithms. In NeurIPS, 2018.

[^34]: Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Semi-supervised learning with scarce annotations. In CVPR Workshops, 2020.

[^35]: Jiawei Ren, Cunjun Yu, Xiao Ma, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Balanced meta-softmax for long-tailed visual recognition. In NeurIPS, 2020.

[^36]: Subhankar Roy, Mingxuan Liu, Zhun Zhong, Nicu Sebe, and Elisa Ricci. Class-incremental novel class discovery. In ECCV, 2022.

[^37]: Kuniaki Saito, Donghyun Kim, and Kate Saenko. Openmatch: Open-set semi-supervised learning with open-set consistency regularization. In NeurIPS, 2021.

[^38]: Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPS, 2020.

[^39]: Yiyou Sun and Yixuan Li. Opencon: Open-world contrastive learning. TMLR, 2023.

[^40]: Kiat Chuan Tan, Yulong Liu, Barbara Ambrose, Melissa Tulig, and Serge Belongie. The herbarium challenge 2019 dataset. arXiv preprint arXiv:1906.05372, 2019.

[^41]: Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In NeurIPS, 2017.

[^42]: Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 2008.

[^43]: Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Generalized category discovery. In CVPR, 2022.

[^44]: Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Open-set recognition: A good closed-set classifier is all you need? In ICLR, 2022.

[^45]: Feng Wang, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. Normface: L2 hypersphere embedding for face verification. In ACM MM, 2017.

[^46]: Xudong Wang, Long Lian, Zhongqi Miao, Ziwei Liu, and Stella X Yu. Long-tailed recognition by routing diverse distribution-aware experts. In ICLR, 2021.

[^47]: Xudong Wang, Zhirong Wu, Long Lian, and Stella X Yu. Debiased learning from naturally imbalanced pseudo-labels. In CVPR, 2022.

[^48]: Peter Welinder, Steve Branson, Takeshi Mita, Catherine Wah, Florian Schroff, Serge Belongie, and Pietro Perona. Caltech-ucsd birds 200. Technical Report CNS-TR-201, Caltech, 2010.

[^49]: Xin Wen, Bingchen Zhao, Anlin Zheng, Xiangyu Zhang, and Xiaojuan Qi. Self-supervised visual representation learning with semantic grouping. In NeurIPS, 2022.

[^50]: Qing Yu, Daiki Ikami, Go Irie, and Kiyoharu Aizawa. Multi-task curriculum framework for open-set semi-supervised learning. In ECCV, 2020.

[^51]: Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4l: Self-supervised semi-supervised learning. In ICCV, 2019.

[^52]: Bingchen Zhao and Kai Han. Novel visual category discovery with dual ranking statistics and mutual knowledge distillation. In NeurIPS, 2021.

[^53]: Bingchen Zhao and Oisin Mac Aodha. Incremental generalized category discovery. In ICCV, 2023.

[^54]: Bingchen Zhao and Xin Wen. Distilling visual priors from self-supervised learning. In ECCV Workshops, 2020.

[^55]: Bingchen Zhao, Xin Wen, and Kai Han. Learning semi-supervised gaussian mixture models for generalized category discovery. In ICCV, 2023.

[^56]: Zhun Zhong, Enrico Fini, Subhankar Roy, Zhiming Luo, Elisa Ricci, and Nicu Sebe. Neighborhood contrastive learning for novel class discovery. In CVPR, 2021.

[^57]: Zhun Zhong, Linchao Zhu, Zhiming Luo, Shaozi Li, Yi Yang, and Nicu Sebe. Openmix: Reviving known knowledge for discovering novel visual categories in an open world. In CVPR, 2021.