---
paper id: 2504.06120v1
title: Hyperbolic Category Discovery
authors: [Yuanpei Liu, Zhenqi He, Kai Han]
publication date: 2025-04-08T15:12
abstract: "Generalized Category Discovery (GCD) is an intriguing open-world problem that has garnered increasing attention. Given a dataset that includes both labelled and unlabelled images, GCD aims to categorize all images in the unlabelled subset, regardless of whether they belong to known or unknown classes. In GCD, the common practice typically involves applying a spherical projection operator at the end of the self-supervised pretrained backbone, operating within Euclidean or spherical space. However, both of these spaces have been shown to be suboptimal for encoding samples that possesses hierarchical structures. In contrast, hyperbolic space exhibits exponential volume growth relative to radius, making it inherently strong at capturing the hierarchical structure of samples from both seen and unseen categories. Therefore, we propose to tackle the category discovery challenge in the hyperbolic space. We introduce HypCD, a simple \\underline{Hyp}erbolic framework for learning hierarchy-aware representations and classifiers for generalized \\underline{C}ategory \\underline{D}iscovery. HypCD first transforms the Euclidean embedding space of the backbone network into hyperbolic space, facilitating subsequent representation and classification learning by considering both hyperbolic distance and the angle between samples. This approach is particularly helpful for knowledge transfer from known to unknown categories in GCD. We thoroughly evaluate HypCD on public GCD benchmarks, by applying it to various baseline and state-of-the-art methods, consistently achieving significant improvements."
comments: Accepted as a conference paper at CVPR 2025
pdf: "[[paper/pdf/Hyperbolic Category Discovery (2504.06120v1).pdf]]"
url: https://arxiv.org/abs/2504.06120v1
tags: []
---
### 1. Tóm tắt Cốt lõi

Bài báo đề xuất framework HypCD, chuyển đổi biểu diễn dữ liệu sang không gian Hyperbolic để giải quyết bài toán GCD. Không gian này giúp mã hóa tốt hơn cấu trúc phân cấp (hierarchical structure) của đối tượng, cải thiện mạnh mẽ việc truyền tri thức sang các danh mục chưa biết.

---
### 2. Vấn đề Nghiên cứu

- **Bài toán GCD:** Gom cụm và phân loại tập dữ liệu chưa gán nhãn chứa cả danh mục đã biết và chưa biết.
- **Điểm mù của SOTA:** Các mô hình hiện tại dùng không gian Euclidean hoặc Spherical – những không gian bị giới hạn về thể tích, không đủ để biểu diễn cấu trúc phân cấp phức tạp của các bộ phận đối tượng.

---
### 3. Giải pháp Đề xuất

- **Mapping:** Dùng phép ánh xạ hàm mũ (exponential mapping) để đưa đặc trưng từ backbone sang quả cầu Poincaré (không gian Hyperbolic).
- **Hybrid Loss:** Kết hợp cả khoảng cách và góc để tối ưu hóa việc học biểu diễn.
- **Phân loại:** Thay thế MLP truyền thống bằng mạng truyền thẳng Hyperbolic (HypFFN).

---
### 4. Kết quả Thực nghiệm

- Thử nghiệm trên các bộ dữ liệu thô (CIFAR, ImageNet-100) và mịn (CUB, Stanford-Cars) với DINO/DINOv2.
- **Metric:** All Novel Known Accuracy.
- **Đột phá:** Đạt SOTA mới, ví dụ Hyp-SelEx đạt ACC 90.7% trên tập CUB. Giảm mạnh mức chênh lệch hiệu suất giữa danh mục cũ và mới.