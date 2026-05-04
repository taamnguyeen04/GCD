---
paper id: 2108.08536v4
title: A Unified Objective for Novel Class Discovery
authors: [Enrico Fini, Enver Sangineto, Stéphane Lathuilière, Zhun Zhong, Moin Nabi, Elisa Ricci]
publication date: 2021-08-19T07:22
abstract: "In this paper, we study the problem of Novel Class Discovery (NCD). NCD aims at inferring novel object categories in an unlabeled set by leveraging from prior knowledge of a labeled set containing different, but related classes. Existing approaches tackle this problem by considering multiple objective functions, usually involving specialized loss terms for the labeled and the unlabeled samples respectively, and often requiring auxiliary regularization terms. In this paper, we depart from this traditional scheme and introduce a UNified Objective function (UNO) for discovering novel classes, with the explicit purpose of favoring synergy between supervised and unsupervised learning. Using a multi-view self-labeling strategy, we generate pseudo-labels that can be treated homogeneously with ground truth labels. This leads to a single classification objective operating on both known and unknown classes. Despite its simplicity, UNO outperforms the state of the art by a significant margin on several benchmarks (~+10% on CIFAR-100 and +8% on ImageNet). The project page is available at: https://ncd-uno.github.io."
comments: ICCV 2021 (Oral)
pdf: "[[paper/pdf/A Unified Objective for Novel Class Discovery (2108.08536v4).pdf]]"
url: https://arxiv.org/abs/2108.08536v4
tags: []
---
### 1. Tóm tắt Cốt lõi

Bài báo giới thiệu phương pháp UNO cho bài toán NCD loại bỏ hoàn toàn bước tiền huấn luyện tự giám sát. Bằng cách nối các logit và dùng thuật toán gán nhãn giả, mô hình xử lý cả dữ liệu có nhãn và không nhãn qua một hàm Cross-Entropy duy nhất, đạt hiệu suất SOTA vượt trội.

---
### 2. Vấn đề Nghiên cứu

- **Bài toán**: Phân cụm các lớp mới trong tập dữ liệu không nhãn, tận dụng tri thức từ tập đã gán nhãn (không trùng lớp).
- **Hạn chế cũ**: Các mô hình cũ quá cồng kềnh vì dùng nhiều hàm loss rời rạc (BCE, CE, consistency loss) và bắt buộc phải qua bước tiền huấn luyện (pretraining) tự giám sát trên toàn bộ dữ liệu.

---
### 3. Phương pháp Đề xuất (UNO)

- **Kiến trúc hợp nhất**: Nối logit từ nhánh dự đoán lớp có nhãn (linear classifier) và nhánh dự đoán lớp không nhãn (MLP) vào chung một lớp Softmax.
- **Gán nhãn giả đa góc nhìn (Multi-view Pseudo-labeling)**: Tạo 2 biến thể của cùng một ảnh. Dùng thuật toán Sinkhorn-Knopp để chia đều xác suất nhãn giả cho một biến thể, sau đó dùng nhãn này để giám sát biến thể còn lại.
- **Bổ trợ**: Dùng thêm nhánh phân cụm siêu hạt (overclustering) để ép mô hình học biểu diễn đặc trưng sâu hơn.

---
### 4. Dữ liệu & Đánh giá

- **Dữ liệu**: Đánh giá trên CIFAR-10, CIFAR-100, ImageNet. Tác giả đề xuất thêm bộ CIFAR100-50 (50 lớp có nhãn, 50 lớp không nhãn) để mô phỏng thực tế khó khăn hơn.
- **Metric:** Accuracy và Clustering Accuracy.
- **Kết quả**: Đè bẹp các SOTA trước đó: +10% độ chính xác trên CIFAR-100 và +8% trên ImageNet. Đặc biệt xuất sắc trong bài test _task-agnostic_ (phải tự phân biệt lớp cũ/mới).