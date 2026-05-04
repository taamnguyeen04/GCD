---
paper id: 2308.05017v1
title: "When and How Does Known Class Help Discover Unknown Ones? Provable Understanding Through Spectral Analysis"
authors: [Yiyou Sun, Zhenmei Shi, Yingyu Liang, Yixuan Li]
publication date: 2023-08-09T15:27
abstract: "Novel Class Discovery (NCD) aims at inferring novel classes in an unlabeled set by leveraging prior knowledge from a labeled set with known classes. Despite its importance, there is a lack of theoretical foundations for NCD. This paper bridges the gap by providing an analytical framework to formalize and investigate when and how known classes can help discover novel classes. Tailored to the NCD problem, we introduce a graph-theoretic representation that can be learned by a novel NCD Spectral Contrastive Loss (NSCL). Minimizing this objective is equivalent to factorizing the graph's adjacency matrix, which allows us to derive a provable error bound and provide the sufficient and necessary condition for NCD. Empirically, NSCL can match or outperform several strong baselines on common benchmark datasets, which is appealing for practical usage while enjoying theoretical guarantees."
comments: ICML 2023
pdf: "[[paper/pdf/When and How Does Known Class Help Discover Unknown Ones Provable Understanding Through Spectral Analysis (2308.05017v1).pdf]]"
url: https://arxiv.org/abs/2308.05017v1
tags: []
---
### 1. Tóm tắt Cốt lõi

Bài báo xây dựng nền tảng lý thuyết toán học đầu tiên cho bài toán NCD thông qua biểu diễn đồ thị và phân tích phổ. Hàm mất mát mới NSCL (NCD Spectral Contrastive Loss) được đề xuất giúp giải thích cơ chế chuyển giao tri thức từ lớp đã biết sang lớp chưa biết và đạt hiệu suất vượt trội.

---
### 2. Vấn đề Nghiên cứu (Problem Statement)

- **Bài toán**: Giải quyết NCD - gom cụm dữ liệu chưa biết dựa trên tri thức từ dữ liệu đã dán nhãn thuộc các nhóm đã biết.
- **Hạn chế của SOTA cũ**: Các phương pháp trước đây thiếu nền tảng lý thuyết để trả lời câu hỏi cốt lõi: "khi nào và làm thế nào" dữ liệu đã biết thực sự mang lại lợi ích cho dữ liệu chưa biết.

---
### 3. Phương pháp & Giải pháp Đề xuất (Methodology & Proposed Solution)

- **Kiến trúc cốt lõi**: Mô hình hóa NCD bằng Đồ thị Tăng cường (Augmentation Graph) và huấn luyện bằng hàm mất mát Đối lập Phổ (NSCL). Tối thiểu hóa NSCL tương đương với việc phân rã SVD trên ma trận kề của đồ thị.
- **Đóng góp chính**: Cung cấp giới hạn sai số (error bound) toán học, chứng minh sai số phân lớp giảm về 0 khi không gian đặc trưng của dữ liệu đã biết "bao phủ" (covers) trọn vẹn không gian thiếu hụt (ignorance space) của dữ liệu mới.

---
### 4. Dữ liệu & Đánh giá Thực nghiệm (Datasets & Evaluation)

- **Dữ liệu**: Đánh giá trên CIFAR-10, CIFAR-100, tập mô phỏng 3D (toy example). 
- **Metric:** All Accuracy, Novel Accuracy, Known Accuracy.
- **Kết quả**: NSCL đánh bại mọi phương pháp tiền nhiệm, tiêu biểu là vượt ComEx 10.6% trên CIFAR-100-50.