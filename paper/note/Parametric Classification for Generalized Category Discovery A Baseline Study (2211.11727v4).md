---
paper id: 2211.11727v4
title: "Parametric Classification for Generalized Category Discovery: A Baseline Study"
authors: [Xin Wen, Bingchen Zhao, Xiaojuan Qi]
publication date: 2022-11-21T18:47
abstract: "Generalized Category Discovery (GCD) aims to discover novel categories in unlabelled datasets using knowledge learned from labelled samples. Previous studies argued that parametric classifiers are prone to overfitting to seen categories, and endorsed using a non-parametric classifier formed with semi-supervised k-means. However, in this study, we investigate the failure of parametric classifiers, verify the effectiveness of previous design choices when high-quality supervision is available, and identify unreliable pseudo-labels as a key problem. We demonstrate that two prediction biases exist: the classifier tends to predict seen classes more often, and produces an imbalanced distribution across seen and novel categories. Based on these findings, we propose a simple yet effective parametric classification method that benefits from entropy regularisation, achieves state-of-the-art performance on multiple GCD benchmarks and shows strong robustness to unknown class numbers. We hope the investigation and proposed simple framework can serve as a strong baseline to facilitate future studies in this field. Our code is available at: https://github.com/CVMI-Lab/SimGCD."
comments: "v3: ICCV'23 version; v4: updated the dataset table"
pdf: "[[paper/pdf/Parametric Classification for Generalized Category Discovery A Baseline Study (2211.11727v4).pdf]]"
url: https://arxiv.org/abs/2211.11727v4
tags: []
---
### 1. Tóm tắt

Bài báo chứng minh sự thất bại của các bộ phân loại tham số trong bài toán GCD bắt nguồn từ nhãn giả kém chất lượng và dự đoán thiên lệch. Đề xuất phwuong pháp SimGCD, một baseline đơn giản dùng self-distillation và entropy regularisation, vượt qua các phương pháp SOTA dựa trên k-means phức tạp và giảm đáng kể thời gian tính toán.

### 2. Vấn đề Nghiên cứu

- **Nhiệm vụ:** Vừa phân loại các lớp đã biết, vừa khám phá các lớp mới từ dữ liệu không nhãn.
- **Hạn chế trước đây:** Phương pháp phi tham số (k-means) tốn quá nhiều tài nguyên tính toán. Trong khi đó, bộ phân loại tham số (như UNO) lại bị thiên lệch: thường xuyên đoán nhầm lớp mới thành lớp cũ ("False Old") và tạo ra phân phối nhãn giả mất cân bằng.

---
### 3. Giải pháp Đề xuất (SimGCD)

- **Kiến trúc:** Dùng bộ phân loại tham số gắn trực tiếp sau feature backbone.
- **Tự chưng cất (Self-Distillation):** Tạo nhãn giả mềm từ dự đoán của một góc nhìn biến đổi (augmented view) để giám sát góc nhìn khác.
- **Điều chuẩn Entropy (Entropy Regularisation):** Cực đại hóa entropy trung bình trên từng batch để ép mô hình phân bổ dự đoán đều hơn, loại bỏ sự thiên lệch.

---
### 4. Dữ liệu & Kết quả

- **Thử nghiệm:** Chạy trên đa dạng dữ liệu CIFAR, ImageNet, CUB, Herbarium 19.
- **Kết quả:** Cải thiện khoảng 10% độ chính xác trên các lớp mới so với SOTA (GCD, UNO+). Tốc độ suy luận nhanh gấp nhiều lần (9.5 phút so với 36 phút của GCD trên ImageNet-100).
