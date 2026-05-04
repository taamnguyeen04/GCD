---
paper id: 2201.02609v2
title: Generalized Category Discovery
authors: [Sagar Vaze, Kai Han, Andrea Vedaldi, Andrew Zisserman]
publication date: 2022-01-07T18:58
abstract: "In this paper, we consider a highly general image recognition setting wherein, given a labelled and unlabelled set of images, the task is to categorize all images in the unlabelled set. Here, the unlabelled images may come from labelled classes or from novel ones. Existing recognition methods are not able to deal with this setting, because they make several restrictive assumptions, such as the unlabelled instances only coming from known - or unknown - classes, and the number of unknown classes being known a-priori. We address the more unconstrained setting, naming it 'Generalized Category Discovery', and challenge all these assumptions. We first establish strong baselines by taking state-of-the-art algorithms from novel category discovery and adapting them for this task. Next, we propose the use of vision transformers with contrastive representation learning for this open-world setting. We then introduce a simple yet effective semi-supervised $k$-means method to cluster the unlabelled data into seen and unseen classes automatically, substantially outperforming the baselines. Finally, we also propose a new approach to estimate the number of classes in the unlabelled data. We thoroughly evaluate our approach on public datasets for generic object classification and on fine-grained datasets, leveraging the recent Semantic Shift Benchmark suite. Project page at https://www.robots.ox.ac.uk/~vgg/research/gcd"
comments: "CVPR 22. Changes from pre-print highlighted in GitHub repo"
pdf: "[[paper/pdf/Generalized Category Discovery (2201.02609v2).pdf]]"
url: https://arxiv.org/abs/2201.02609v2
tags: []
---
### 1. Tóm tắt Cốt lõi

Bài báo định nghĩa bài toán GCD, dữ liệu chưa gán nhãn có thể thuộc lớp đã biết hoặc một lớp hoàn toàn mới. Giải pháp cốt lõi kết hợp Vision Transformer (ViT), học đối ngẫu (contrastive learning) và phân cụm k-means bán giám sát thay vì dùng bộ phân loại tham số, giúp vượt trội đáng kể so với các phương pháp trước đây.

---

### 2. Vấn đề Nghiên cứu

- **Bài toán**: Cần phân loại dữ liệu chưa gán nhãn $\mathcal{D}_{\mathcal{U}}$ vào các lớp đã biết $\mathcal{Y}_{\mathcal{L}}$ hoặc lớp mới $\mathcal{Y}_{\mathcal{U}} \setminus \mathcal{Y}_{\mathcal{L}}$, đồng thời tự động ước lượng tổng số lớp (số cụm $k$).
- **Hạn chế của phương pháp cũ**: Các mô hình Novel Category Discovery (NCD) giả định sai lệch rằng toàn bộ dữ liệu chưa nhãn đều là lớp mới, và việc sử dụng các hàm phân loại (parametric classifiers) khiến chúng dễ bị overfit vào tập dữ liệu đã có nhãn.

---

### 3. Phương pháp Đề xuất

- **Học biểu diễn**: Dùng ViT-B-16 (pre-train bằng DINO), fine-tune bằng hàm mất mát đối ngẫu (contrastive loss) kết hợp cả có giám sát và không giám sát để biểu diễn đặc trưng mạnh mẽ hơn.
- **Phân loại phi tham số**: Bỏ hoàn toàn Linear Head. Sử dụng thuật toán k-means bán giám sát trực tiếp trên không gian đặc trưng, ép các điểm có nhãn cùng lớp phải rơi vào chung một cụm.
- **Ước lượng $k$**: Đánh giá heuristic bằng cách chạy k-means trên toàn bộ dữ liệu và chọn $k$ mang lại độ chính xác cao nhất trên tập dữ liệu con có nhãn.

---

### 4. Thực nghiệm & Kết quả

- **Dữ liệu**: Generic (CIFAR10/100, ImageNet-100) và Fine-grained/Long-tailed (CUB, Stanford Cars, Herbarium19, FGVC-Aircraft).
- **Metric**: Đo bằng Clustering Accuracy. Phương pháp đề xuất vượt baseline ~9.3% (tuyệt đối) trên tập Generic và 8.9% (tuyệt đối) trên tập Fine-grained khó.
- **Đánh giá ước lượng $k$**: Có sai số <10% với ảnh thường, nhưng còn hạn chế (~18.9% sai số) với bộ dữ liệu fine-grained.
