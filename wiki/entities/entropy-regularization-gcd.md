# Entropy regularization (GCD)

## Overview
Regularizer tối đa hóa entropy dự đoán trung bình trên batch nhằm chống collapse/imbalance trong pseudo-label distribution.

## Relevance
Trong SimGCD, đây là thành phần then chốt để giảm bias Old/New và bias phân phối theo lớp, đồng thời tăng robustness khi K không chuẩn.

## Related Concepts
- [Prediction bias in GCD](../concepts/prediction-bias-in-gcd.md)
- [Self-distillation pseudo-labeling](../concepts/self-distillation-pseudo-labeling.md)

## Related Sources
- [Parametric Classification for Generalized Category Discovery: A Baseline Study](../sources/2026-05-04-parametric-classification-gcd-baseline-study.md)

## Open Questions
- Làm sao chọn cường độ regularization theo loại phân phối dữ liệu (balanced vs long-tailed)?

## Sources
- [Parametric Classification for Generalized Category Discovery: A Baseline Study](../sources/2026-05-04-parametric-classification-gcd-baseline-study.md)
