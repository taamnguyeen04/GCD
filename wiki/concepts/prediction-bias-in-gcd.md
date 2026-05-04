# Prediction bias in GCD

## Definition
Hiện tượng mô hình thiên lệch dự đoán trong GCD, gồm: (1) bias giữa Old/New classes và (2) bias phân phối dự đoán giữa các lớp.

## Why It Matters
Bias làm suy giảm khả năng khám phá novel categories và khiến pseudo-label self-training bị sai lệch tích lũy.

## Evidence
SimGCD phân tích lỗi cho thấy tỉ lệ “False Old” cao và phân phối per-class lệch; entropy regularization giúp giảm các lỗi này.

## Related Entities
- [SimGCD](../entities/simgcd.md)
- [Entropy regularization (GCD)](../entities/entropy-regularization-gcd.md)

## Open Questions
- Có thể kết hợp bias-aware objectives với structure-aware geometry (ETF/hyperbolic) để giảm bias mạnh hơn không?

## Sources
- [Parametric Classification for Generalized Category Discovery: A Baseline Study](../sources/2026-05-04-parametric-classification-gcd-baseline-study.md)
