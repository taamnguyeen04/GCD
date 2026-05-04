# Self-distillation pseudo-labeling

## Definition
Chiến lược sinh pseudo-label mềm từ augmented view thứ hai (teacher temperature sắc hơn) để huấn luyện view hiện tại theo cross-entropy/distillation.

## Why It Matters
Cải thiện chất lượng pseudo-label so với self-labeling đơn giản, giúp joint training parametric classifier ổn định hơn trong GCD.

## Evidence
Trong SimGCD, thay self-label bằng self-distillation mang lại cải thiện lớn, đặc biệt ở New-class accuracy.

## Related Entities
- [SimGCD](../entities/simgcd.md)
- [Entropy regularization (GCD)](../entities/entropy-regularization-gcd.md)

## Open Questions
- Teacher temperature schedule tối ưu phụ thuộc dataset và backbone thế nào?

## Sources
- [Parametric Classification for Generalized Category Discovery: A Baseline Study](../sources/2026-05-04-parametric-classification-gcd-baseline-study.md)
