# Parametric classification for GCD

## Definition
Thiết lập GCD trong đó classifier có tham số học được (prototype/classifier head), được tối ưu cùng representation thay vì chỉ gán nhãn bằng clustering non-parametric ở pha hậu xử lý.

## Why It Matters
Cho phép suy luận nhanh hơn và có thể học ranh giới phân lớp end-to-end, nhưng phụ thuộc mạnh vào chất lượng pseudo-label.

## Evidence
SimGCD cho thấy khi dùng self-distillation + entropy regularization, parametric approach đạt hiệu năng cạnh tranh/vượt non-parametric.

## Related Entities
- [SimGCD](../entities/simgcd.md)
- [GCD (CVPR 2022)](../entities/gcd-cvpr-2022.md)

## Open Questions
- Trong bối cảnh open-world có domain shift, parametric hay non-parametric ổn định hơn?

## Sources
- [Parametric Classification for Generalized Category Discovery: A Baseline Study](../sources/2026-05-04-parametric-classification-gcd-baseline-study.md)
