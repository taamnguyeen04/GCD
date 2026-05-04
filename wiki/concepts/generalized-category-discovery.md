# Generalized Category Discovery (GCD)

## Definition
Bài toán học biểu diễn/phân loại trong bối cảnh có một phần dữ liệu known có nhãn và phần dữ liệu unlabeled chứa cả known lẫn novel categories, mục tiêu là giữ chính xác ở known đồng thời khám phá novel.

## Why It Matters
GCD phản ánh bối cảnh open-world thực tế hơn so với giả định đóng của supervised/semi-supervised truyền thống.

## Evidence
Nguồn mô tả rõ setting \(D=D^l\cup D^u\), tập nhãn known/novel, và mục tiêu đồng thời cho old/new categories.

## Related Entities
- [NC-GCD](../entities/nc-gcd.md)
- [Semantic Consistency Matcher (SCM)](../entities/semantic-consistency-matcher.md)

## Open Questions
- Chiến lược ước lượng số cụm K nào ổn định nhất khi không có GT K?

## Sources
- [Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../sources/2026-05-04-consistent-supervised-unsupervised-alignment-gcd.md)
