# Consistent supervised-unsupervised alignment

## Definition
Chiến lược thống nhất mục tiêu tối ưu cho dữ liệu có nhãn và không nhãn bằng cách kết hợp supervised ETF alignment và unsupervised ETF alignment trong cùng một hình học prototype cố định.

## Why It Matters
Giảm thiên lệch học về known categories và cải thiện khả năng tách novel categories trong GCD.

## Evidence
NC-GCD dùng hybrid loss: \(L_{ETF}=(1-\gamma)L_{ETF}^u+\gamma L_{ETF}^s\), đồng thời dùng SCM để ổn định ánh xạ nhãn.

## Related Entities
- [NC-GCD](../entities/nc-gcd.md)
- [Semantic Consistency Matcher (SCM)](../entities/semantic-consistency-matcher.md)

## Open Questions
- Trade-off tối ưu giữa old/new accuracy phụ thuộc vào \(\gamma,\beta,\alpha,T\) thế nào theo từng dataset family?

## Sources
- [Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../sources/2026-05-04-consistent-supervised-unsupervised-alignment-gcd.md)
