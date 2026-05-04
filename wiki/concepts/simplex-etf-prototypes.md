# Simplex ETF prototypes

## Definition
Tập prototype cố định với cấu trúc equiangular tight frame trong không gian đặc trưng, đảm bảo chuẩn hóa và độ tương đồng cặp có cấu trúc.

## Why It Matters
Tạo mục tiêu hình học nhất quán để kéo cả supervised và unsupervised features về các cực tách biệt tốt, giảm category overlap.

## Evidence
NC-GCD pre-assign ETF prototypes cho toàn bộ categories (known + novel ước lượng) và dùng chúng làm neo cho cả hai loss alignment.

## Related Entities
- [Neural Collapse](../entities/neural-collapse.md)
- [NC-GCD](../entities/nc-gcd.md)

## Open Questions
- Khi K bị ước lượng lệch, cách điều chỉnh ETF prototype set động nào hiệu quả nhất?

## Sources
- [Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../sources/2026-05-04-consistent-supervised-unsupervised-alignment-gcd.md)
