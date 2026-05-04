# Hybrid hyperbolic contrastive loss

## Definition
Hàm mất mát contrastive trong hyperbolic space kết hợp đồng thời thành phần dựa trên khoảng cách hyperbolic và thành phần dựa trên góc (cosine), với trọng số điều phối theo tiến trình huấn luyện.

## Why It Matters
Việc phối hợp distance + angle giúp tối ưu ổn định hơn trong không gian hyperbolic so với chỉ dùng một loại similarity.

## Evidence
HypCD dùng alpha_d để trộn hai loss và báo cáo rằng cấu hình tối ưu phụ thuộc loại dữ liệu (fine-grained vs generic).

## Related Entities
- [HypCD](../entities/hypcd.md)
- [Poincaré ball model](../entities/poincare-ball-model.md)

## Open Questions
- Có thể tự động điều chỉnh alpha_d thay vì lịch tuyến tính cố định không?

## Sources
- [Hyperbolic Category Discovery](../sources/2026-05-04-hyperbolic-category-discovery.md)
