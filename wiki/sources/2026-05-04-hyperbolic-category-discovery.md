# Hyperbolic Category Discovery

- Date ingested: 2026-05-04
- Raw source: [Hyperbolic Category Discovery](../../paper/raw/Hyperbolic%20Category%20Discovery.md)
- Type: paper clipping

## Summary
Bài báo đề xuất HypCD, một framework đưa GCD từ không gian Euclidean/spherical sang hyperbolic (Poincaré ball) để học biểu diễn có ý thức thứ bậc (hierarchy-aware). HypCD ánh xạ đặc trưng bằng clipping + exponential map, tối ưu contrastive loss lai giữa distance-based và angle-based trong hyperbolic space, đồng thời dùng hyperbolic classifier cho nhánh parametric. Kết quả cho thấy cải thiện nhất quán khi gắn vào các baseline/sota như GCD, SimGCD, SelEx.

## Key Claims
- Euclidean/spherical space là chưa tối ưu để mã hóa cấu trúc phân cấp nội tại của dữ liệu GCD.
- Hyperbolic space có tăng trưởng thể tích theo cấp số mũ theo bán kính, phù hợp hơn cho cấu trúc phân cấp/tree-like.
- HypCD cải thiện rõ ACC (All/Old/New) trên cả fine-grained (SSB) và generic datasets khi áp vào nhiều phương pháp nền.
- Khoảng cách hiệu năng Old-New được thu hẹp trong nhiều thiết lập, gợi ý cải thiện knowledge transfer từ known sang unknown.

## Entities
- [HypCD](../entities/hypcd.md)
- [Poincaré ball model](../entities/poincare-ball-model.md)
- [SelEx](../entities/selex.md)

## Concepts
- [Hyperbolic geometry for GCD](../concepts/hyperbolic-geometry-for-gcd.md)
- [Hierarchy-aware representation learning](../concepts/hierarchy-aware-representation-learning.md)
- [Hybrid hyperbolic contrastive loss](../concepts/hybrid-hyperbolic-contrastive-loss.md)

## Contradictions / Uncertainty
- Hiệu quả phụ thuộc tuning theo dataset (curvature c, clipping r, alpha_d), chưa có cấu hình universal.
- Một phần kết quả SOTA (dấu *) là từ implementation của tác giả, cần cẩn trọng khi so sánh tuyệt đối với báo cáo gốc.

## Actionable Follow-ups
- Tạo trang so sánh NC-GCD vs HypCD theo trục: geometry, prototype strategy, label-stability, sensitivity.
- Ingest thêm DebGCD, SelEx, SimGCD bản gốc để củng cố comparative evidence.

## Sources
- Raw: [Hyperbolic Category Discovery](../../paper/raw/Hyperbolic%20Category%20Discovery.md)
