# Parametric Classification for Generalized Category Discovery: A Baseline Study

- Date ingested: 2026-05-04
- Raw source: [Parametric Classification for Generalized Category Discovery A Baseline Study](../../paper/raw/Parametric%20Classification%20for%20Generalized%20Category%20Discovery%20A%20Baseline%20Study.md)
- Type: paper clipping

## Summary
Bài báo tái thẩm định giả định rằng parametric classifier kém trong GCD và cho thấy vấn đề cốt lõi không phải do bản chất parametric mà do pseudo-label không đáng tin cậy gây thiên lệch dự đoán. Tác giả đề xuất SimGCD: one-stage parametric framework kết hợp representation learning kiểu GCD, self-distillation cho unlabeled data, và entropy regularization để giảm bias Old/New và bias phân phối theo lớp. Kết quả cho thấy SimGCD đạt hiệu năng mạnh trên nhiều benchmark, đồng thời có độ bền tốt hơn khi số lớp K không chính xác.

## Key Claims
- Parametric classifier có thể hoạt động rất tốt cho GCD nếu xử lý đúng chất lượng pseudo-label.
- Hai bias quan trọng được chỉ ra: thiên lệch dự đoán về Old classes và phân phối pseudo-label mất cân bằng toàn cục.
- Entropy regularization giúp giảm hai dạng bias và tăng khả năng nhận diện novel classes.
- SimGCD đạt SOTA tại thời điểm công bố và có chi phí suy luận thấp hơn non-parametric k-means-based pipelines.

## Entities
- [SimGCD](../entities/simgcd.md)
- [GCD (CVPR 2022)](../entities/gcd-cvpr-2022.md)
- [Entropy regularization (GCD)](../entities/entropy-regularization-gcd.md)

## Concepts
- [Parametric classification for GCD](../concepts/parametric-classification-for-gcd.md)
- [Prediction bias in GCD](../concepts/prediction-bias-in-gcd.md)
- [Self-distillation pseudo-labeling](../concepts/self-distillation-pseudo-labeling.md)

## Contradictions / Uncertainty
- Mức cải thiện Old/New trade-off có thể phụ thuộc mạnh vào đặc trưng dữ liệu (fine-grained vs generic vs long-tailed).
- Entropy regularization quá mạnh có thể gây lệch phân phối dự đoán ở một số tập class-balanced.

## Actionable Follow-ups
- Tạo trang so sánh theo timeline: GCD -> SimGCD -> NC-GCD -> HypCD.
- Ingest paper GCD gốc để chuẩn hóa đối chiếu claim và metric setting.

## Sources
- Raw: [Parametric Classification for Generalized Category Discovery A Baseline Study](../../paper/raw/Parametric%20Classification%20for%20Generalized%20Category%20Discovery%20A%20Baseline%20Study.md)
