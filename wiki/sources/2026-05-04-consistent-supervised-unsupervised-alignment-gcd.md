# Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery

- Date ingested: 2026-05-04
- Raw source: [Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../raw/Consistent%20Supervised-Unsupervised%20Alignment%20for%20Generalized%20Category%20Discovery.md)
- Type: paper clipping

## Summary
Bài báo đề xuất NC-GCD: một framework GCD lấy cảm hứng từ Neural Collapse, tiền-gán các prototype dạng Simplex ETF cố định để thống nhất mục tiêu tối ưu giữa dữ liệu có nhãn (known) và không nhãn (known+novel). Hai thành phần chính là Consistent ETF Alignment Loss (kết hợp supervised/unsupervised ETF alignment) và Semantic Consistency Matcher (SCM) để ổn định pseudo-label qua các vòng clustering. Kết quả báo cáo cho thấy cải thiện mạnh ở novel-category accuracy trên nhiều benchmark.

## Key Claims
- Nguyên nhân chính gây giảm hiệu năng novel ở GCD trước đây là mục tiêu tối ưu không nhất quán và thiếu ràng buộc hình học dẫn đến category confusion.
- Việc pre-assign fixed ETF prototypes tạo cấu trúc hình học tối ưu và nhất quán cho cả known/novel.
- SCM giảm dao động pseudo-label bằng ánh xạ một-một giữa các cụm ở các iteration liên tiếp và giữa cụm dự đoán với nhãn thật trong supervised alignment.
- NC-GCD đạt SOTA trên nhiều benchmark và cải thiện rõ rệt ở nhóm novel categories.

## Entities
- [NC-GCD](../entities/nc-gcd.md)
- [Semantic Consistency Matcher (SCM)](../entities/semantic-consistency-matcher.md)
- [Neural Collapse](../entities/neural-collapse.md)

## Concepts
- [Generalized Category Discovery (GCD)](../concepts/generalized-category-discovery.md)
- [Simplex ETF prototypes](../concepts/simplex-etf-prototypes.md)
- [Consistent supervised-unsupervised alignment](../concepts/consistent-supervised-unsupervised-alignment.md)

## Contradictions / Uncertainty
- Clipping chưa kèm toàn bộ appendix chi tiết triển khai code nên chưa xác nhận đầy đủ sensitivity ngoài các bảng được trích.
- Một số so sánh backbone (CLIP) có ghi chú tác giả tự reproduce baseline, cần thận trọng khi so ngang tuyệt đối.

## Actionable Follow-ups
- Ingest thêm paper nền: SimGCD, CMS, SelEx để tạo comparative matrix nhất quán trong wiki.
- Tạo trang query tổng hợp “GCD methods by prototype strategy” sau khi có thêm 3–5 nguồn.

## Sources
- Raw: [Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../raw/Consistent%20Supervised-Unsupervised%20Alignment%20for%20Generalized%20Category%20Discovery.md)
