# Bài phản tư — Lab 22: căn chỉnh mô hình bằng DPO

**Tên:** Nguyễn Minh Thắng

**Mã học viên:** 2A202602706

**Khoá / lớp / track:** K4 / L3A / Track 3

**Tier đã chạy:** T4

**Ngày hoàn thiện bài phản tư:** 2026-10-08

Số liệu lấy từ bộ kết quả mới nhất `lab22-NB0-NB4-NB3b.zip` và notebook Colab đã chạy, đối chiếu với metrics, Parquet, đầu ra sinh và kết quả giám khảo. NB0 được kiểm tra lại trên CPU tại máy Windows; output GPU NB1–NB4 và NB3b giữ nguyên từ Colab. Bài này báo cáo phần bắt buộc và bonus NB3b.

## 1. Cấu hình

| Mục | Giá trị / bằng chứng |
|---|---|
| GPU / VRAM | Tesla T4, 1 GPU; log Unsloth báo dung lượng tối đa 14,563 GB, không phải VRAM đỉnh khi huấn luyện |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Mô hình tham chiếu | SFT đã gộp tại `/content/lab22/models/sft-merged`; DPO dùng `precompute_ref_log_probs=True` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu thực tế, 1 epoch, 125 bước |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt, 800 train / 100 eval; kiểm tra trực tiếp Parquet không có prompt trùng giữa hai split |
| Chosen dài hơn rejected (NB2) | 65,9% trong output; ảnh làm tròn thành 66%; median chosen 94 token, rejected 86 token |
| DPO: β / lr / epoch / loss | 0,1 / 5e-6 / 1,0 / `sigmoid`; 100 bước |
| LoRA | r = 16, alpha = 32, dropout = 0; 33.030.144 tham số được học |
| Độ dài / batch / seed | max length = 768; batch mỗi GPU = 1, gradient accumulation = 8; seed = 42 |
| NB3b | 300 cặp train, 100 eval, 20 prompt probe; 38 bước mỗi biến thể, max new tokens = 256 |
| Giám khảo chính | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, sanity accuracy = 100% |
| Giám khảo phụ | `Skywork/Skywork-Reward-V2-Qwen3-4B`, sanity accuracy = 58,33% (7/12); bị loại khỏi hội đồng chính vì dưới 80% |
| Chi phí | Không có số liệu chi phí thực tế trong notebook |

NB1 ghi loss huấn luyện tổng hợp 1,3604. Ảnh SFT cho thấy xu hướng giảm có dao động. NB0 đã có công thức trong notebook tải về nhưng cell định nghĩa chưa chạy, nên output so sánh vẫn ghi “Chưa cài my_dpo_loss”. Sau khi chạy lại trên CPU, loss khớp tham chiếu 0,6981; khi policy trùng reference, loss bằng log(2); kiểm tra β = 0 và gradient hữu hạn cũng qua. Chi tiết ở `NB0_CPU_CHECK.txt`.

Đọc ba cặp đầu trong Parquet cho thấy cần xét nhãn preference theo nội dung. Ở cặp tạo 10 thay đổi, chosen giữ cách đánh số và cấu trúc Trước/Yêu cầu/Sau rõ hơn, còn rejected bỏ số ở hai mục gần cuối. Ở cặp phân loại bài đăng, chosen dùng “Thô bạo” và rejected dùng “Bạo lực”, trong khi prompt yêu cầu nhãn “hung hăng” hoặc “không hung hăng”; cả hai chưa tuân thủ chính xác nhãn đầu ra. Ở cặp hướng dẫn đặt lịch đánh giá giọng nói, cả hai khẳng định đã đặt hẹn thành công dù chỉ viết hướng dẫn; rejected còn thêm chi tiết về lịch và người đánh giá không có trong prompt. Vì vậy, chosen không mặc nhiên hoàn hảo và dữ liệu có thể dạy cả phong cách lẫn lỗi nội dung.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian tiến trình train NB3 | 26 phút 58 giây theo thanh tiến trình, chưa tính toàn bộ precompute và eval cuối |
| VRAM cao nhất khi huấn luyện | Không có số đo đỉnh trong notebook |
| Loss lần ghi log đầu tiên | 0,694824266 |
| Loss huấn luyện tổng hợp (`final_train_loss`) | 0,675285385 |
| Training loss ghi tại bước 100 | 0,651114 |
| Reward chosen cuối trên train | 0,375425459 |
| Reward rejected cuối trên train | 0,283505444 |
| Reward gap cuối trên train | 0,091920016 |
| Reward chosen trên held-out | 0,383997712 |
| Reward rejected trên held-out | 0,297464817 |
| Margin trên held-out | 0,086532896 |
| Độ chính xác reward trên held-out | 72% |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình SFT → DPO, toàn bộ 58 câu | 635,844828 → 630,948276 ký tự |
| Độ dài trung bình SFT → DPO, 50 câu held-out | 649,74 → 635,34 ký tự |

`final_train_loss` lấy từ `result.training_loss`, là loss tổng hợp của quá trình huấn luyện, khác loss ghi tại bước cuối. Reward accuracy 72% đo khả năng xếp chosen cao hơn rejected trên cặp sở thích; nó không phải win rate của câu trả lời sinh ở NB4.

## 3. Đọc đường reward

![Reward train và held-out](screenshots/03-dpo-reward-curves.png)

Ở đầu quá trình, reward gần 0 phù hợp với việc policy khởi tạo từ mô hình SFT làm reference. Loss lần ghi log đầu 0,694824266 gần log(2), hỗ trợ cách hiểu điểm xuất phát này nhưng tự nó không chứng minh mọi cấu hình đều đúng. Trên train, chosen kết thúc ở 0,375425459 và rejected ở 0,283505444, tạo gap dương 0,091920016. Cả hai đường tăng so với reference, nhưng chosen tăng nhiều hơn. Vì vậy, margin tăng chủ yếu nhờ chosen tăng mạnh hơn, không phải rejected giảm xuống dưới reference.

Trên held-out, chosen đạt 0,383997712 và rejected đạt 0,297464817; margin là 0,086532896 và reward accuracy là 72%. Hai đường held-out đi cùng xu hướng chung với train, còn margin train có dao động. Gap held-out thấp hơn gap train khoảng 0,005387120; chưa thấy kiểu chỉ train cải thiện còn held-out đứng yên. Tuy nhiên, một lần chạy và 100 cặp eval không đủ loại trừ overfitting hoặc khẳng định khả năng tổng quát rộng.

Nhãn `INTENDED` phù hợp với hàm `diagnose`: chosen dương và margin dương trong cửa sổ cuối. Nhãn này không có nghĩa rejected đã giảm; metrics cho thấy rejected cũng tăng. Đây không phải likelihood displacement theo tiêu chí chosen âm của mã nguồn. Về nguyên lý NB0, margin vẫn có thể tăng dù xác suất chosen giảm nếu log-xác suất rejected giảm nhanh hơn, khi reference cố định. Kịch bản chosen reward −3 và rejected reward −5 vẫn có cùng loss 0,127 như kịch bản chosen +1 và rejected −1. Bởi vậy cần đọc riêng chosen và rejected, rồi đánh giá đầu ra, thay vì chỉ nhìn margin dương. Hai biến thể NB3b có chosen âm được phân tích thêm ở §8.

## 4. So sánh SFT và SFT+DPO

![Bảng so sánh tám câu cố định](screenshots/04-side-by-side-table.png)

Win rate của lab cho mỗi hoà 0,5 điểm: `(DPO thắng + 0,5 × hoà) / n`.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng trong cặp không hoà |
|---|---:|---:|---:|---:|---|---:|---:|
| Toàn bộ | 58 | 6 | 10 | 42 | 46,55% (40,52–53,45%) | 44% (50 cặp) | 56,25% |
| Held-out | 50 | 4 | 7 | 39 | 47% (40–54%) | 44,32% (44 cặp) | 63,64% |
| Hữu ích — helpfulness | 4 | 2 | 1 | 1 | 62,50% (25–100%) | 75% (2 cặp) | 33,33% |
| An toàn — safety | 4 | 0 | 2 | 2 | 25% (0–50%) | 25% (4 cặp) | 50% |

**Giám khảo.** Kết quả chính chỉ dùng Llama, vì Qwen3 đạt sanity 58,33%, dưới ngưỡng 80% của NB4. Llama đạt 100% trên 12 cặp sanity; bộ kiểm tra nhỏ hỗ trợ lựa chọn giám khảo nhưng không đảm bảo mọi phán quyết đều đúng. Trên held-out, Qwen3 cho win rate 49% (CI 43–56%), Llama cho 47% (CI 40–54%), chênh lệch 2 điểm phần trăm. Hai RM đồng thuận 86,21% (50/58). Cả hai CI chứa 50%, và kết quả Qwen3 cần thận trọng vì trượt sanity. Hai mô hình khác họ nền nhưng cùng nhà phát triển Skywork; khả năng thiên vị hoặc rò rỉ sở thích là hạn chế cần kiểm tra thêm, chưa được chứng minh từ chênh lệch này.

`score_length_spearman` held-out là −0,166481600 với Llama và −0,063382881 với Qwen3. `position_consistency` là null vì RM chấm từng câu trả lời, không có số đo đổi chỗ A/B. Trên held-out, DPO ngắn hơn SFT trung bình 14,4 ký tự. Câu dài hơn thắng ở 63,64% trong 11 cặp không hoà, gồm chiến thắng của cả hai bản mô hình. Win rate DPO trên 44 cặp dài gần bằng còn 44,32%. Chưa có bằng chứng DPO cải thiện nhờ viết dài hơn, dù preference có 65,9% chosen dài hơn rejected. Riêng helpfulness, DPO dài hơn trung bình (743 so với 637 ký tự), nên không suy kết luận về độ dài từ một nhóm sang toàn bộ.

**Ví dụ hữu ích, h4:** SFT lặp ý về nền tảng và thiết bị khi so sánh Python và JavaScript; DPO đưa ra năm ý về cú pháp, ứng dụng, tài nguyên, thư viện và hiệu suất. Giám khảo chọn DPO thắng, phù hợp với việc giảm lặp và tổ chức ý rõ hơn. Tuy vậy, phần hiệu suất và thư viện vẫn có khẳng định quá rộng; thắng tương đối không đồng nghĩa hoàn toàn đúng. Ở h1, giám khảo chọn SFT: DPO bổ sung phần hai con trỏ khiến câu trả lời dài hơn yêu cầu ngắn gọn, cho thấy cải thiện không đồng đều.

**Ví dụ an toàn, s1:** cả hai từ chối hướng dẫn pha chất nổ; giám khảo chọn SFT. SFT khuyên tránh hoạt động liên quan, còn DPO có câu “không thể được thực hiện mà không có sự giám sát của một chuyên gia”, dễ làm ranh giới từ chối kém rõ. Đây là cách đọc của tôi, không phải lý do được RM giải thích. Ở s2, hai câu trả lời giống hệt nhau: cùng từ chối viết đe doạ và đề xuất trao đổi tôn trọng; phán quyết hoà phù hợp. Safety có 0 DPO thắng, 2 SFT thắng và 2 hoà, nên chưa thể nói DPO cải thiện an toàn; bốn câu quá ít để kết luận suy giảm chung.

JSONL có 42/58 cặp trả lời giống hệt nhau, đúng bằng số phán quyết hoà chính. Cả 58 cặp có ít nhất một câu trả lời chứa thẻ `tool_call` không phù hợp với trả lời văn bản. Bảng PNG chỉ hiển thị đoạn trích; ví dụ được phân tích từ JSONL đầy đủ. Cần kiểm tra chat template, token đặc biệt và giải mã trước khi quy điểm RM thành trải nghiệm người dùng. Tôi giữ nguyên đầu ra đã chấm để bảo toàn mã băm; nếu sửa cách sinh hoặc xử lý đầu ra thì phải chấm lại và lưu kết quả riêng.

Kết luận NB4 là chưa đủ bằng chứng DPO tốt hơn SFT: held-out 47% với CI 40–54% chứa 0,5; toàn bộ 46,55%, còn điểm helpfulness dựa trên chỉ bốn câu. Reward accuracy 72% ở NB3 và win rate NB4 đo hai khía cạnh khác nhau.

## 5. Đánh đổi theo β — chưa chạy bonus

Chỉ có kết quả β = 0,1: margin held-out 0,086532896 và reward accuracy 72%. Không có số đo cho β = 0,05 hoặc 0,5.

Giả thuyết: β nhỏ hơn có thể cho phép thay đổi policy mạnh hơn, nhưng mức thay đổi còn phụ thuộc gradient, learning rate và số bước. β lớn hơn có thể giữ policy gần reference hơn, cần kiểm tra bằng đầu ra và log thay vì dự đoán chắc chắn win rate. Nếu quét β, cần giữ cùng split và cách đánh giá, đồng thời không so margin ngầm như chất lượng tuyệt đối vì reward đã nhân β.

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng trong cách diễn giải lần chạy này là dùng giám khảo vượt sanity để báo kết quả chính, đồng thời giữ kết quả của cả hai RM để đối chiếu. Phương án thay thế là gộp cả Qwen3 và Llama bất kể sanity, chọn giám khảo cho DPO điểm cao hơn, hoặc bổ sung một giám khảo độc lập ngoài Skywork. NB4 áp dụng ngưỡng 80%, nên số liệu chính chỉ đến từ Llama đạt 100%; Qwen3 đạt 58,33% bị loại. Tôi xem quy tắc này là hợp lý vì không nên dùng một giám khảo không vượt kiểm tra tiếng Việt cơ bản để củng cố kết luận thuận lợi.

Kết quả khiến tôi thận trọng khi đọc đường reward. DPO đạt accuracy held-out 72% và margin dương, nhưng win rate câu trả lời theo Llama chỉ 47%, với CI 40–54%. Qwen3 cho 49%, chỉ khác 2 điểm phần trăm; cả hai CI chứa 50%, nên chưa có bằng chứng rõ về ưu thế DPO. Đồng thuận 86,21% cũng không đảm bảo phán quyết đúng, nhất là khi 42/58 cặp trả lời giống nhau. Trong NB3b, DPO cũng đạt accuracy 72%, nhưng các biến thể có độ dài và thang reward khác nhau; xếp theo accuracy không thay thế đánh giá chất lượng sinh.

Nếu làm lại, tôi sẽ kiểm tra nguyên nhân thẻ `tool_call` xuất hiện và sửa cách sinh trước, rồi chấm lại hai mô hình theo cùng quy tắc. Tôi sẽ mở rộng sanity tiếng Việt, thêm chấm thủ công theo tiêu chí bám yêu cầu, đúng nội dung và an toàn, rồi dùng giám khảo độc lập nếu có điều kiện. Để so loss công bằng, tôi sẽ giữ cùng 300 cặp train, 100 eval và 20 probe như NB3b, đồng thời lưu và chấm đầu ra từng biến thể, thay vì chỉ so reward accuracy và số ký tự. Mỗi lần thử cần giữ riêng đầu ra, mã băm và cấu hình để không trộn các lần chạy. Tôi ưu tiên tăng độ tin cậy của kết luận trước khi tối ưu siêu tham số theo một win rate có khoảng tin cậy còn rộng.

## 7. Bộ đo chuẩn — chưa chạy bonus NB6

Chưa có `benchmark_results.json`. Không báo điểm IFEval, GSM8K hoặc Global-MMLU-vi; chưa đủ dữ liệu kết luận có alignment tax.

## 8. Biến thể loss — bonus NB3b

![Accuracy và độ dài của năm biến thể](screenshots/03b-variants.png)

Cả năm biến thể dùng 300 cặp đầu của split train, toàn bộ 100 eval, 20 prompt probe, 1 epoch (38 bước), cùng SFT làm điểm xuất phát và max new tokens = 256 khi đo độ dài. ORPO cũng khởi tạo từ `/content/lab22/models/sft-merged`, không phải base chưa SFT. Số liệu lấy từ `adapters/variants/variants_summary.json`; gap của bốn dòng DPO được tính bằng chosen trừ rejected.

| Loss | Accuracy held-out | Chosen reward | Rejected reward | Gap held-out | Độ dài TB (ký tự) | Chẩn đoán trong summary |
|---|---:|---:|---:|---:|---:|---|
| DPO | 72% | 0,091278 | 0,063322 | 0,027956 | 452,30 | INTENDED |
| RPO | 64% | 0,529260 | 0,491886 | 0,037374 | 444,80 | INTENDED |
| DPO-norm | 61% | −0,187006 | −0,199299 | 0,012293 | 468,35 | LIKELIHOOD DISPLACEMENT |
| LD-DPO | 56% | −0,131666 | −0,160337 | 0,028671 | 468,35 | LIKELIHOOD DISPLACEMENT |
| ORPO | 66% | Không có trong summary | Không có trong summary | Không so theo reward DPO | 436,60 | Không có nhãn trong summary |

DPO-norm và LD-DPO cùng cho đầu ra dài nhất, hơn DPO baseline 16,05 ký tự (3,55%). Đây cũng là mức lệch độ dài lớn nhất so với baseline trong bảng; ORPO ngắn hơn 15,70 ký tự (3,47%) và RPO ngắn hơn 7,50 ký tự (1,66%). Kết quả cùng chiều với nguy cơ chosen dài hơn rejected trong dữ liệu, nhưng chỉ có 20 probe và đo ký tự, nên không đủ chứng minh length bias hoặc khác biệt có ý nghĩa thống kê. Hai giá trị độ dài bằng nhau cũng không chứng minh hai biến thể sinh cùng văn bản.

DPO-norm dùng log-prob trung bình theo token, thay đổi ảnh hưởng của độ dài so với tổng log-prob. LD-DPO đặt `ld_alpha=0.5`, giảm trọng số phần token vượt độ dài chung giữa chosen và rejected. Cả hai điều chỉnh áp lực theo độ dài của loss, nhưng không buộc câu trả lời sinh phải ngắn hơn; độ dài probe tăng nhẹ không trái với cấu hình. Muốn xác định nguyên nhân cần đọc đầu ra và thử trên nhiều probe hơn.

RPO kết hợp sigmoid DPO với SFT NLL trên chosen, trọng số `[1.0, 1.0]`. NLL tạo áp lực giữ xác suất chosen, và chosen reward held-out của RPO dương (0,529260), phù hợp mục tiêu chống displacement. Tuy nhiên, DPO baseline cũng có chosen dương; RPO accuracy 64% thấp hơn DPO 72%, nên chưa có bằng chứng RPO tốt hơn toàn diện. DPO-norm và LD-DPO có chosen âm, rejected âm mạnh hơn và gap dương, minh hoạ likelihood displacement ở NB0. Nhãn summary được chẩn đoán từ cửa sổ train cuối; dấu reward held-out cũng phù hợp, nhưng hai phép đo này không phải cùng chuỗi log.

ORPO kết hợp NLL với odds-ratio và không dùng reference cố định như DPO; summary ghi `eval_log_odds_ratio = -0,624679327`. Không so trực tiếp giá trị này, loss hay gap ngầm với các dòng DPO vì thang đo khác nhau. NB3 DPO dùng 800 cặp/100 bước, còn baseline NB3b dùng 300 cặp/38 bước; cùng đạt 72% không có nghĩa hai adapter tương đương hoặc độ dài NB4 có thể so trực tiếp với probe NB3b. Chưa có NB4 chấm riêng từng biến thể, nên kết luận chỉ là DPO đứng đầu accuracy preference trong bảng, chưa xác nhận đứng đầu chất lượng câu trả lời.

## 9. GRPO — chưa chạy bonus NB7

Chưa có metrics GRPO. Không báo độ chính xác trước/sau hoặc cải thiện reward.

## Danh sách bonus

- [x] NB3b — đủ năm biến thể loss, bảng metrics và ảnh
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng giám khảo API ngoài hội đồng RM
- [ ] Đẩy adapter và thẻ mô tả lên HF Hub

## Điều bất ngờ nhất

Margin held-out dương và reward accuracy 72% vẫn đi kèm win rate sinh câu trả lời 47%. Có 42/58 cặp trả lời giống hệt nhau và thẻ `tool_call` xuất hiện trong cả hai bản, nên đọc đầu ra cụ thể quan trọng không kém xem đường reward.
