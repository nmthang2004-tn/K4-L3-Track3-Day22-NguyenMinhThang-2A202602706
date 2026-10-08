# Trạng thái bằng chứng nộp bài

Đã nhận ZIP `lab22-NB0-NB4-NB3b.zip` và notebook `Lab22_DPO_T4.ipynb` đã chạy. Đã nhập đủ cấu hình SFT/reference/DPO, Parquet, split fingerprint, kết quả đánh giá và ảnh. NB3b có đủ năm biến thể cùng metrics, cấu hình và ảnh. Không còn thiếu file bằng chứng cho phạm vi NB0–NB4 và NB3b.

Bài phản tư dùng số liệu mới: reward accuracy DPO 72%, held-out win rate 47% (CI 40–54%). NB0 có công thức đã điền nhưng output còn cũ; đã chạy lại trên CPU và lưu output thực tế trong notebook cùng `NB0_CPU_CHECK.txt`. Các output GPU từ Colab giữ nguyên.

Đường dẫn Colab `/content/lab22/models/sft-merged` trong adapter config được đổi sang đường dẫn tương đối `models/sft-merged` để chuyển repo sang Windows. Chỉ trường đường dẫn này thay đổi; mã băm nguồn và danh sách thay đổi ghi trong `artifact_import.json`. Verifier không bị sửa hoặc bỏ kiểm tra. `.gitignore` cho phép commit config mô hình và các biến thể; vẫn chặn trọng số và tokenizer lớn.

Notebook còn giữ một lỗi ở NB5 do thiếu `offload_dir`; NB5 không nằm trong phạm vi báo cáo. Không có kết quả NB5–NB7, β-sweep, giám khảo API hoặc HF Hub được nhận là đã hoàn thành.

Chương trình của target `make verify` được chạy bằng `python scripts/verify.py` vì Windows chưa có GNU Make. Kết quả cuối ở `VERIFY_RESULT.txt`. Trọng số không được tải từ Colab; lần kiểm tra này xác minh bằng chứng nộp bài, không phải chạy lại huấn luyện GPU trên Windows.

Các thay đổi chưa commit/push. Khi nộp GitHub, cần commit notebook có output, config, Parquet, metrics, ảnh và bài phản tư. Không cần gửi thêm file cho phạm vi hiện tại.
