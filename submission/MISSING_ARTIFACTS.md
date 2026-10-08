# File cần bổ sung từ phiên Colab đã chạy

Đã có bốn PNG bắt buộc, `dpo_metrics.json`, `side_by_side.jsonl`, `judge_summary.json`, `judge_results_rm.json` và bài phản tư đã viết theo các kết quả này.

Máy Windows chưa có lệnh `make`. Đã chạy trực tiếp `python scripts/verify.py`, đúng lệnh target `make verify` gọi. Kết quả: exit code 1; còn thiếu năm file mà verifier kiểm tra ngay:

| File trong `/content/lab22/` | Mục đích |
|---|---|
| `adapters/sft-mini/adapter_config.json` | Cấu hình adapter SFT |
| `models/sft-merged/config.json` | Cấu hình mô hình SFT đã gộp |
| `data/pref/train.parquet` | Split train thực tế |
| `data/pref/eval.parquet` | Split held-out thực tế |
| `adapters/dpo/adapter_config.json` | Cấu hình adapter DPO và đường dẫn reference |

Cần thêm `adapters/dpo/split.json`: verifier hiện chưa tới bước này vì thiếu adapter config. File fingerprint này dùng để kiểm tra hai Parquet đúng là split lúc huấn luyện DPO.

Cần tải notebook Colab `.ipynb` **đã chạy và còn output** bằng **File → Download → Download .ipynb**. Hai notebook trong repo hiện là bản chưa chạy, không có output; yêu cầu nộp bài cần bằng chứng NB0–NB4 đã chạy. Output này cũng giúp xác nhận số mẫu SFT/preference thực tế, NB0 assertions, ba cặp preference mẫu và các thông tin cấu hình còn thiếu. Không thay output bằng kết quả tự tạo.

## Tải gọn sáu file từ Colab

Chạy cell sau trong **phiên đã hoàn thành NB1–NB4**, rồi gửi file ZIP tải xuống. Cell chỉ đọc các file nhỏ và đóng gói chúng; không tải trọng số mô hình.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

root = Path('/content/lab22')
required = [
    'adapters/sft-mini/adapter_config.json',
    'models/sft-merged/config.json',
    'data/pref/train.parquet',
    'data/pref/eval.parquet',
    'adapters/dpo/adapter_config.json',
    'adapters/dpo/split.json',
]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise FileNotFoundError('Thiếu file từ phiên đã chạy: ' + ', '.join(missing))

archive = Path('/content/lab22-extra-artifacts.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
    for name in required:
        z.write(root / name, arcname=name)
files.download(str(archive))
```

Nếu phiên cũ đã mất các file, cần khôi phục từ bản lưu cùng lần chạy hoặc chạy lại NB1–NB4 để có bộ bằng chứng nhất quán. Không tạo config hoặc split giả để vượt verify.

## Sau khi bổ sung

Giữ nguyên cấu trúc thư mục khi đưa file về repo. Cấu hình DPO từ Colab có thể chứa đường dẫn `/content/lab22/models/sft-merged`; verifier trên Windows so đường dẫn tuyệt đối với vị trí repo hiện tại, nên có thể báo `WRONG REF` sau khi chuyển máy. Cần đối chiếu config gốc và notebook trước khi xử lý việc chuyển đường dẫn; đây không phải lý do sửa số liệu hay bỏ bước kiểm tra.

Có thể xác minh trong Colab bằng cách tải bài `REFLECTION.md` đã hoàn thiện vào `/content/lab22/submission/REFLECTION.md`, rồi chạy:

```python
%cd /content/lab22
!make verify
```

Trên Windows, lệnh tương đương là `python scripts/verify.py`; khi bổ sung Parquet và adapter config, chương trình còn cần thư viện `datasets` để nhập phần kiểm tra split (Python hiện tại chưa có thư viện này).

Hiện tại **chưa xác nhận bài sẵn sàng nộp**. Chỉ kết luận verify đạt sau khi chương trình thoát với mã 0; notebook đã chạy cũng cần đưa vào repo theo README. Kết quả mô hình không vượt SFT vẫn là kết quả hợp lệ để báo cáo. Các file mới trong repo hiện chưa được commit/push.
