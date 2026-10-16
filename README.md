# Vietnamese OCR Pipeline (YOLO + VietOCR)

Module nhận dạng ký tự quang học (OCR) tiếng Việt kết hợp giữa **Ultralytics YOLO** (Text/Word Detection) và **VietOCR** (Text Recognition). Hỗ trợ nhận diện chính xác trên cả **File Ảnh** và **Tài liệu PDF nhiều trang**, xuất kết quả ra văn bản thô hoặc file JSON chuẩn cấu trúc.

---

## 🌟 Tính Năng Nổi Bật

- **Phát hiện từ/dòng chính xác**: Sử dụng mô hình YOLO tối ưu cho văn bản tiếng Việt.
- **Sắp xếp thứ tự đọc tự nhiên**: Tự động gom dòng (Line clustering) và sắp xếp bounding box từ trên xuống dưới, từ trái qua phải.
- **Nhận diện tiếng Việt không dấu và có dấu**: Tích hợp mô hình Transformer / Seq2Seq của VietOCR.
- **Hỗ trợ PDF thông minh**: Xử lý PDF nhiều trang tự động thông qua PyMuPDF (`fitz`), không cần cài đặt Poppler bên ngoài.
- **Cấu hình tập trung**: Toàn bộ đường dẫn weights, ngưỡng threshold và thiết bị (CUDA/CPU) được quản lý qua `config/ocr.yaml`.
- **Đầu ra JSON chi tiết**: Cung cấp tọa độ bounding box, độ tin cậy (confidence score) và văn bản từng từ/dòng.

---

## 🏗️ Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[ Ảnh / PDF ]
      │
      ▼
┌─────────────────────────┐
│ 1. WordDetector (YOLO)  │ ──> Phát hiện bounding box & crop ảnh từ/dòng
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ 2. Box Sorter & Cluster │ ──> Sắp xếp theo thứ tự đọc tự nhiên (Reading Order)
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ 3. TextRecognizer       │ ──> VietOCR dự đoán batch các ảnh crop
│    (VietOCR)            │
└─────────────────────────┘
      │
      ▼
[ Kết quả: Full text + Từng dòng + File JSON ]
```

---

## 🚀 Cài Đặt Môi Trường

Yêu cầu: **Python 3.10+** và GPU NVIDIA (khuyên dùng CUDA) hoặc CPU.

```bash
# 1. Cài đặt các thư viện lõi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics vietocr pymupdf pyyaml opencv-python pillow
```

---

## ⚙️ Cấu Hình (`config/ocr.yaml`)

Mọi thông số và đường dẫn weights đều được tùy chỉnh trong file `config/ocr.yaml`:

```yaml
device: "cuda"  # 'cuda' hoặc 'cpu'

detector:
  model_path: "weight/best_yolo_ep100.pt"  # File weight YOLO
  conf_thres: 0.25
  iou_thres: 0.45
  imgsz: 1024

recognizer:
  config_base: "vgg_transformer"  # 'vgg_transformer' hoặc 'vgg_seq2seq'
  weight_path: "weight/vgg_transformer_best.pth"  # File weight VietOCR
  beamsearch: false

sorting:
  y_threshold_ratio: 0.5   # Tỷ lệ chênh lệch trục Y để gom các từ vào cùng 1 dòng
  line_break: true         # Tự động xuống dòng (\n)
```

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Sử dụng trong code Python

```python
from pipeline.ocr_pipeline import OCRPipeline

# Khởi tạo pipeline từ file config
pipeline = OCRPipeline(config_path="config/ocr.yaml")

# 1. Xử lý Ảnh
result_img = pipeline.predict("path/to/image.jpg", save_json="output_img.json")
print(result_img["full_text"])

# 2. Xử lý File PDF nhiều trang
result_pdf = pipeline.predict("path/to/document.pdf", save_json="output_pdf.json")
print(f"Tổng số trang: {result_pdf['total_pages']}")
print(result_pdf["full_text"])
```

### 2. Chạy thử nghiệm nhanh qua Script

```bash
python test/test_ocr.py
```

---

## 📊 Cấu Trúc Kết Quả JSON Đầu Ra

File JSON xuất ra có định dạng UTF-8 chuẩn, không bị mã hóa ký tự tiếng Việt:

```json
{
    "file_name": "sample.pdf",
    "total_pages": 2,
    "full_text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc...",
    "lines": [
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc"
    ],
    "pages": [
        {
            "page": 1,
            "image_size": { "width": 1654, "height": 2339 },
            "full_text": "...",
            "lines": ["..."],
            "results": [
                {
                    "box": [120, 45, 230, 78],
                    "score": 0.9452,
                    "text": "CỘNG"
                }
            ]
        }
    ]
}
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
sgod_model_ocr/
├── config/
│   └── ocr.yaml            # File cấu hình tập trung
├── model/
│   ├── detect_word.py      # Module wrapper YOLO Detector
│   └── recognizer.py       # Module wrapper VietOCR Recognizer
├── utils/
│   └── box_utils.py        # Thuật toán gom dòng & sort reading order
├── pipeline/
│   └── ocr_pipeline.py     # Class điều phối OCR chính (Image & PDF)
├── test/
│   └── test_ocr.py         # Script kiểm thử
├── .gitignore              # Bỏ qua file weight và thư mục output
└── README.md               # Tài liệu hướng dẫn
```
