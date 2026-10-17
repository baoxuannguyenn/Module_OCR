import os
import sys

# Thêm thư mục gốc 'sgod_model_ocr' vào sys.path để import được pipeline, model, utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from pipeline.ocr_pipeline import OCRPipeline

# Khởi tạo tự động đọc từ config/ocr.yaml
pipeline = OCRPipeline(config_path="config/ocr.yaml")

# Hỗ trợ cả file PDF lẫn file ảnh
file_path = r"E:\SGOD\21_9_2026\img\2026_531-VBHN_109_2026_VBHN-NĐ-NHNN.pdf"

print(f"Đang chạy OCR cho: {file_path}...")
output = pipeline.predict(file_path, save_json="json_output/output.json")
