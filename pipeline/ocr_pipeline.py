import os
import json
import cv2
import yaml
import numpy as np
from typing import List, Dict, Any, Union, Optional
from model.detect_word import WordDetector
from model.recognizer import TextRecognizer
from utils.box_utils import sort_boxes_reading_order

class OCRPipeline:
    def __init__(self, config_path: str = "config/ocr.yaml", **overrides):
        """
        Khởi tạo OCR Pipeline dựa trên file cấu hình YAML.
        """
        self.config = self._load_config(config_path)
        
        # Áp dụng các tham số ghi đè nếu có
        for key, value in overrides.items():
            if key in self.config:
                self.config[key] = value

        self.device = self.config.get("device", "cuda")
        
        # 1. Khởi tạo Word Detector (YOLO)
        det_cfg = self.config.get("detector", {})
        det_model_path = det_cfg.get("model_path", "runs/detect/word_detection_v1/weights/best.pt")
        self.det_conf = det_cfg.get("conf_thres", 0.25)
        self.det_iou = det_cfg.get("iou_thres", 0.45)
        self.det_imgsz = det_cfg.get("imgsz", 1024)
        
        print(f"[OCR] Khởi tạo YOLO Detector: {det_model_path} (Device: {self.device})")
        self.detector = WordDetector(
            model_path=det_model_path,
            device=self.device,
            conf_thres=self.det_conf,
            iou_thres=self.det_iou
        )

        # 2. Khởi tạo Text Recognizer (VietOCR)
        rec_cfg = self.config.get("recognizer", {})
        rec_base = rec_cfg.get("config_base", "vgg_transformer")
        rec_weight = rec_cfg.get("weight_path", None)
        
        print(f"[OCR] Khởi tạo VietOCR ({rec_base}): {rec_weight} (Device: {self.device})")
        self.recognizer = TextRecognizer(
            config_base=rec_base,
            weight_path=rec_weight,
            device=self.device
        )
        
        # Cấu hình gom dòng / sắp xếp
        self.sort_cfg = self.config.get("sorting", {})
        self.y_threshold_ratio = self.sort_cfg.get("y_threshold_ratio", 0.5)
        self.line_break = self.sort_cfg.get("line_break", True)

    def _load_config(self, config_path: str) -> dict:
        """Đọc file YAML config an toàn"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def predict_image(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Xử lý OCR cho 1 trang ảnh (numpy array hoặc file path).
        """
        img_name = ""
        if isinstance(image, str):
            img_name = os.path.basename(image)
            image_mat = cv2.imread(image)
            if image_mat is None:
                raise ValueError(f"Không đọc được ảnh từ đường dẫn: {image}")
        else:
            image_mat = image

        h_orig, w_orig = image_mat.shape[:2]

        # 1. Phát hiện từ bằng YOLO
        boxes, crops = self.detector.detect(image_mat, imgsz=self.det_imgsz)
        if not boxes:
            return {
                "image_name": img_name,
                "image_size": {"width": int(w_orig), "height": int(h_orig)},
                "full_text": "",
                "lines": [],
                "results": []
            }

        # 2. Sắp xếp thứ tự đọc và gom dòng
        flat_indices, line_indices = sort_boxes_reading_order(
            boxes, 
            y_threshold_ratio=self.y_threshold_ratio
        )
        
        sorted_boxes = [boxes[i] for i in flat_indices]
        sorted_crops = [crops[i] for i in flat_indices]

        # 3. Nhận diện chữ bằng VietOCR batch
        texts = self.recognizer.predict_batch(sorted_crops)

        # 4. Gom văn bản theo từng dòng
        pred_map = {idx: text.strip() for idx, text in zip(flat_indices, texts)}

        line_texts = []
        for line in line_indices:
            words_in_line = [pred_map[idx] for idx in line if pred_map[idx]]
            if words_in_line:
                line_texts.append(" ".join(words_in_line))

        separator = "\n" if self.line_break else " "
        full_text = separator.join(line_texts)

        # Chi tiết từng từ
        word_results = []
        for box, text in zip(sorted_boxes, texts):
            text = text.strip()
            if text:
                word_results.append({
                    "box": [int(b) for b in box[:4]],
                    "score": round(float(box[4]), 4),
                    "text": text
                })

        return {
            "image_name": img_name,
            "image_size": {"width": int(w_orig), "height": int(h_orig)},
            "full_text": full_text,
            "lines": line_texts,
            "results": word_results
        }

    def predict_pdf(self, pdf_path: str, dpi: int = 200, save_json: Optional[str] = None) -> Dict[str, Any]:
        """
        Xử lý OCR cho toàn bộ file PDF nhiều trang.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "Để xử lý file PDF, vui lòng cài đặt thư viện pymupdf bằng lệnh: pip install pymupdf"
            )

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF tại: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"Tổng số trang: {total_pages}")

        pages_result = []
        content_by_pages = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            print(f"Đang xử lý trang {page_num}/{total_pages}...")
            page = doc[page_idx]

            # Render trang PDF ra Pixmap với DPI mong muốn
            pix = page.get_pixmap(dpi=dpi)
            
            # Chuyển đổi Pixmap sang mảng numpy BGR cho OpenCV
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:  # RGBA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Chạy OCR cho trang ảnh này
            page_res = self.predict_image(img)
            
            # Lưu content và lines chuẩn theo từng trang
            content_by_pages.append({
                "page": page_num,
                "content": page_res.get("full_text", ""),
                "lines": page_res.get("lines", [])
            })

            # pages chỉ lưu thông tin tọa độ/kết quả từ (box, score, text) để tránh trùng lặp content/full_text
            pages_result.append({
                "page": page_num,
                "image_size": page_res.get("image_size"),
                "results": page_res.get("results", [])
            })

        output_data = {
            "file_name": os.path.basename(pdf_path),
            "total_pages": total_pages,
            "content": content_by_pages,
            "pages": pages_result
        }

        if save_json:
            self.save_to_json(output_data, save_json)

        return output_data

    def predict(self, input_path: Union[str, np.ndarray], dpi: int = 200, save_json: Optional[str] = None) -> Dict[str, Any]:
        """
        Hàm tổng quát: Tự động phân loại Ảnh hoặc PDF dựa vào phần mở rộng file.
        """
        if isinstance(input_path, str) and input_path.lower().endswith(".pdf"):
            return self.predict_pdf(input_path, dpi=dpi, save_json=save_json)
        else:
            res = self.predict_image(input_path)
            output_data = {
                "file_name": res.get("image_name"),
                "total_pages": 1,
                "content": [
                    {
                        "page": 1,
                        "content": res.get("full_text", ""),
                        "lines": res.get("lines", [])
                    }
                ],
                "pages": [
                    {
                        "page": 1,
                        "image_size": res.get("image_size"),
                        "results": res.get("results", [])
                    }
                ]
            }
            if save_json:
                self.save_to_json(output_data, save_json)
            return output_data

    @staticmethod
    def save_to_json(data: Dict[str, Any], json_path: str):
        """Lưu kết quả ra file JSON có hỗ trợ tiếng Việt UTF-8"""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu kết quả OCR vào: {json_path}")
