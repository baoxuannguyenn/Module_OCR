import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

class WordDetector:
    def __init__(
        self, 
        model_path: str = "yolo11m.pt", 
        device: str = "cuda", 
        conf_thres: float = 0.25, 
        iou_thres: float = 0.45
    ):
        """
        Word Detector sử dụng thư viện chính thức của Ultralytics (YOLOv8 / YOLO11).
        
        Args:
            model_path: Đường dẫn tới file weights (.pt, .onnx, .engine).
            device: 'cuda' hoặc 'cpu' (hoặc số GPU như '0').
            conf_thres: Ngưỡng độ tin cậy (Confidence threshold).
            iou_thres: Ngưỡng NMS IoU (Intersection over Union).
        """
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load weights thông qua Ultralytics YOLO API
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, imgsz: int = 1024) -> Tuple[List[List[float]], List[np.ndarray]]:
        """
        Dự đoán vị trí các từ trong ảnh và crop từng vùng từ.
        
        Args:
            image: Ảnh đầu vào dạng numpy array (BGR OpenCV).
            imgsz: Kích thước resize khi infer (khuyến nghị 1024 cho OCR).
            
        Returns:
            boxes: Danh sách bounding boxes dạng [x1, y1, x2, y2, score, class_id]
            cropped_words: Danh sách các patch ảnh đã crop sẵn để đưa sang Recognizer
        """
        # Dự đoán bằng Ultralytics predict API
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        boxes_data = result.boxes.data.cpu().numpy()  # Tensor Nx6: [x1, y1, x2, y2, conf, cls]
        
        h_orig, w_orig = image.shape[:2]
        boxes = []
        cropped_words = []
        
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Giới hạn tọa độ trong kích thước ảnh gốc
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            
            # Crop vùng từ/chữ
            word_crop = image[y1:y2, x1:x2]
            
            if word_crop.size > 0:
                boxes.append([x1, y1, x2, y2, float(conf), int(cls_id)])
                cropped_words.append(word_crop)
                
        return boxes, cropped_words

    @staticmethod
    def train_model(
        data_yaml: str,
        model_name: str = "yolo11m.pt",
        epochs: int = 100,
        imgsz: int = 1024,
        batch_size: int = 16,
        device: str = "0",
        project: str = "runs/detect",
        name: str = "word_detection",
        **kwargs
    ):
        """
        Huấn luyện model YOLO Word Detection bằng Ultralytics.
        
        Args:
            data_yaml: Đường dẫn tới file config dữ liệu .yaml (chứa train/val path và names).
            model_name: Base model weights ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt').
            epochs: Số epoch huấn luyện.
            imgsz: Kích thước ảnh training (1024 đề xuất cho OCR word detection).
            batch_size: Batch size.
            device: Thiết bị train ('0', 'cpu', '0,1').
            project: Thư mục lưu kết quả.
            name: Tên của đợt huấn luyện.
        """
        model = YOLO(model_name)
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            # Cấu hình tối ưu cho Word Detection
            max_det=1000,      # Tối đa 1000 box/ảnh (tài liệu chứa nhiều chữ)
            mosaic=0.5,        # Augmentation ghép ảnh
            degrees=5.0,       # Xoay nhẹ ảnh ±5 độ
            scale=0.5,         # Scale ngẫu nhiên
            box=7.5,           # Bbox loss weight
            cls=0.5,           # Classification loss weight (vì chỉ có 1 class)
            dfl=1.5,           # DFL loss weight
            **kwargs
        )
        return results


