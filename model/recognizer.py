import PIL.Image
# Vá lỗi tương thích cho VietOCR khi dùng Pillow >= 10.0.0
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import numpy as np
from PIL import Image
from typing import List, Union
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import cv2
class TextRecognizer:
    def __init__(self, config_base: str = "vgg_transformer", weight_path: str = None, device: str = "cuda"):
        """
        Khởi tạo VietOCR Predictor.
        Args:
            config_base: 'vgg_transformer'
            weight_path: Đường dẫn file weight .pth đã fine-tune
            device: 'cuda' hoặc 'cpu'
        """
        self.config = Cfg.load_config_from_name(config_base)
        if weight_path:
            self.config['weights'] = weight_path
        self.config['device'] = device
        self.config['predictor']['beamsearch'] = False  # True nếu cần chính xác cao hơn nhưng chậm hơn
        
        self.detector = Predictor(self.config)

    def predict(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Nhận diện 1 ảnh crop"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.detector.predict(image)

    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[str]:
        """Nhận diện danh sách nhiều ảnh crop"""
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_images.append(img)
        return self.detector.predict_batch(pil_images)
