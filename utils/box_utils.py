import numpy as np
from typing import List, Tuple

def sort_boxes_reading_order(boxes: List[List[float]], y_threshold_ratio: float = 0.5) -> Tuple[List[int], List[List[int]]]:
    """
    Sắp xếp các bounding boxes [x1, y1, x2, y2, ...] theo thứ tự đọc tự nhiên:
    Dòng trên trước, dòng dưới sau; Trong cùng một dòng thì từ Trái qua Phải.
    
    Args:
        boxes: Danh sách bounding boxes [[x1, y1, x2, y2, conf, cls], ...]
        y_threshold_ratio: Tỷ lệ so với chiều cao trung bình để xác định 2 box cùng dòng.
        
    Returns:
        flat_indices: Danh sách index phẳng theo thứ tự đọc hoàn chỉnh.
        line_indices: Danh sách index gom theo từng dòng [[id_w1, id_w2], [id_w3, id_w4]]
    """
    if len(boxes) == 0:
        return [], []
        
    # Gắn kèm index ban đầu để không bị lệch thứ tự với crops
    indexed_boxes = list(enumerate(boxes))
    
    # Sắp xếp sơ bộ theo tọa độ y1 (từ trên xuống)
    indexed_boxes.sort(key=lambda item: item[1][1])
    
    lines = []
    current_line = [indexed_boxes[0]]
    
    for item in indexed_boxes[1:]:
        idx, box = item
        # Chiều cao trung bình của dòng hiện tại
        avg_h = sum(b[1][3] - b[1][1] for b in current_line) / len(current_line)
        line_y_center = sum((b[1][1] + b[1][3]) / 2 for b in current_line) / len(current_line)
        box_y_center = (box[1] + box[3]) / 2
        
        # Nếu chênh lệch tâm Y nhỏ hơn ngưỡng chiều cao -> cùng dòng
        if abs(box_y_center - line_y_center) < (avg_h * y_threshold_ratio):
            current_line.append(item)
        else:
            # Sắp xếp các từ trong dòng vừa gom từ trái qua phải (theo x1)
            current_line.sort(key=lambda x: x[1][0])
            lines.append(current_line)
            current_line = [item]
            
    if current_line:
        current_line.sort(key=lambda x: x[1][0])
        lines.append(current_line)
        
    # Tách ra danh sách index gom theo từng dòng và danh sách index phẳng
    line_indices = [[item[0] for item in line] for line in lines]
    flat_indices = [item[0] for line in lines for item in line]
    
    return flat_indices, line_indices