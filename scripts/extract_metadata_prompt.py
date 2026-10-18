#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuẩn bị dữ liệu Fine-tune cho Qwen:
1. Lấy text Trang 1 (Đầu văn bản).
2. Định vị trang kết thúc văn bản chính (bỏ qua phụ lục) và lấy nguyên văn cụm text thô của khối chữ ký.
3. Ghép nối thành 1 chuỗi ngữ cảnh duy nhất (Context) đưa vào Prompt để Qwen tự suy luận.
4. Lưu vào dataset_for_qwen/train.jsonl theo chuẩn ChatML.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Các từ khóa chức danh/thẩm quyền ký để định vị khối ký
SIGNATURE_KEYWORDS = [
    r"\bKT\.", r"\bTM\.", r"\bTL\.", r"\bTUQ\.",
    r"THỦ TƯỚNG", r"PHÓ THỦ TƯỚNG",
    r"THỐNG ĐỐC", r"PHÓ THỐNG ĐỐC",
    r"BỘ TRƯỞNG", r"THỨ TRƯỞNG",
    r"CHỦ TỊCH", r"PHÓ CHỦ TỊCH",
    r"GIÁM ĐỐC", r"TỔNG GIÁM ĐỐC",
    r"CỤC TRƯỞNG", r"VỤ TRƯỞNG", r"CHÁNH VĂN PHÒNG"
]
ANCHOR_REGEX = re.compile("|".join(SIGNATURE_KEYWORDS), re.IGNORECASE)

# Từ khóa nhận diện trang Phụ lục để bỏ qua
APPENDIX_REGEX = re.compile(
    r"^\s*(PHỤ LỤC|PHU LUC|DANH MỤC|BẢNG KÊ|BẢNG TỔNG HỢP)\b|"
    r"(KÈM THEO (QUYẾT ĐỊNH|THÔNG TƯ|NGHỊ ĐỊNH|CÔNG VĂN))",
    re.IGNORECASE
)

IGNORE_HEADER_KEYWORDS = ["CỘNG HÒA", "ĐỘC LẬP", "HẠNH PHÚC", "CÔNG BÁO"]


def is_appendix_page(page_results: List[Dict[str, Any]], page_lines: List[str], page_height: float) -> bool:
    """Kiểm tra trang Phụ lục để bỏ qua"""
    for item in page_results:
        text = item.get("text", "").strip()
        box = item.get("box", [0, 0, 0, 0])
        ymin = box[1]
        if ymin < 0.25 * page_height and APPENDIX_REGEX.search(text):
            return True
            
    for line in page_lines[:5]:
        if APPENDIX_REGEX.search(line.strip()):
            return True
    return False


def get_raw_signature_block(page_results: List[Dict[str, Any]], page_lines: List[str], page_width: float, is_first_page: bool = False) -> Optional[str]:
    """
    Chỉ lấy NGUYÊN VĂN (RAW TEXT) của khối chữ ký:
    - Tìm dòng chức danh (KT., THỦ TƯỚNG, BỘ TRƯỞNG...) ở nửa bên phải (X > 40%).
    - Bốc nguyên vẹn toàn bộ cụm dòng từ chức danh đó đến hết vùng ký (3-5 dòng tiếp theo).
    - Không can thiệp bóc tách tên hay chức vụ, để Qwen tự quyết định!
    """
    # 1. Quét theo bounding box nếu có
    if page_results:
        right_boxes = []
        for item in page_results:
            text = item.get("text", "").strip()
            box = item.get("box", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            x_center = (xmin + xmax) / 2
            
            if x_center > 0.40 * page_width and len(text.split()) <= 6:
                if is_first_page and any(k in text.upper() for k in IGNORE_HEADER_KEYWORDS):
                    continue
                right_boxes.append({"text": text, "ymin": ymin})
                
        if right_boxes:
            right_boxes.sort(key=lambda x: x["ymin"])
            for i, item in enumerate(right_boxes):
                if ANCHOR_REGEX.search(item["text"]):
                    # Lấy trọn vẹn cụm dòng của khối ký (tối đa 4 dòng tiếp theo)
                    block = right_boxes[i : min(i + 5, len(right_boxes))]
                    return "\n".join([b["text"] for b in block])

    # 2. Fallback quét theo lines text thuần
    if page_lines:
        for i, line in enumerate(page_lines):
            text = line.strip()
            if is_first_page and any(k in text.upper() for k in IGNORE_HEADER_KEYWORDS):
                continue
            if ANCHOR_REGEX.search(text) and len(text.split()) <= 6:
                block = page_lines[i : min(i + 5, len(page_lines))]
                return "\n".join([l.strip() for l in block if l.strip()])

    return None


def build_qwen_input_context(ocr_json_path: str) -> Dict[str, Any]:
    """
    Đọc file OCR và ghép nối:
    [Nội dung Trang 1] + [Nội dung Khối ký thô]
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = data.get("file_name", "")
    content_list = data.get("content", [])
    pages_list = data.get("pages", [])
    total_pages = data.get("total_pages", len(content_list))

    if total_pages == 0:
        return {"error": "Tài liệu rỗng"}

    content_map = {item.get("page"): item for item in content_list}
    pages_map = {item.get("page"): item for item in pages_list}

    # 1. Text Trang 1
    page_1_data = content_map.get(1, {})
    page_1_raw_text = page_1_data.get("content", "").strip()

    # 2. Tìm khối ký thô bằng cách duyệt ngược từ trang cuối
    raw_signature_text = ""
    found_page = None

    for page_num in range(total_pages, 0, -1):
        p_content = content_map.get(page_num, {})
        p_page = pages_map.get(page_num, {})
        
        results = p_page.get("results", [])
        lines = p_content.get("lines", [])
        
        img_size = p_page.get("image_size") or {}
        page_w = img_size.get("width") or 1800
        page_h = img_size.get("height") or 2400

        # Bỏ qua phụ lục
        if is_appendix_page(results, lines, page_h):
            continue

        sig_block = get_raw_signature_block(results, lines, page_width=page_w, is_first_page=(page_num == 1))
        if sig_block:
            raw_signature_text = sig_block
            found_page = page_num
            break

    # 3. Ghép nối thành 1 Context Text duy nhất đưa vào prompt
    context_text = f"""[PHẦN ĐẦU VĂN BẢN (TRANG 1)]:
{page_1_raw_text}

[KHỐI CHỮ KÝ VĂN BẢN]:
{raw_signature_text if raw_signature_text else "Không có khối chữ ký"}"""

    # 4. Prompt hoàn chỉnh để Qwen tự suy luận
    full_prompt = f"""Dưới đây là nội dung trích xuất từ phần đầu và khối chữ ký của một văn bản hành chính Việt Nam. Hãy đọc kỹ ngữ cảnh và trích xuất thông tin thành một chuỗi JSON hợp lệ.

NGỮ CẢNH VĂN BẢN:
\"\"\"
{context_text}
\"\"\"

LƯU Ý QUAN TRỌNG:
- Bỏ qua thông tin chữ ký số điện tử của cơ quan phát hành ở đầu trang (ví dụ: "Ký bởi: CÔNG BÁO...", "Cơ quan: VĂN PHÒNG CHÍNH PHỦ...", "Ngày ký:..."). Cơ quan ban hành là cơ quan có thẩm quyền ban hành văn bản chính thức (nằm ở góc trên bên trái).
- "cong_bao": Lấy thông tin số công báo và ngày đăng nếu có (ví dụ: "CÔNG BÁO/Số 282/Ngày 23-05-2026" hoặc "Công báo số 528..."). Nếu không có thì để null.

NHIỆM VỤ:
Trích xuất thông tin vào JSON với các trường sau (nếu không có trong văn bản, để giá trị là null):
1. "loai_van_ban": Loại văn bản (QUYẾT ĐỊNH, NGHỊ ĐỊNH, THÔNG TƯ, VĂN BẢN HỢP NHẤT, CHỈ THỊ...).
2. "so_ky_hieu": Số, ký hiệu văn bản (Ví dụ: 840/QĐ-TTg, 147/2026/VBHN-LQ-VPQH...).
3. "co_quan_ban_hanh": Cơ quan ban hành văn bản (Ví dụ: THỦ TƯỚNG CHÍNH PHỦ, BỘ XÂY DỰNG,VĂN PHÒNG QUỐC HỘI,NGÂN HÀNG NHÀ NƯỚC VIỆT NAM,...).
4. "ngay_ban_hanh": Ngày tháng năm ban hành văn bản theo định dạng DD/MM/YYYY.
5. "trich_yeu": Tóm tắt nội dung/tiêu đề chính của văn bản.
6. "nguoi_ky": Họ và tên của người ký văn bản.
7. "cong_bao": Thông tin số Công báo và ngày đăng văn bản.
"""

    return {
        "file_name": file_name,
        "total_pages": total_pages,
        "signature_page": found_page,
        "raw_signature_text": raw_signature_text,
        "context_text": context_text,
        "full_prompt": full_prompt
    }


def save_dataset_for_qwen(info: Dict[str, Any], dataset_dir: str = "dataset_for_qwen", metadata_csv_path: Optional[str] = None):
    """Lưu dữ liệu vào train.jsonl theo chuẩn ChatML"""
    out_dir = Path(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = info.get("file_name", "")
    
    # Chuẩn bị nhãn Ground Truth (assistant output) để fine-tune
    assistant_output = {
        "loai_van_ban": None,
        "so_ky_hieu": None,
        "co_quan_ban_hanh": None,
        "ngay_ban_hanh": None,
        "trich_yeu": None,
        "nguoi_ky": None,
        "cong_bao": None
    }
    
    # Đối chiếu metadata.csv nếu có
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        try:
            import urllib.parse
            import pandas as pd
            df = pd.read_csv(metadata_csv_path)

            def normalize_fname(name: str) -> str:
                name = urllib.parse.unquote(str(name)).lower()
                name = re.sub(r"\s*\(\d+\)", "", name)  # loại bỏ duplicate suffix như " (2)"
                return Path(name).name.strip()

            target_norm = normalize_fname(file_name)
            df["norm_fname"] = df["file_name"].apply(normalize_fname)
            matched = df[df["norm_fname"] == target_norm]

            # Nếu chưa khớp theo tên file chuẩn hóa, thử tìm kiếm chứa substring
            if matched.empty and target_norm:
                matched = df[df["norm_fname"].str.contains(re.escape(Path(target_norm).stem), na=False)]

            if not matched.empty:
                row = matched.iloc[0]
                assistant_output["loai_van_ban"] = str(row.get("Loại văn bản", "")).strip() or None
                assistant_output["so_ky_hieu"] = str(row.get("Số, ký hiệu", "")).strip() or None
                assistant_output["co_quan_ban_hanh"] = str(row.get("Cơ quan ban hành", "")).strip() or None
                assistant_output["ngay_ban_hanh"] = str(row.get("Ngày ban hành", "")).strip() or None
                assistant_output["trich_yeu"] = str(row.get("Trích yếu", "")).strip() or None
                if pd.notna(row.get("Người ký")) and str(row.get("Người ký")).strip():
                    assistant_output["nguoi_ky"] = str(row.get("Người ký")).strip()
                if pd.notna(row.get("Công báo")) and str(row.get("Công báo")).strip():
                    assistant_output["cong_bao"] = str(row.get("Công báo")).strip()
        except Exception as e:
            print(f"[!] Lỗi đọc metadata.csv: {e}")

    # Tạo mẫu ChatML cho Qwen
    chat_sample = {
        "messages": [
            {
                "role": "system",
                "content": "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ."
            },
            {
                "role": "user",
                "content": info.get("full_prompt", "")
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_output, ensure_ascii=False, indent=2)
            }
        ]
    }

    # Lưu vào train.jsonl
    jsonl_path = out_dir / "train.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")

    # Lưu thêm 1 file .txt thuần túy để người dùng dễ đọc trực quan
    stem_name = Path(file_name).stem
    txt_path = out_dir / f"{stem_name}_context.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"FILE NGUỒN: {file_name}\n")
        f.write(f"KHỐI CHỮ KÝ NẰM TẠI TRANG: Trang {info.get('signature_page')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(info.get("full_prompt", ""))

    return str(jsonl_path), str(txt_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_json = project_root / "json_output" / "output.json"
    metadata_csv = project_root / "dataset_congbao_download" / "metadata.csv"
    dataset_dir = project_root / "dataset_for_qwen"

    if sample_json.exists():
        print(f"[*] Đang xử lý file OCR: {sample_json.name}\n")
        info = build_qwen_input_context(str(sample_json))

        # Lưu vào dataset_for_qwen
        jsonl_path, txt_path = save_dataset_for_qwen(
            info, 
            dataset_dir=str(dataset_dir),
            metadata_csv_path=str(metadata_csv) if metadata_csv.exists() else None
        )
        print(f"\n[+] ĐÃ LƯU THÀNH CÔNG VÀO DATASET CHO QWEN:")
        print(f"    - File TXT dễ đọc: {txt_path}")
        print(f"    - File JSONL huấn luyện: {jsonl_path}")
    else:
        print(f"Không tìm thấy file: {sample_json}")
