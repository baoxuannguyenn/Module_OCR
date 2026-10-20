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
import sys
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

    # 4. Prompt tối ưu hoàn chỉnh: Hướng dẫn chi tiết từng trường + Quy tắc lọc nhiễu + Schema JSON cố định
    full_prompt = f"""Dưới đây là nội dung trích xuất từ phần đầu và khối chữ ký của một văn bản hành chính Việt Nam. Hãy đọc kỹ ngữ cảnh và trích xuất thông tin thành một chuỗi JSON hợp lệ.

NGỮ CẢNH VĂN BẢN:
\"\"\"
{context_text}
\"\"\"

QUY TẮC TRÍCH XUẤT VÀ LỌC NHIỄU:
- Bỏ qua các dòng chữ ký số điện tử phát hành (như "Ký bởi: ...", "Cơ quan: VĂN PHÒNG CHÍNH PHỦ...", "Ngày ký: ...").
- "co_quan_ban_hanh": Tên cơ quan ban hành văn bản chính thức ở góc trên bên trái (Ví dụ: THỦ TƯỚNG CHÍNH PHỦ, BỘ XÂY DỰNG, BỘ CÔNG AN, BỘ TƯ PHÁP, VĂN PHÒNG QUỐC HỘI...). TUYỆT ĐỐI BỎ QUA Quốc hiệu và Tiêu ngữ ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc").
- "ngay_ban_hanh": Là ngày tháng năm ký/ban hành văn bản ở dòng địa danh bên dưới tiêu ngữ hoặc bên cạnh số ký hiệu (Ví dụ: "Hà Nội, ngày 13 tháng 5 năm 2026" -> chuẩn hóa thành "13/05/2026"). TUYỆT ĐỐI KHÔNG lấy nhầm ngày đăng công báo.
- "cong_bao": Thông tin số công báo và ngày đăng (Ví dụ: dòng "CÔNG BÁO/Số 282/Ngày 23-05-2026 69" -> CHỈ LẤY "CÔNG BÁO/Số 282/Ngày 23-05-2026", BỎ QUA số trang ở cuối dòng như 69, 15...). Nếu không có thì để null.
- "nguoi_ky": Chỉ lấy họ và tên người ký chính thức ở khối chữ ký cuối văn bản (Ví dụ: "Hồ Quốc Dũng", "Phạm Gia Túc"). BỎ QUA các từ thẩm quyền/chức vụ như "KT. THỦ TƯỚNG", "PHÓ THỦ TƯỚNG", "BỘ TRƯỞNG".

NHIỆM VỤ:
Trích xuất thông tin vào duy nhất một chuỗi JSON với đầy đủ 7 trường sau (nếu trường nào không có trong văn bản, đặt giá trị là null, tuyệt đối không được thiếu bất kỳ trường nào):
1. "loai_van_ban": Tên loại văn bản viết in hoa (Ví dụ: QUYẾT ĐỊNH, NGHỊ ĐỊNH, THÔNG TƯ, VĂN BẢN HỢP NHẤT, CHỈ THỊ...).
2. "so_ky_hieu": Số, ký hiệu văn bản (Ví dụ: 840/QĐ-TTg, 47/2026/TT-BCA...).
3. "co_quan_ban_hanh": Tên cơ quan ban hành chính thức.
4. "ngay_ban_hanh": Ngày tháng năm ban hành định dạng DD/MM/YYYY.
5. "trich_yeu": Tóm tắt tiêu đề/nội dung chính của văn bản.
6. "nguoi_ky": Họ và tên người ký văn bản.
7. "cong_bao": Thông tin số Công báo và ngày đăng văn bản.

ĐỊNH DẠNG ĐẦU RA YÊU CẦU:
```json
{{
  "loai_van_ban": "...",
  "so_ky_hieu": "...",
  "co_quan_ban_hanh": "...",
  "ngay_ban_hanh": "...",
  "trich_yeu": "...",
  "nguoi_ky": "...",
  "cong_bao": "..."
}}
```"""

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
                "content": "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
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

    # Lưu thêm 1 file .txt thuần túy chứa đúng prompt để inference trực tiếp
    stem_name = Path(file_name).stem
    txt_path = out_dir / f"{stem_name}_context.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(info.get("full_prompt", ""))

    return str(jsonl_path), str(txt_path)


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parents[1]

    # Tự động tìm metadata.csv (Ưu tiên HF Cache -> Kaggle Input -> local)
    default_csv = project_root / "dataset_congbao_download" / "metadata.csv"
    
    # 1. Kiểm tra Hugging Face Cache
    hf_bases = [
        Path("/root/.cache/huggingface/hub"),
        Path(os.path.expanduser("~/.cache/huggingface/hub"))
    ]
    found_hf = False
    for hf_base in hf_bases:
        if hf_base.exists():
            found = list(hf_base.rglob("metadata.csv"))
            if found:
                default_csv = found[0]
                found_hf = True
                break

    # 2. Kiểm tra Kaggle Input nếu chưa thấy ở HF
    if not found_hf:
        kaggle_csv = Path("/kaggle/input")
        if kaggle_csv.exists():
            found = list(kaggle_csv.rglob("metadata.csv"))
            if found:
                default_csv = found[0]

    parser = argparse.ArgumentParser(description="Trích xuất Prompt và chuẩn bị Dataset Qwen từ file OCR JSON")
    parser.add_argument("--json_path", type=str, default=str(project_root / "json_output"), help="Đường dẫn file .json hoặc thư mục chứa các file OCR JSON")
    parser.add_argument("--metadata_csv", type=str, default=str(default_csv), help="Đường dẫn file metadata.csv")
    parser.add_argument("--dataset_dir", type=str, default=str(project_root / "dataset_for_qwen"), help="Thư mục xuất dataset")
    parser.add_argument("--reset", action="store_true", help="Xóa train.jsonl cũ trước khi sinh mới (tránh trùng lặp khi chạy lại)")

    args = parser.parse_args()

    input_p = Path(args.json_path)
    csv_path = args.metadata_csv if os.path.exists(args.metadata_csv) else None
    dataset_p = Path(args.dataset_dir)
    dataset_p.mkdir(parents=True, exist_ok=True)

    # 1. Tìm danh sách file JSON cần xử lý
    json_files: List[Path] = []
    if input_p.is_file() and input_p.suffix.lower() == ".json":
        json_files = [input_p]
    elif input_p.is_dir():
        json_files = sorted(list(input_p.glob("*.json")))

    if not json_files:
        print(f"[!] Không tìm thấy file JSON nào tại: {input_p}")
        sys.exit(0)

    # Nếu có cờ --reset hoặc chạy batch thư mục, xóa file train.jsonl cũ để tránh duplicate
    train_jsonl = dataset_p / "train.jsonl"
    if args.reset and train_jsonl.exists():
        train_jsonl.unlink()
        print(f"[*] Đã reset file dataset cũ: {train_jsonl}")

    print("=" * 60)
    print(f"[*] BẮT ĐẦU XỬ LÝ {len(json_files)} FILE OCR JSON SANG DATASET QWEN")
    print(f"[*] Metadata CSV: {csv_path if csv_path else 'Không tìm thấy (nhãn sẽ để null)'}")
    print(f"[*] Thư mục đích: {dataset_p.resolve()}")
    print("=" * 60)

    success_cnt = 0
    for idx, jf in enumerate(json_files, 1):
        try:
            info = build_qwen_input_context(str(jf))
            if "error" in info:
                print(f"[{idx}/{len(json_files)}] [!] Bỏ qua {jf.name}: {info['error']}")
                continue

            jsonl_path, txt_path = save_dataset_for_qwen(
                info,
                dataset_dir=str(dataset_p),
                metadata_csv_path=csv_path
            )
            success_cnt += 1
            if len(json_files) <= 10 or idx % 10 == 0 or idx == len(json_files):
                print(f"[{idx}/{len(json_files)}] [+] Đã xử lý: {jf.name}")
        except Exception as e:
            print(f"[{idx}/{len(json_files)}] [!] Lỗi khi xử lý {jf.name}: {e}")

    print("\n" + "=" * 60)
    print(f"[+] HOÀN THÀNH: {success_cnt}/{len(json_files)} file đã được lưu vào:")
    print(f"    - File JSONL huấn luyện: {dataset_p / 'train.jsonl'}")
    print(f"    - File TXT dễ đọc:      {dataset_p}")
    print("=" * 60)
