#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chạy OCR hàng loạt (Batch OCR) cho nhiều file PDF:
- Tự động nhận diện môi trường Kaggle và tìm kiếm weights trong /kaggle/input/
- Hỗ trợ resume (bỏ qua file đã OCR) để tiết kiệm thời gian
- Lưu kết quả mỗi PDF thành 1 file .json tương ứng trong thư mục output
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Optional, List

# Thêm project_root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pipeline.ocr_pipeline import OCRPipeline


def find_file_in_kaggle(pattern: str) -> Optional[str]:
    """Tìm file khớp với pattern trong thư mục /kaggle/input"""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def resolve_weights(yolo_arg: Optional[str], vietocr_arg: Optional[str]):
    """Tự động phân giải đường dẫn weights (ưu tiên CLI -> Kaggle -> config)"""
    yolo_path = yolo_arg
    vietocr_path = vietocr_arg

    # 1. Kiểm tra trên Kaggle nếu chưa truyền tham số
    if not yolo_path:
        yolo_path = find_file_in_kaggle("*yolo*.pt") or find_file_in_kaggle("*.pt")
    if not vietocr_path:
        vietocr_path = find_file_in_kaggle("*transformer*.pth") or find_file_in_kaggle("*.pth")

    # 2. Kiểm tra các thư mục weights cục bộ phổ biến
    local_weights_dir = PROJECT_ROOT / "weights"
    if not yolo_path and local_weights_dir.exists():
        pts = list(local_weights_dir.glob("*.pt"))
        if pts:
            yolo_path = str(pts[0])
    if not vietocr_path and local_weights_dir.exists():
        pths = list(local_weights_dir.glob("*.pth"))
        if pths:
            vietocr_path = str(pths[0])

    return yolo_path, vietocr_path


def resolve_pdf_dir(pdf_dir_arg: Optional[str]) -> Path:
    """Tự động tìm thư mục chứa PDF (Ưu tiên CLI -> Hugging Face Cache -> Kaggle Input -> Local)"""
    if pdf_dir_arg:
        p = Path(pdf_dir_arg)
        if p.exists():
            return p

    # 1. Kiểm tra trong Hugging Face Cache nếu có tải dataset từ HF
    hf_bases = [
        Path("/root/.cache/huggingface/hub"),
        Path(os.path.expanduser("~/.cache/huggingface/hub"))
    ]
    for hf_base in hf_bases:
        if hf_base.exists():
            # Tìm thư mục pdfs hoặc bất kỳ thư mục con nào chứa .pdf
            for pdf_file in hf_base.rglob("*.pdf"):
                print(f"[*] Tự động nhận diện thư mục PDF từ Hugging Face Cache: {pdf_file.parent}")
                return pdf_file.parent

    # 2. Kiểm tra trên Kaggle Input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if d.is_dir() and list(d.glob("*.pdf")):
                return d

    # 3. Thư mục mặc định trong repo
    default_dirs = [
        PROJECT_ROOT / "dataset_congbao_download" / "pdfs",
        PROJECT_ROOT / "datatest",
        PROJECT_ROOT / "pdfs"
    ]
    for d in default_dirs:
        if d.exists() and list(d.glob("*.pdf")):
            return d

    return PROJECT_ROOT / "dataset_congbao_download" / "pdfs"


def main():
    parser = argparse.ArgumentParser(description="Chạy OCR hàng loạt cho nhiều file PDF")
    parser.add_argument("--pdf_dir", type=str, default=None, help="Thư mục chứa các file PDF cần OCR")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "json_output"), help="Thư mục lưu file JSON OCR")
    parser.add_argument("--config_path", type=str, default=str(PROJECT_ROOT / "config" / "ocr.yaml"), help="Đường dẫn file config OCR")
    parser.add_argument("--yolo_weight", type=str, default=None, help="Đường dẫn file weight YOLO (.pt)")
    parser.add_argument("--vietocr_weight", type=str, default=None, help="Đường dẫn file weight VietOCR (.pth)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--dpi", type=int, default=200, help="DPI khi render PDF ra ảnh (mặc định 200)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số file PDF cần chạy (0 = tất cả, ví dụ: 100)")
    parser.add_argument("--force", action="store_true", help="Chạy lại cả các file đã có kết quả JSON")

    args = parser.parse_args()

    # 1. Xác định thiết bị
    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("           CHẠY OCR HÀNG LOẠT (BATCH OCR)")
    print("=" * 60)
    print(f"[*] Thiết bị sử dụng: {device.upper()}")

    # 2. Phân giải weights
    yolo_weight, vietocr_weight = resolve_weights(args.yolo_weight, args.vietocr_weight)
    overrides = {"device": device}

    if yolo_weight and os.path.exists(yolo_weight):
        print(f"[+] YOLO Weight: {yolo_weight}")
        overrides["detector"] = {"model_path": yolo_weight}
    else:
        print(f"[!] Dùng YOLO Weight mặc định từ config")

    if vietocr_weight and os.path.exists(vietocr_weight):
        print(f"[+] VietOCR Weight: {vietocr_weight}")
        overrides["recognizer"] = {"weight_path": vietocr_weight}
    else:
        print(f"[!] Dùng VietOCR Weight mặc định từ config")

    # 3. Khởi tạo Pipeline
    pipeline = OCRPipeline(config_path=args.config_path, **overrides)

    # 4. Tìm danh sách PDF
    pdf_dir = resolve_pdf_dir(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục chứa file PDF tại: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    if not pdf_files:
        print(f"[!] Không có file PDF nào trong thư mục: {pdf_dir}")
        sys.exit(0)

    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]

    total_files = len(pdf_files)
    print(f"[+] Tìm thấy {total_files} file PDF trong: {pdf_dir}")

    # 5. Tạo thư mục output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Thư mục xuất JSON: {output_dir}")
    print("-" * 60)

    # 6. Chạy lặp qua từng file
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        target_json = output_dir / f"{pdf_path.stem}.json"
        
        # Bỏ qua nếu đã OCR từ trước (hỗ trợ resume khi disconnect)
        if target_json.exists() and not args.force:
            print(f"[{idx}/{total_files}] Bỏ qua (Đã tồn tại): {pdf_path.name}")
            skipped_count += 1
            continue

        print(f"\n[{idx}/{total_files}] Đang xử lý: {pdf_path.name}...")
        try:
            pipeline.predict_pdf(str(pdf_path), dpi=args.dpi, save_json=str(target_json))
            success_count += 1
        except Exception as e:
            print(f"[!] Lỗi xử lý file {pdf_path.name}: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print("HOÀN THÀNH BATCH OCR:")
    print(f"  - Tổng số file:     {total_files}")
    print(f"  - Thành công mới:   {success_count}")
    print(f"  - Đã có sẵn:        {skipped_count}")
    print(f"  - Thất bại:         {failed_count}")
    print(f"  - Thư mục JSON:     {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
