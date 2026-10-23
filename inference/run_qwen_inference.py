#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chạy Qwen3-VL Inference trực tiếp từ file .txt hoặc .jsonl.
Được chuẩn hóa chuẩn xác theo hướng dẫn chính thức của Qwen3-VL.
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoProcessor


from transformers import Qwen3VLForConditionalGeneration

class QwenInference:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """Khởi tạo mô hình và processor Qwen3-VL theo chuẩn chính thức."""
        print(f"[*] Đang tải model: {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto"
        )
        self.model.eval()
        print(f"[✓] Đã tải model thành công vào: {self.model.device}")

    def predict(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024, do_sample: bool = False) -> str:
        """
        Inference sinh kết quả từ messages theo chuẩn Qwen3-VL.
        Mặc định: do_sample=False (Greedy Search) để trích xuất JSON chính xác tuyệt đối.
        """
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample  # False = tắt temperature/ngẫu nhiên, tối ưu cho trích xuất JSON
            )

        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

    def run_on_text(self, text_content: str, max_new_tokens: int = 1024, do_sample: bool = False) -> str:
        """Chạy inference trực tiếp từ nội dung văn bản (file .txt)."""
        system_prompt = (
            "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. "
            "Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: "
            "loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text_content.strip()}]
            }
        ]
        return self.predict(messages, max_new_tokens=max_new_tokens, do_sample=do_sample)


def parse_json_from_response(raw_text: str) -> Dict[str, Any]:
    """Bóc tách chuỗi JSON chuẩn từ kết quả trả về của LLM."""
    cleaned = re.sub(r"^```json\s*|```$", "", raw_text.strip(), flags=re.MULTILINE | re.IGNORECASE).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_response": raw_text, "error": "JSON parse error"}


def main():
    parser = argparse.ArgumentParser(description="Chạy Qwen3-VL inference trực tiếp trên .txt hoặc .jsonl")
    parser.add_argument(
        "--input_path",
        type=str,
        default=r"E:\SGOD\21_9_2026\sgod_model_ocr\dataset_for_qwen\2026_313_50_2026_TT-BQP_context.txt",
        help="Đường dẫn đến file .txt, .jsonl hoặc thư mục chứa file .txt"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Tên mô hình Qwen3-VL"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_predictions",
        help="Thư mục lưu kết quả inference (dạng .json)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Số token tối đa cần sinh"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Bật sampling ngẫu nhiên (mặc định False: chạy Greedy Search chính xác 100%)"
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo mô hình
    runner = QwenInference(model_id=args.model_id)

    # 1. Nếu đầu vào là file .jsonl
    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        print(f"\n[*] Đang chạy trên file JSONL: {input_path}")
        out_jsonl = output_dir / f"{input_path.stem}_pred.jsonl"
        with open(input_path, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)
                infer_msgs = [m for m in item.get("messages", []) if m.get("role") != "assistant"]
                raw_pred = runner.predict(infer_msgs, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
                item["prediction"] = parse_json_from_response(raw_pred)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[✓] Hoàn thành. Kết quả lưu tại: {out_jsonl}")

    # 2. Nếu đầu vào là file .txt đơn lẻ hoặc thư mục chứa file .txt
    else:
        txt_files = [input_path] if (input_path.is_file() and input_path.suffix.lower() == ".txt") else (
            sorted(list(input_path.glob("*.txt"))) if input_path.is_dir() else []
        )

        if not txt_files:
            print(f"[!] Không tìm thấy file .txt nào tại: {input_path}")
            return

        print(f"\n[*] Tìm thấy {len(txt_files)} file .txt cần chạy inference.")
        for idx, tf in enumerate(txt_files, 1):
            print(f"[{idx}/{len(txt_files)}] Đang xử lý: {tf.name}...")
            content = tf.read_text(encoding="utf-8")
            raw_response = runner.run_on_text(content, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
            parsed_json = parse_json_from_response(raw_response)

            # In trực tiếp ra console để theo dõi
            print(f"[>] Kết quả trích xuất:\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}")

            # Lưu ra file .json
            out_file = output_dir / f"{tf.stem}_result.json"
            out_file.write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n[✓] Đã xử lý xong toàn bộ. Kết quả lưu tại: {output_dir}")


if __name__ == "__main__":
    main()