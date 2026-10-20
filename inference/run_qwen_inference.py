#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chạy Qwen Inference trực tiếp từ file .txt hoặc .jsonl đã có sẵn.
Không cần trích xuất hay tạo định dạng prompt thủ công.
"""
import re
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


class QwenInference:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = None):
        """Khởi tạo mô hình và processor Qwen."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Đang tải model: {model_id} trên {self.device}...")

        # 1. Tải Processor (hoặc Tokenizer)
        try:
            self.qwen_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.is_vl = True
        except Exception:
            from transformers import AutoTokenizer
            self.qwen_processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.is_vl = False

        # 2. Tải Model
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            if Qwen3VLForConditionalGeneration is not None and "qwen3" in model_id.lower():
                try:
                    self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_id,
                        dtype="auto",
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                except Exception:
                    self.qwen_model = None

            if not hasattr(self, "qwen_model") or self.qwen_model is None:
                if self.is_vl:
                    try:
                        self.qwen_model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            device_map="auto" if self.device == "cuda" else None,
                            trust_remote_code=True
                        )
                    except Exception:
                        self.qwen_model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            device_map="auto" if self.device == "cuda" else None,
                            trust_remote_code=True
                        )
                else:
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải model {model_id}: {e}")

        if self.device != "cuda":
            self.qwen_model.to(self.device)

        self.qwen_model.eval()
        print(f"[✓] Đã tải model thành công vào device: {self.qwen_model.device}")

    def predict(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024, temperature: float = 0.01) -> str:
        """Thực hiện inference với messages đầu vào."""
        inputs = self.qwen_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.qwen_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if temperature > 0 and False:  # chỉ dùng khi do_sample=True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True

        with torch.no_grad():
            generated_ids = self.qwen_model.generate(
                **inputs,
                **gen_kwargs
            )

        trimmed_ids = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
        response = self.qwen_processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        return response

    def run_on_text(self, text_content: str, max_new_tokens: int = 1024) -> str:
        """Chạy inference trực tiếp từ nội dung văn bản (file .txt đã có sẵn prompt và schema)."""
        prompt = text_content.strip()

        # System prompt đồng bộ chuẩn xác với train.jsonl
        system_prompt = (
            "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. "
            "Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: "
            "loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
        )

        if self.is_vl:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        return self.predict(messages, max_new_tokens=max_new_tokens)


def parse_json_from_response(raw_text: str) -> Dict[str, Any]:
    """Bóc tách chuỗi JSON chuẩn từ kết quả trả về của LLM."""
    cleaned = re.sub(r'```json\s*|\s*```', '', raw_text, flags=re.IGNORECASE).strip()
    match = re.search(r'(\{[\s\S]*\})', cleaned)
    if match:
        json_str = match.group(1)
    else:
        json_str = cleaned
    try:
        return json.loads(json_str)
    except Exception:
        return {"raw_response": raw_text, "error": "JSON parse error"}


def main():
    parser = argparse.ArgumentParser(description="Chạy Qwen inference trực tiếp trên dữ liệu .txt hoặc .jsonl")
    parser.add_argument(
        "--input_path",
        type=str,
        default=r"E:\SGOD\21_9_2026\sgod_model_ocr\dataset_for_qwen\2026_282_840_QĐ-TTg_context.txt",
        help="Đường dẫn đến file .txt, thư mục chứa các file .txt hoặc file .jsonl"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Tên mô hình (ví dụ: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-VL-2B-Instruct)"
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
            for idx, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_msgs = item.get("messages", [])
                infer_msgs = [m for m in raw_msgs if m.get("role") != "assistant"]

                raw_pred = runner.predict(infer_msgs, max_new_tokens=args.max_new_tokens)
                parsed_pred = parse_json_from_response(raw_pred)
                
                print(f"\n--- Mẫu {idx} ---")
                print(json.dumps(parsed_pred, ensure_ascii=False, indent=2))

                item["prediction"] = parsed_pred
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[✓] Hoàn thành. Kết quả lưu tại: {out_jsonl}")

    # 2. Nếu đầu vào là file .txt đơn lẻ hoặc thư mục chứa các file .txt
    else:
        txt_files: List[Path] = []
        if input_path.is_file() and input_path.suffix.lower() == ".txt":
            txt_files = [input_path]
        elif input_path.is_dir():
            txt_files = sorted(list(input_path.glob("*.txt")))

        if not txt_files:
            print(f"[!] Không tìm thấy file .txt nào tại: {input_path}")
            return

        print(f"\n[*] Tìm thấy {len(txt_files)} file .txt cần chạy inference.")
        for idx, tf in enumerate(txt_files, 1):
            print(f"\n[{idx}/{len(txt_files)}] Đang xử lý: {tf.name}...")
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read()

            raw_response = runner.run_on_text(content, max_new_tokens=args.max_new_tokens)
            parsed_json = parse_json_from_response(raw_response)

            print(f"[>] Kết quả JSON bóc tách từ Qwen:")
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))

            # Lưu trực tiếp ra file .json
            out_file = output_dir / f"{tf.stem}_result.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(parsed_json, out_f, ensure_ascii=False, indent=2)

        print(f"\n[✓] Đã xử lý xong toàn bộ. Kết quả lưu dạng .json trong thư mục: {output_dir}")


if __name__ == "__main__":
    main()
