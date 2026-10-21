#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hậu xử lý (Post-Processing) chuẩn hóa siêu dữ liệu bóc tách từ LLM (Qwen).
Xử lý các lỗi phổ biến trong trích xuất văn bản hành chính Việt Nam:
1. co_quan_ban_hanh: Cắt bỏ Quốc hiệu/Tiêu ngữ dính kèm (Cộng hòa xã hội chủ nghĩa Việt Nam, Độc lập...), chuẩn hóa tên cơ quan.
2. so_ky_hieu: Sửa VVBHN -> VBHN, xóa khoảng trắng thừa trong mã, ưu tiên số VBHN nếu là văn bản hợp nhất.
3. loai_van_ban: Tự động phát hiện và gán 'VĂN BẢN HỢP NHẤT' nếu số hiệu là VBHN.
4. ngay_ban_hanh: Chuẩn hóa DD/MM/YYYY, tránh nhầm lẫn với ngày Công báo.
5. nguoi_ky: Làm sạch chức danh, quân hàm, giữ lại họ tên người ký.
"""

import re
import unicodedata
from typing import Dict, Any, Optional


def normalize_unicode(s: str) -> str:
    """Chuẩn hóa chuỗi về Unicode chuẩn dựng sẵn NFC."""
    if not s:
        return ""
    return unicodedata.normalize('NFC', str(s)).strip()


def clean_co_quan_ban_hanh(agency: str, so_ky_hieu: str = "", context: str = "") -> str:
    """
    Làm sạch cơ quan ban hành:
    - Loại bỏ triệt để quốc hiệu / tiêu ngữ dính kèm do lỗi đọc ngang dòng của OCR.
    - Chuẩn hóa tên các cơ quan nhà nước phổ biến.
    """
    if not agency:
        return ""
    
    text = normalize_unicode(agency)

    # 1. Cắt bỏ cụm Quốc hiệu (Cộng hòa xã hội chủ nghĩa Việt Nam) và Tiêu ngữ
    text = re.sub(r'\s*CỘNG\s*H[ÒO]A\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*ĐỘC\s*LẬP.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\_\-]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Xử lý trường hợp đặc biệt
    text_upper = text.upper()

    # Ngân hàng Nhà nước
    if text_upper in ["NGÂN HÀNG NHÀ NƯỚC", "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM"]:
        return "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM"

    # Kiểm toán Nhà nước (tránh nhầm với Tổng kiểm toán nhà nước)
    if text_upper in ["TỔNG KIỂM TOÁN NHÀ NƯỚC", "KIỂM TOÁN NHÀ NƯỚC"]:
        return "KIỂM TOÁN NHÀ NƯỚC"

    # Văn phòng Quốc hội (đối với VBHN do Văn phòng Quốc hội xác thực)
    if "VBHN-VPQH" in so_ky_hieu.upper() or "VBHN-LQ-VPQH" in so_ky_hieu.upper() or "VBHN-NQ-VPQH" in so_ky_hieu.upper():
        if text_upper in ["QUỐC HỘI", "ỦY BAN THƯỜNG VỤ QUỐC HỘI", "VĂN PHÒNG QUỐC HỘI"]:
            return "VĂN PHÒNG QUỐC HỘI"

    # Chính phủ
    if text_upper == "CHÍNH PHỦ":
        return "CHÍNH PHỦ"

    # Thủ tướng Chính phủ
    if text_upper in ["THỦ TƯỚNG", "THỦ TƯỚNG CHÍNH PHỦ"]:
        return "THỦ TƯỚNG CHÍNH PHỦ"

    return text


def clean_so_ky_hieu(code: str, context: str = "") -> str:
    """
    Làm sạch số, ký hiệu văn bản:
    - Sửa lỗi OCR: VVBHN -> VBHN.
    - Loại bỏ khoảng trắng thừa giữa số và chữ (VD: 20/2026/TT- NHNN -> 20/2026/TT-NHNN).
    - Cắt bỏ phần rác dính công báo (VD: '305/Ngày 03-06-2026 36').
    - Nếu là văn bản hợp nhất mà Qwen lấy nhầm số Luật gốc, tìm số VBHN trong context.
    """
    if not code:
        code = ""
    
    code = normalize_unicode(code)
    # Sửa VVBHN -> VBHN
    code = re.sub(r'VVBHN', 'VBHN', code, flags=re.IGNORECASE)
    # Xóa khoảng trắng
    code = re.sub(r'\s+', '', code)

    # Loại bỏ rác dính ngày công báo
    code = re.sub(r'/Ngày\d+.*$', '', code, flags=re.IGNORECASE)

    # Kiểm tra xem có phải văn bản hợp nhất không
    is_vbhn = "VBHN" in code.upper() or (context and "VĂN BẢN HỢP NHẤT" in context.upper())

    if is_vbhn and "VBHN" not in code.upper() and context:
        # Tìm mã VBHN thực sự trong context: dạng \d+/VBHN-... hoặc \d+/YYYY/VBHN-...
        match_vbhn = re.search(r'(\d+[\w\.\-]*/(?:[0-9]{4}/)?(?:VBHN|VVBHN)[\w\.\-]*)', context, flags=re.IGNORECASE)
        if match_vbhn:
            found_code = match_vbhn.group(1).replace("VVBHN", "VBHN").replace("vvbhn", "VBHN")
            code = re.sub(r'\s+', '', found_code)

    return code


def clean_loai_van_ban(loai_vb: str, so_ky_hieu: str = "", context: str = "") -> str:
    """Chuẩn hóa Loại văn bản, phát hiện Văn bản hợp nhất."""
    loai = normalize_unicode(loai_vb).upper()

    # Nếu số hiệu chứa VBHN hoặc ngữ cảnh có XÁC THỰC VĂN BẢN HỢP NHẤT -> Luôn là VĂN BẢN HỢP NHẤT
    if "VBHN" in so_ky_hieu.upper() or (context and "XÁC THỰC VĂN BẢN HỢP NHẤT" in context.upper()):
        return "Văn bản hợp nhất"

    mapping = {
        "QUYẾT ĐỊNH": "Quyết định",
        "NGHỊ ĐỊNH": "Nghị định",
        "THÔNG TƯ": "Thông tư",
        "NGHỊ QUYẾT": "Nghị quyết",
        "CHỈ THỊ": "Chỉ thị",
        "LUẬT": "Luật",
        "PHÁP LỆNH": "Pháp lệnh",
        "VĂN BẢN HỢP NHẤT": "Văn bản hợp nhất",
    }
    return mapping.get(loai, loai_vb.strip() if loai_vb else "")


def clean_ngay_ban_hanh(date_str: str, context: str = "", cong_bao_str: str = "") -> str:
    """
    Chuẩn hóa ngày ban hành về dạng DD/MM/YYYY.
    Nếu ngày bị nhầm với ngày Công báo, quét lại context để lấy ngày ban hành chính xác.
    """
    s = normalize_unicode(date_str)
    # DD-MM-YYYY -> DD/MM/YYYY
    s = re.sub(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', r'\1/\2/\3', s)
    # YYYY-MM-DD -> DD/MM/YYYY
    s = re.sub(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', r'\3/\2/\1', s)
    # Chuẩn hóa về 2 chữ số (1/5/2026 -> 01/05/2026)
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        d, mth, y = m.groups()
        s = f"{int(d):02d}/{int(mth):02d}/{y}"

    # Kiểm tra xem có bị nhầm sang ngày công báo không
    cb_date = ""
    if cong_bao_str:
        cb_m = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', str(cong_bao_str))
        if cb_m:
            cb_d, cb_mth, cb_y = cb_m.groups()
            cb_date = f"{int(cb_d):02d}/{int(cb_mth):02d}/{cb_y}"

    # Nếu ngày ban hành bị trùng ngày công báo và có context, quét ngày ban hành ở dòng địa danh
    if (not s or s == cb_date) and context:
        sign_date_m = re.search(
            r'(?:Hà\s*Nội|TP\.\s*Hồ\s*Chí\s*Minh|[A-ZĐđa-zà-ỹ\s]+),\s*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})',
            context,
            flags=re.IGNORECASE
        )
        if sign_date_m:
            sd, sm, sy = sign_date_m.groups()
            s = f"{int(sd):02d}/{int(sm):02d}/{sy}"

    return s


def clean_nguoi_ky(signer: str) -> str:
    """Làm sạch tên người ký, loại bỏ các chức danh dính kèm."""
    if not signer:
        return ""
    name = normalize_unicode(signer)

    # Loại bỏ tiền tố ký thay
    name = re.sub(r'^(?:KT\.|TM\.|TL\.|TUQ\.)\s*', '', name, flags=re.IGNORECASE)
    # Loại bỏ chức danh nếu dính vào đầu tên
    title_pattern = r'^(?:THỦ\s*TƯỚNG|PHÓ\s*THỦ\s*TƯỚNG|BỘ\s*TRƯỞNG|THỨ\s*TRƯỞNG|CHỦ\s*TỊCH|PHÓ\s*CHỦ\s*TỊCH|ĐẠI\s*TƯỚNG|THƯỢNG\s*TƯỚNG|TRUNG\s*TƯỚNG|THIẾU\s*TƯỚNG)\s*'
    name = re.sub(title_pattern, '', name, flags=re.IGNORECASE).strip()
    return name


def clean_cong_bao(cong_bao_str: str) -> str:
    """
    Làm sạch thông tin Công báo:
    - Loại bỏ số trang thường bị dính ở cuối dòng do nằm cùng line header
      (Ví dụ: 'CÔNG BÁO/Số 282/Ngày 23-05-2026 69' -> 'CÔNG BÁO/Số 282/Ngày 23-05-2026')
    """
    if not cong_bao_str or str(cong_bao_str).strip().lower() in ["none", "null", "nan", ""]:
        return ""
    
    s = normalize_unicode(cong_bao_str)
    
    # 1. Cắt bỏ số trang ở cuối sau năm 4 chữ số (VD: '...2026 69' -> '...2026')
    s = re.sub(r'(\d{4})\s+(?:trang\s*)?\d+[\.\s]*$', r'\1', s, flags=re.IGNORECASE)
    
    # 2. Loại bỏ dấu gạch ngang, chấm hoặc ký tự thừa ở cuối nếu có
    s = re.sub(r'[\s\.\-_]+$', '', s).strip()
    return s


def post_process_prediction(data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """
    Hàm tổng hợp hậu xử lý cho toàn bộ 1 kết quả dự đoán JSON từ mô hình Qwen.
    """
    if not isinstance(data, dict):
        return data

    cleaned = dict(data)

    # Làm sạch công báo trước để phục vụ cho các trường khác nếu cần
    cleaned["cong_bao"] = clean_cong_bao(cleaned.get("cong_bao", ""))

    so_kh = clean_so_ky_hieu(cleaned.get("so_ky_hieu", ""), context=context)
    cleaned["so_ky_hieu"] = so_kh

    cleaned["loai_van_ban"] = clean_loai_van_ban(
        cleaned.get("loai_van_ban", ""),
        so_ky_hieu=so_kh,
        context=context
    )

    cleaned["co_quan_ban_hanh"] = clean_co_quan_ban_hanh(
        cleaned.get("co_quan_ban_hanh", ""),
        so_ky_hieu=so_kh,
        context=context
    )

    cleaned["ngay_ban_hanh"] = clean_ngay_ban_hanh(
        cleaned.get("ngay_ban_hanh", ""),
        context=context,
        cong_bao_str=cleaned.get("cong_bao", "")
    )

    cleaned["nguoi_ky"] = clean_nguoi_ky(cleaned.get("nguoi_ky", ""))

    return cleaned


def extract_and_clean_json(raw_text: str, context: str = "") -> Dict[str, Any]:
    """
    Trích xuất JSON từ phản hồi dạng text/markdown của mô hình LLM và áp dụng hậu xử lý.
    """
    import json
    if not raw_text:
        return {}

    # Tìm khối JSON trong cặp ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Tìm cặp dấu ngoặc nhọn đầu tiên và cuối cùng
        m_brace = re.search(r'(\{.*\})', raw_text, re.DOTALL)
        json_str = m_brace.group(1) if m_brace else raw_text.strip()

    try:
        data = json.loads(json_str)
        return post_process_prediction(data, context=context)
    except Exception:
        return {}
