
import os
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config
from dotenv import load_dotenv


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

BUCKET_NAME = "congbao-vietnam-dataset-bucket"
ENDPOINT_URL = "https://s3.hf.co/baodetrai"

OUTPUT_DIR = PROJECT_ROOT / "dataset_congbao_download"
EXPORT_METADATA_CSV = True


# =========================================================
# LOAD CREDENTIALS
# =========================================================

load_dotenv(ENV_FILE)

access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    raise RuntimeError(
        f"Không tìm thấy credentials trong {ENV_FILE}\n"
        "Kiểm tra AWS_ACCESS_KEY_ID và AWS_SECRET_ACCESS_KEY."
    )


# =========================================================
# CONNECT TO HUGGING FACE S3
# =========================================================

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(
        region_name="us-east-1",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    ),
)


# =========================================================
# LIST ALL OBJECTS
# =========================================================

def list_all_objects():
    print("Đang lấy danh sách file từ bucket...")

    paginator = s3.get_paginator("list_objects_v2")
    objects = []

    for page in paginator.paginate(Bucket=BUCKET_NAME):
        objects.extend(
            obj for obj in page.get("Contents", [])
            if obj["Key"] and not obj["Key"].endswith("/")
        )

    print(f"Tìm thấy {len(objects):,} file.")
    return objects


# =========================================================
# DOWNLOAD DATASET
# =========================================================

def download_all(objects):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(objects)
    downloaded = 0
    skipped = 0
    failed = []

    total_bytes = sum(obj["Size"] for obj in objects)

    print(f"Tổng dung lượng: {total_bytes / 1024**3:.2f} GiB")
    print(f"Thư mục lưu: {OUTPUT_DIR.resolve()}\n")

    for index, obj in enumerate(objects, start=1):
        key = obj["Key"]
        expected_size = obj["Size"]

        relative_path = PurePosixPath(key)

        # Chặn đường dẫn không hợp lệ
        if relative_path.is_absolute() or ".." in relative_path.parts:
            print(f"[SKIP] Đường dẫn không hợp lệ: {key}")
            failed.append(key)
            continue

        local_path = OUTPUT_DIR.joinpath(*relative_path.parts)

        if not local_path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
            print(f"[SKIP] Đường dẫn nằm ngoài output: {key}")
            failed.append(key)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Bỏ qua file đã tải đủ dung lượng
        if (
            local_path.is_file()
            and local_path.stat().st_size == expected_size
        ):
            skipped += 1
            print(f"[{index}/{total}] [ĐÃ CÓ] {key}")
            continue

        try:
            print(
                f"[{index}/{total}] [TẢI] {key} "
                f"({expected_size / 1024**2:.2f} MiB)"
            )

            s3.download_file(
                BUCKET_NAME,
                key,
                str(local_path),
            )

            actual_size = local_path.stat().st_size

            if actual_size != expected_size:
                raise IOError(
                    f"Sai dung lượng: mong đợi {expected_size}, "
                    f"nhận được {actual_size}"
                )

            downloaded += 1
            print(f"    OK: {local_path.name}")

        except Exception as exc:
            print(f"    ERROR: {exc}")
            failed.append(key)

    print("\n" + "=" * 50)
    print("KẾT QUẢ")
    print("=" * 50)
    print(f"Tổng file:       {total}")
    print(f"Tải mới:         {downloaded}")
    print(f"Bỏ qua file cũ:  {skipped}")
    print(f"File lỗi:        {len(failed)}")

    if failed:
        failed_file = OUTPUT_DIR / "download_failed.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"Danh sách lỗi: {failed_file}")
        print("Bạn có thể chạy lại script để thử tải lại.")
    else:
        print("Đã tải đủ các file trong danh sách.")


# =========================================================
# EXPORT PARQUET TO CSV
# =========================================================

def export_metadata_csv():
    if not EXPORT_METADATA_CSV:
        return

    parquet_path = OUTPUT_DIR / "metadata.parquet"
    csv_path = OUTPUT_DIR / "metadata.csv"

    if not parquet_path.is_file():
        print("Không tìm thấy metadata.parquet; bỏ qua chuyển CSV.")
        return

    try:
        import pandas as pd

        print("\nĐang chuyển metadata.parquet sang CSV...")

        df = pd.read_parquet(parquet_path)
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Đã tạo: {csv_path}")
        print(f"Số dòng: {len(df):,}")
        print(f"Cột: {df.columns.tolist()}")

    except Exception as exc:
        print(f"Chuyển CSV thất bại: {exc}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 50)
    print("HUGGING FACE DATASET DOWNLOADER")
    print("=" * 50)

    objects = list_all_objects()

    if not objects:
        print("Bucket không có file hoặc không trả về object nào.")
        return

    download_all(objects)
    export_metadata_csv()

    print("\nHoàn tất.")


if __name__ == "__main__":
    main()
