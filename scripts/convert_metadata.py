
from pathlib import Path
import pandas as pd

# Đường dẫn dataset đã tải về
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset_congbao_download"

parquet_path = DATASET_DIR / "metadata.parquet"
csv_path = DATASET_DIR / "metadata.csv"

# Kiểm tra file nguồn
if not parquet_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file: {parquet_path}"
    )

# Đọc Parquet
df = pd.read_parquet(parquet_path)

# Xuất CSV, hỗ trợ tiếng Việt khi mở bằng Excel
df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

print("Chuyển đổi thành công!")
print(f"Số dòng: {len(df):,}")
print(f"Các cột: {df.columns.tolist()}")
print(f"File CSV: {csv_path.resolve()}")
