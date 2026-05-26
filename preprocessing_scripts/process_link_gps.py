# ==========================
# process_link_gps.py
# ==========================
# -*- coding: utf-8 -*-

"""清洗原始 link_gps 记录，并导出供后续使用的标准化路段坐标数据。"""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT_DIR / "data" / "raw" / "link_gps.v2"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "link_gps_processed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取原始 link_gps 数据
link_gps = pd.read_csv(RAW_PATH, sep='\t', names=['路段ID','经度','纬度'])

# 去重、缺失值处理
link_gps.drop_duplicates(subset=['路段ID'], inplace=True)
link_gps.dropna(subset=['路段ID','经度','纬度'], inplace=True)

# 类型转换
link_gps['路段ID'] = link_gps['路段ID'].astype(int)
link_gps['经度'] = link_gps['经度'].astype(float)
link_gps['纬度'] = link_gps['纬度'].astype(float)

# 保存处理后的 link_gps
link_gps.to_csv(OUTPUT_PATH, index=False)
print(f'link_gps 已处理并保存到: {OUTPUT_PATH}')
