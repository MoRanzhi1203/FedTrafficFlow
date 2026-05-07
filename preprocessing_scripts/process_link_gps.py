# ==========================
# process_link_gps.py
# ==========================
# -*- coding: utf-8 -*-
"""
处理 link_gps 数据
1. 读取原始 CSV
2. 数据清洗（去重、缺失值）
3. 类型转换
4. 输出处理后的 CSV
"""
import pandas as pd
import os

os.makedirs('./data/processed', exist_ok=True)

# 读取原始 link_gps 数据
link_gps = pd.read_csv('./data/raw/link_gps.v2', sep='\t', names=['路段ID','经度','纬度'])

# 去重、缺失值处理
link_gps.drop_duplicates(subset=['路段ID'], inplace=True)
link_gps.dropna(subset=['路段ID','经度','纬度'], inplace=True)

# 类型转换
link_gps['路段ID'] = link_gps['路段ID'].astype(int)
link_gps['经度'] = link_gps['经度'].astype(float)
link_gps['纬度'] = link_gps['纬度'].astype(float)

# 保存处理后的 link_gps
link_gps.to_csv('./data/processed/link_gps_processed.csv', index=False)
print('link_gps 已处理并保存')
