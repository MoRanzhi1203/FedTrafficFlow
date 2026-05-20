# ========================== 
# process_rnsd.py
# ==========================
# -*- coding: utf-8 -*-

"""清洗原始路网属性数据，推导起终点坐标，并导出处理后的 rnsd 数据。"""

import pandas as pd
import numpy as np
import os

os.makedirs('./data/processed', exist_ok=True)

# 读取原始 rnsd 数据
rnsd = pd.read_csv('./data/raw/road_network_sub-dataset.v2', sep='\t')

# 列名映射
column_mapping = {
    'link_id': '路段ID',
    'width': '宽度',
    'direction': '方向',
    'snodeid': '起始节点ID',
    'enodeid': '结束节点ID',
    'length': '长度',
    'speedclass': '速度等级',
    'lanenum': '车道数'
}
rnsd = rnsd.rename(columns=column_mapping)

# 去重、缺失值处理
rnsd.drop_duplicates(subset=['路段ID'], inplace=True)
rnsd.dropna(subset=['路段ID','起始节点ID','结束节点ID','长度','方向'], inplace=True)

# 类型转换
rnsd['路段ID'] = rnsd['路段ID'].astype(int)
for col in ['宽度','长度','速度等级','车道数','方向']:
    if col in rnsd.columns:
        rnsd[col] = pd.to_numeric(rnsd[col], errors='coerce')

# 计算起始/结束节点经纬度（向量化）
lat = rnsd['纬度'].to_numpy() if '纬度' in rnsd.columns else np.zeros(len(rnsd))
lon = rnsd['经度'].to_numpy() if '经度' in rnsd.columns else np.zeros(len(rnsd))
distance = rnsd['长度'].to_numpy()
direction = rnsd['方向'].to_numpy()

half_distance = distance / 2
lat_rad = np.radians(lat)

# 起点坐标
start_lat = np.where(direction == 1, lat + half_distance / 111.32,
             np.where(direction == 2, lat - half_distance / 111.32, lat))
start_lon = np.where(direction == 3, lon + half_distance / (111.32 * np.cos(lat_rad)),
             np.where(direction == 4, lon - half_distance / (111.32 * np.cos(lat_rad)), lon))

# 终点坐标（方向反转）
rev_dir = np.where(direction == 1, 2,
           np.where(direction == 2, 1,
           np.where(direction == 3, 4,
           np.where(direction == 4, 3, direction))))
end_lat = np.where(rev_dir == 1, lat + half_distance / 111.32,
           np.where(rev_dir == 2, lat - half_distance / 111.32, lat))
end_lon = np.where(rev_dir == 3, lon + half_distance / (111.32 * np.cos(lat_rad)),
           np.where(rev_dir == 4, lon - half_distance / (111.32 * np.cos(lat_rad)), lon))

# 添加计算好的经纬度列
rnsd['start_lat'] = np.round(start_lat,6)
rnsd['start_lon'] = np.round(start_lon,6)
rnsd['end_lat'] = np.round(end_lat,6)
rnsd['end_lon'] = np.round(end_lon,6)

# 保存处理后的 rnsd 数据
rnsd.to_csv('./data/processed/rnsd_processed.csv', index=False)
print('rnsd 已处理并保存到: ./data/processed/rnsd_processed.csv')
