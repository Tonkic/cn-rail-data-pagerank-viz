import json
import csv
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import geopandas as gpd
from adjustText import adjust_text

# --- 数据文件 ---
LINE_DATA_FILE = "line.csv"
STATION_COORDS_CSV = "cnstation.csv"
CHINA_MAP_SHP = "ne_50m_admin_0_countries.shp"
# ---

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def build_graph_from_csv(line_csv_file, graph):
    """
    加载 line.csv 并从 'src' 和 'dst' 列构建图。
    """
    try:
        with open(line_csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            print(f"  检测到线路文件 {line_csv_file} 为 [CSV] 格式。")

            for row in reader:
                try:
                    src_station = row.get("src")
                    dst_station = row.get("dst")

                    if src_station and dst_station:
                        graph.add_edge(src_station, dst_station)

                except KeyError:
                    pass

        if graph.number_of_nodes() == 0:
            print(f"  警告: 图为空。未能从 {line_csv_file} 解析出任何线路。")

        return True

    except FileNotFoundError:
        print(f"错误：找不到线路文件 '{line_csv_file}'", file=sys.stderr)
        return False
    except Exception as e:
        print(f"解析线路文件 {line_csv_file} 时出错: {type(e).__name__} - {e}", file=sys.stderr)
        return False

def load_coordinates_from_csv(coords_csv_file):
    """
    加载 cnstation.csv, 并去除站名末尾的'站'字以提高匹配率。
    """
    pos = {}
    try:
        with open(coords_csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            print(f"  检测到坐标文件 {coords_csv_file} 为 [CSV] 格式。")
            for row in reader:
                try:
                    station_name = row.get("站名")
                    lon_str = row.get("WGS84_Lng")
                    lat_str = row.get("WGS84_Lat")

                    if station_name and lon_str and lat_str:
                        if station_name.endswith('站'):
                            station_name = station_name[:-1] # 例如 "北京站" -> "北京"
                        pos[station_name] = (float(lon_str), float(lat_str))
                except (ValueError, TypeError, KeyError): pass
        if not pos: print(f"  警告: 坐标字典为空。")
        return pos
    except FileNotFoundError:
        print(f"错误：找不到坐标文件 '{coords_csv_file}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"解析坐标文件 {coords_csv_file} 时出错: {e}", file=sys.stderr)
        return None
# --- (函数定义结束) ---


# --- 主脚本开始 ---
print(f"1. 正在从 {LINE_DATA_FILE} 加载铁路图...")
G = nx.DiGraph()
if not build_graph_from_csv(LINE_DATA_FILE, G):
    sys.exit()
print(f"   图加载完毕: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边。")

print(f"2. 正在从 {STATION_COORDS_CSV} 加载车站坐标...")
pos = load_coordinates_from_csv(STATION_COORDS_CSV)
if not pos:
    print("   未能加载坐标数据，程序退出。")
    sys.exit()
print(f"   坐标加载完毕: {len(pos)} 个车站拥有坐标。")

print(f"3. 正在过滤图 (只保留既有线路又有坐标的车站)...")
nodes_in_graph = set(G.nodes())
nodes_with_coords = set(pos.keys())
valid_nodes = list(nodes_in_graph.intersection(nodes_with_coords))
if not valid_nodes:
    print("错误：线路图中的车站与坐标文件中的车站没有任何交集。")
    sys.exit()
G_filtered = G.subgraph(valid_nodes)
pos_filtered = {node: coords for node, coords in pos.items() if node in valid_nodes}
print(f"   过滤后，图中有 {G_filtered.number_of_nodes()} 个车站。")

print("4. 正在计算 PageRank...")
pagerank_scores = nx.pagerank(G_filtered)

print("5. 准备可视化参数 (使用对数缩放)...")
scores = np.array([pagerank_scores.get(node, 0) for node in G_filtered.nodes()])
node_colors = scores
sizes_log = np.log10(scores + 1e-9)
min_log_size = np.min(sizes_log)
max_log_size = np.max(sizes_log)
norm_sizes = (sizes_log - min_log_size) / (max_log_size - min_log_size)
min_display_size = 5
max_display_size = 300
node_sizes_scaled = norm_sizes * (max_display_size - min_display_size) + min_display_size

print("6. 筛选 Top 15 的车站用于标注...")
sorted_stations = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
labels_to_draw = {}
print("--- Top 15 重要车站 (PageRank) ---")
for i, (station, score) in enumerate(sorted_stations[:15]):
    labels_to_draw[station] = station
    print(f"{i+1}. {station} (Score: {score:.6f})")

print("7. 正在绘制图形 (带地图背景和防重叠标签)...")
# --- 关键修改：设置 Figure 的透明背景 ---
fig, ax = plt.subplots(figsize=(15, 15), facecolor='none')
# --- 修改结束 ---

# 1. 绘制地图背景 (zorder=1)
try:
    world = gpd.read_file(CHINA_MAP_SHP)
    china_map = world[world.NAME == 'China']
    china_map.plot(ax=ax, color='#DDDDDD', edgecolor='#999999', zorder=1)
except Exception as e:
    print(f"警告：无法加载地图文件 '{CHINA_MAP_SHP}'。 {e}", file=sys.stderr)

# 2. 绘制边 (zorder=2)
nx.draw_networkx_edges(
    G_filtered, pos_filtered,
    ax=ax,
    edge_color="#777777",
    width=0.3,
    alpha=0.3
)

# 3. 绘制节点 (zorder=3)
nx.draw_networkx_nodes(
    G_filtered, pos_filtered,
    ax=ax,
    node_size=node_sizes_scaled,
    node_color=node_colors,
    cmap=plt.cm.plasma,
    alpha=0.6,
    nodelist=G_filtered.nodes(),
    edgecolors='#000000',
    linewidths=0.5
)

# 4. 准备标签 (zorder=4)
texts = []
for station, (lon, lat) in pos_filtered.items():
    if station in labels_to_draw:
        texts.append(plt.text(lon, lat, station,
                              fontsize=8,
                              fontfamily='SimHei',
                              zorder=4,
                              # 注意：bbox 的 facecolor 设置为 'white' 会创建一个白色矩形背景，
                              # 如果需要标签本身也透明，可以调整 alpha，但通常白色背景更易读
                              bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')
                             ))

# 5. 使用 adjustText 自动调整标签位置
adjust_text(texts,
            ax=ax,
            force_points=0.5,
            force_text=0.5,
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5))

# --- 关键修改：设置 Title 的颜色以在透明背景下可见 ---
plt.title("中国铁路网络 PageRank 地理可视化", fontsize=30, color='black') # <-- 明确指定颜色
# --- 修改结束 ---

ax.set_xlim(70, 140)
ax.set_ylim(15, 55)
plt.axis("off")

# --- 可选：如果你想保存成文件并且背景透明，请取消下面这行的注释 ---
# plt.savefig("railway_pagerank_transparent.png", transparent=True, bbox_inches='tight', dpi=300)

plt.show()