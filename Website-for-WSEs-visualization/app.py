import os
import re
import json
from io import BytesIO
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_file
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, AllChem
from PIL import Image, ImageDraw, ImageFont


RDLogger.DisableLog('rdApp.error')

# 初始化Flask应用
app = Flask(__name__)

# 设置相对路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))                 # 项目根（app.py 所在目录）
STATIC_DIR = os.path.join(BASE_DIR, 'static')
IMAGE_DIR = os.path.join(STATIC_DIR, 'images')                        # ./static/images
SPLASH_DIR = os.path.join(STATIC_DIR, 'splash')                       # ./static/splash
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Kmeans_4')                 # ./data/Kmeans_4

# 创建目录
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SPLASH_DIR, exist_ok=True)

_IMAGE_INDEX = None  # canonical_smiles -> filename

# 创建占位图片
def create_placeholder_image():

    img = Image.new('RGB', (300, 300), color='#666666')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
        draw.text((150, 150), "No Image Available", fill='#666666', anchor='mm', font=font)
    except:
        pass

    img_path = os.path.join(IMAGE_DIR, 'placeholder.png')
    img.save(img_path)


# 创建首页的全屏 Splash 图片（如果用户没有提供）
def create_splash_image(output_path, size=(1920, 1080)):
    
    w, h = size
    import random
    random.seed(42)

    # soft vertical gradient background
    base = Image.new('RGBA', (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(base)
    top = (250, 250, 252)
    bottom = (235, 242, 255)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b, 255))

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # pastel palette (match the site's cluster colors, but more airy)
    palette = [
        (230, 153, 167, 160),  # #E699A7
        (254, 221, 158, 150),  # #FEDD9E
        (166, 217, 192, 150),  # #A6D9C0
        (113, 167, 210, 150),  # #71A7D2
    ]

    # nodes
    nodes = []
    for _ in range(140):
        x = random.randint(int(w * 0.08), int(w * 0.92))
        y = random.randint(int(h * 0.10), int(h * 0.90))
        r = random.randint(6, 18)
        color = random.choice(palette)
        odraw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        nodes.append((x, y))

    # edges
    for _ in range(170):
        (x1, y1) = random.choice(nodes)
        (x2, y2) = random.choice(nodes)
        color = random.choice(palette)
        odraw.line((x1, y1, x2, y2), fill=(color[0], color[1], color[2], 90), width=2)

    composed = Image.alpha_composite(base, overlay).convert('RGB')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composed.save(output_path)


# Splash helpers: let you use splash.jpg / splash.jpeg / splash.png without editing templates
def pick_splash_filename():

    candidates = [
        os.path.join(SPLASH_DIR, 'splash.jpg'),
        os.path.join(SPLASH_DIR, 'splash.jpeg'),
        os.path.join(SPLASH_DIR, 'splash.png'),
    ]
    for abs_path in candidates:
        if os.path.exists(abs_path):
            # return path relative to /static so url_for('static', filename=...) works
            return os.path.relpath(abs_path, STATIC_DIR).replace('\\', '/')
    return 'splash/splash.jpg'


# 提取SMILES的函数
def extract_smiles_from_filename(filename):
    # 匹配下划线和点号之间的内容，提取SMILES式
    match = re.search(r'_([^_]+)\.', filename)
    if match:
        return match.group(1)
    return None


# 标准化SMILES
def standardize_smiles(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None


# 构建图片索引，支持不同格式图片
def build_image_index(image_dir_fs: str):

    idx = {}
    if not os.path.exists(image_dir_fs):
        return idx

    for filename in os.listdir(image_dir_fs):
        lower = filename.lower()
        if not lower.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        extracted = extract_smiles_from_filename(filename)
        if not extracted:
            continue

        mol = Chem.MolFromSmiles(extracted)
        if mol is None:
            # silently ignore (prevents parse spam)
            continue

        can = Chem.MolToSmiles(mol, canonical=True)
        idx[can] = filename

    return idx


# 获取图片路径
def get_image_path(smiles, image_dir):

    global _IMAGE_INDEX

    # 允许传入 IMAGE_DIR 或 './static/images/' 等，统一成绝对路径
    image_dir_fs = image_dir
    if not os.path.isabs(image_dir_fs):
        image_dir_fs = os.path.join(BASE_DIR, image_dir_fs)
    image_dir_fs = os.path.abspath(image_dir_fs)

    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = build_image_index(image_dir_fs)

    std_smiles = standardize_smiles(smiles)
    if std_smiles and std_smiles in _IMAGE_INDEX:
        return f"/static/images/{_IMAGE_INDEX[std_smiles]}"

    # 如果没有找到匹配的图片，使用哈希值作为备用方案（写到 ./static/images）
    img_filename = f"{abs(hash(smiles))}.png"
    img_path = os.path.join(IMAGE_DIR, img_filename)

    # 如果图片不存在，生成它
    if not os.path.exists(img_path):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(img_path)

            # ✅ update index so future lookups won't rescan
            std_smiles2 = standardize_smiles(smiles)
            if std_smiles2:
                if _IMAGE_INDEX is None:
                    _IMAGE_INDEX = {}
                _IMAGE_INDEX[std_smiles2] = img_filename
        else:
            return '/static/images/placeholder.png'

    return f'/static/images/{img_filename}'


# 加载四个聚类簇的数据
def load_cluster_data():
    clusters = []

    # 加载四个聚类簇的数据
    for i in range(4):
        # ✅ CSV 改成 ./data/Kmeans_4/cluster_{i+1}.csv
        csv_path = os.path.join(DATA_DIR, f'cluster_{i+1}.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Cluster'] = i+1  # 添加聚类标签，从1开始
            clusters.append(df)
            print(f"成功加载聚类簇 {i+1}，共 {len(df)} 个分子")
        else:
            print(f"警告: 文件 {csv_path} 不存在")

    # 合并所有聚类簇的数据
    if clusters:
        combined_df = pd.concat(clusters, ignore_index=True)
        print(f"总共加载 {len(combined_df)} 个分子")
        return combined_df
    else:
        # 如果没有找到任何聚类文件，使用示例数据
        print("使用示例数据")
        return generate_sample_data()


# 生成示例数据（如果不存在）
def generate_sample_data():

    data = {
        'SMILES': ['CCO', 'CC(=O)O', 'C1=CC=CC=C1', 'CNC', 'C1CCCCC1',
                  'C1=CC=C(C=C1)O', 'CCN(CC)CC', 'C1COCCO1', 'C1CCOC1', 'CC(C)CO'],
        'Es-Ea (eV)': [-2.3, -1.8, -3.1, -2.5, -2.9, -2.1, -1.9, -2.4, -2.6, -2.0],
        'LUMO_sol (eV)': [-1.2, -1.5, -0.9, -1.3, -1.1, -1.4, -1.6, -1.0, -1.2, -1.3],
        'HOMO_sol (eV)': [-5.2, -5.5, -4.9, -5.3, -5.1, -5.4, -5.6, -5.0, -5.2, -5.3],
        'Dielectric constant of solvents': [0.45, 0.38, 0.52, 0.42, 0.48, 0.40, 0.36, 0.50, 0.44, 0.41],
        'PC1': [1.2, 0.8, -0.5, 1.5, -1.0, 0.3, -0.7, 1.8, -0.2, 0.9],
        'PC2': [0.5, -0.3, 1.2, -0.8, 0.7, -1.1, 0.9, -0.4, 1.3, -0.6],
        'PC3': [-0.7, 1.1, 0.4, -1.2, 0.8, -0.5, 1.0, -0.9, 0.6, -1.3],
        'Cluster': [1, 1, 2, 2, 3, 3, 4, 4, 1, 2]  # 修改聚类标签为1-4
    }

    df = pd.DataFrame(data)

    # 为每个分子生成图像
    for smiles in df['SMILES']:
        img_filename = f"{abs(hash(smiles))}.png"
        img_path = os.path.join(IMAGE_DIR, img_filename)

        if not os.path.exists(img_path):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                img.save(img_path)

    return df


# 创建3D散点图
def create_3d_scatter(df):
    # 为四个聚类簇分配不同颜色
    colors = {
        1: '#E699A7',  # 红色 - cluster1
        2: '#FEDD9E',  # 黄色 - cluster2
        3: '#A6D9C0',  # 绿色 - cluster3
        4: '#71A7D2'   # 蓝色 - cluster4
    }

    # 确定使用的坐标轴
    x_col, y_col, z_col = 'PC1', 'PC2', 'PC3'

    # 检查列是否存在，如果不存在则使用替代列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Cluster']

    if x_col not in df.columns and len(numeric_cols) > 0:
        x_col = numeric_cols[0]

    if y_col not in df.columns and len(numeric_cols) > 1:
        y_col = numeric_cols[1]

    if z_col not in df.columns and len(numeric_cols) > 2:
        z_col = numeric_cols[2]

    print(f"使用以下坐标轴创建3D图: X={x_col}, Y={y_col}, Z={z_col}")

    # 创建3D散点图
    fig = go.Figure()

    # 为每个聚类簇添加一个轨迹
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]

        fig.add_trace(go.Scatter3d(
            x=cluster_df[x_col],
            y=cluster_df[y_col],
            z=cluster_df[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=colors.get(cluster_id, '#7f7f7f'),
                opacity=0.8
            ),
            name=f'Cluster {cluster_id}',
            text=cluster_df['SMILES'],
            hoverinfo='text',
            customdata=np.stack((
                cluster_df['SMILES'],
                cluster_df[x_col],
                cluster_df[y_col],
                cluster_df[z_col],
                cluster_df.get('Dielectric constant of solvents', 0),
                cluster_df.get('Es-Ea (eV)', 0),
                cluster_df.get('LUMO_sol (eV)', 0),
                cluster_df.get('HOMO_sol (eV)', 0)
            ), axis=-1)
        ))

    # 更新布局
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        title='3D Visualization of Molecular Clusters',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#000"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig


def mol_from_smiles_largest_fragment(smiles: str):
    """Parse SMILES and keep the largest fragment."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # If multiple fragments exist, keep the one with most heavy atoms.
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if frags:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        except Exception:
            pass
        return mol
    except Exception:
        return None


def _morgan_fp(mol, radius: int, nbits: int = 2048):

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

# 相似性搜索
def hybrid_similarity(query_mol, target_mol):

    if query_mol is None or target_mol is None:
        return 0.0, {
            'tanimoto_r2': 0.0,
            'tanimoto_r3': 0.0,
            'containment': 0.0,
            'substructure': False,
        }

    # Fingerprints
    fpq2 = _morgan_fp(query_mol, radius=2)
    fpt2 = _morgan_fp(target_mol, radius=2)
    fpq3 = _morgan_fp(query_mol, radius=3)
    fpt3 = _morgan_fp(target_mol, radius=3)

    tanimoto_r2 = DataStructs.TanimotoSimilarity(fpq2, fpt2)
    tanimoto_r3 = DataStructs.TanimotoSimilarity(fpq3, fpt3)

    containment_q_in_t = DataStructs.TverskySimilarity(fpq2, fpt2, 0.9, 0.1)
    containment_t_in_q = DataStructs.TverskySimilarity(fpt2, fpq2, 0.9, 0.1)
    containment = max(containment_q_in_t, containment_t_in_q)

    substructure = False
    try:
        if query_mol.GetNumHeavyAtoms() <= 24:
            substructure = target_mol.HasSubstructMatch(query_mol) or query_mol.HasSubstructMatch(target_mol)
    except Exception:
        substructure = False

    score = max(tanimoto_r2, 0.98 * tanimoto_r3, containment)
    if substructure:
        score = min(1.0, score + 0.05)

    meta = {
        'tanimoto_r2': float(tanimoto_r2),
        'tanimoto_r3': float(tanimoto_r3),
        'containment': float(containment),
        'substructure': bool(substructure),
    }
    return float(score), meta


# 路由定义
@app.route('/')
def index():
    return render_template('index.html', splash_file=pick_splash_filename())

@app.route('/data')
def data_page():
    return render_template('data.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

# API端点：获取聚类数据
@app.route('/api/cluster_data')

def get_cluster_data():

    df = load_cluster_data()
    fig = create_3d_scatter(df)

    # 转换为JSON格式
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 准备分子数据
    molecules = []
    for _, row in df.iterrows():
        image_path = get_image_path(row['SMILES'], IMAGE_DIR)
        molecules.append({
            'SMILES': row['SMILES'],
            'Image': image_path,
            'Relative binding energy (eV)': round(row.get('Es-Ea (eV)', 0), 2),
            'LUMO (eV)': round(row.get('LUMO_sol (eV)', 0), 2),
            'HOMO (eV)': round(row.get('HOMO_sol (eV)', 0), 2),
            'Dielectric constant': round(row.get('Dielectric constant of solvents', 0), 2),
            'Cluster': int(row.get('Cluster', 1)),
            'PCA1': float(row.get('PC1', 0)),
            'PCA2': float(row.get('PC2', 0)),
            'PCA3': float(row.get('PC3', 0))
        })

    return jsonify({
        'graph': graphJSON,
        'molecules': molecules
    })


# API端点：搜索分子
@app.route('/api/search', methods=['POST'])
def search_molecules():

    data = request.json
    query = data.get('query', '')
    threshold = data.get('threshold', 0.7)

    # 标准化查询SMILES，并取最大片段
    standardized_query = standardize_smiles(query)
    query_mol = mol_from_smiles_largest_fragment(standardized_query) if standardized_query else None

    if query_mol is None:
        return jsonify({'error': 'Invalid SMILES string'}), 400

    # 加载数据
    df = load_cluster_data()

    # 计算相似度（hybrid: tanimoto + containment + optional substructure boost）
    results = []
    for _, row in df.iterrows():
        target_mol = mol_from_smiles_largest_fragment(row.get('SMILES', ''))
        if target_mol is None:
            continue

        similarity, meta = hybrid_similarity(query_mol, target_mol)

        effective_threshold = float(threshold)
        if meta.get('substructure') and query_mol.GetNumHeavyAtoms() <= 12:
            effective_threshold = min(effective_threshold, max(0.35, 0.80 * float(threshold)))

        if similarity >= effective_threshold:
            image_path = get_image_path(row['SMILES'], IMAGE_DIR)
            results.append({
                'SMILES': row['SMILES'],
                'Image': image_path,
                'Relative binding energy (eV)': round(row.get('Es-Ea (eV)', 0), 2),
                'LUMO (eV)': round(row.get('LUMO_sol (eV)', 0), 2),
                'HOMO (eV)': round(row.get('HOMO_sol (eV)', 0), 2),
                'Dielectric constant': round(row.get('Dielectric constant of solvents', 0), 2),
                'Similarity': round(similarity, 4),
                'SimilarityDetails': meta,
                'Cluster': int(row.get('Cluster', 1)),
                'PCA1': float(row.get('PC1', 0)),
                'PCA2': float(row.get('PC2', 0)),
                'PCA3': float(row.get('PC3', 0))
            })

    results.sort(key=lambda x: x['Similarity'], reverse=True)
    return jsonify({'results': results})


# API端点：获取集群统计信息
@app.route('/api/cluster_stats')
def get_cluster_stats():
    df = load_cluster_data()

    # 计算集群统计
    cluster_counts = df['Cluster'].value_counts().to_dict()
    total_molecules = len(df)

    cluster_counts = {int(k): int(v) for k, v in cluster_counts.items()}

    # 计算属性统计
    properties = ['Es-Ea (eV)', 'LUMO_sol (eV)', 'HOMO_sol (eV)', 'Dielectric constant of solvents']
    stats = {}

    for prop in properties:
        if prop in df.columns:
            stats[prop] = {
                'min': float(df[prop].min()),
                'max': float(df[prop].max()),
                'mean': float(df[prop].mean()),
                'std': float(df[prop].std())
            }

    # 按集群计算平均属性值
    cluster_means = {}
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        cluster_means[int(cluster_id)] = {}

        for prop in properties:
            if prop in cluster_df.columns:
                cluster_means[int(cluster_id)][prop] = float(cluster_df[prop].mean())

    return jsonify({
        'total_molecules': int(total_molecules),
        'cluster_counts': cluster_counts,
        'property_stats': stats,
        'cluster_means': cluster_means
    })


# API端点：下载数据
@app.route('/api/download', methods=['POST'])
def download_data():
    data = request.json
    selected_smiles = data.get('smiles', [])
    cluster = data.get('cluster', None)

    df = load_cluster_data()

    if cluster is not None:
        # 下载整个簇
        result_df = df[df['Cluster'] == cluster]
        filename = f'cluster_{cluster}.csv'
    elif selected_smiles:
        # 下载选中的分子
        result_df = df[df['SMILES'].isin(selected_smiles)]
        filename = 'selected_molecules.csv'
    else:
        # 下载全部数据
        result_df = df
        filename = 'all_molecules.csv'

    # 创建内存中的CSV文件
    output = BytesIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


# 运行应用
if __name__ == '__main__':
    # 创建占位图片
    placeholder_path = os.path.join(IMAGE_DIR, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        create_placeholder_image()

    # 创建首页 Splash 图片（默认 static/splash/splash.jpg，也支持jpeg/png）
    os.makedirs(SPLASH_DIR, exist_ok=True)
    splash_candidates = [
        os.path.join(SPLASH_DIR, 'splash.jpg'),
        os.path.join(SPLASH_DIR, 'splash.jpeg'),
        os.path.join(SPLASH_DIR, 'splash.png'),
    ]
    if not any(os.path.exists(p) for p in splash_candidates):
        create_splash_image(splash_candidates[0])

    _IMAGE_INDEX = build_image_index(IMAGE_DIR)

    # 确保数据已加载
    df = load_cluster_data()

    for _, row in df.iterrows():
        smiles = row['SMILES']
        _ = get_image_path(smiles, IMAGE_DIR)

    app.run(debug=True, port=5000)