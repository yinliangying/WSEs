import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
import sqlite3
import os
from flask import Flask, render_template, request, jsonify, send_file
import json
import base64
from io import BytesIO
import re
import shutil
import csv
from PIL import Image, ImageDraw, ImageFont

# 初始化Flask应用
app = Flask(__name__)

# 创建数据目录
if not os.path.exists('/Users/guozhenning/Desktop/SRT/WSEs_0827'):
    os.makedirs('/Users/guozhenning/Desktop/SRT/WSEs_0827')

# 创建占位图片
def create_placeholder_image():
    # 创建简单的占位图片
    img = Image.new('RGB', (300, 300), color='#666666')
    draw = ImageDraw.Draw(img)
    
    try:
        # 尝试使用默认字体
        font = ImageFont.load_default()
        # 添加文本
        draw.text((150, 150), "No Image Available", fill='#666666', anchor='mm', font=font)
    except:
        # 如果字体加载失败，仍然创建图片但没有文本
        pass
    
    # 保存图片
    img_path = os.path.join('static', 'images', 'placeholder.png')
    img.save(img_path)

# 提取SMILES的函数
def extract_smiles_from_filename(filename):
    # 匹配下划线和点号之间的内容
    match = re.search(r'_([^_]+)\.', filename)
    if match:
        return match.group(1)
    return None

# 获取图片路径
def get_image_path(smiles, image_dir):
    # 首先尝试查找匹配的图片文件
    for filename in os.listdir(image_dir):
        extracted_smiles = extract_smiles_from_filename(filename)
        if extracted_smiles:
            # 标准化SMILES进行比较
            std_extracted = standardize_smiles(extracted_smiles)
            std_smiles = standardize_smiles(smiles)
            if std_extracted and std_smiles and std_extracted == std_smiles:
                return f'/static/images/{filename}'
    
    # 如果没有找到匹配的图片，使用哈希值作为备用方案
    img_filename = f"{abs(hash(smiles))}.png"
    img_path = os.path.join('static', 'images', img_filename)
    
    # 如果图片不存在，生成它
    if not os.path.exists(img_path):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(img_path)
        else:
            # 如果无法生成分子图像，使用占位图片
            return '/static/images/placeholder.png'
    
    return f'/static/images/{img_filename}'

# 加载四个聚类簇的数据
def load_cluster_data():
    clusters = []
    
    # 加载四个聚类簇的数据
    for i in range(4):
        csv_path = f'/Users/guozhenning/Desktop/SRT/KMeans_4/cluster_{i+1}.csv'
        
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
    image_dir = '/Users/guozhenning/Desktop/SRT/WSEs_0827'
    for smiles in df['SMILES']:
        img_filename = f"{abs(hash(smiles))}.png"
        img_path = os.path.join(image_dir, img_filename)
        
        if not os.path.exists(img_path):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                img.save(img_path)
    
    return df

# 创建3D散点图
def create_3d_scatter(df):
    # 科学期刊风格的配色方案 - 为四个聚类簇分配不同颜色
    colors = {
        1: '#E699A7',  # 红色 - 聚类1
        2: '#FEDD9E',  # 黄色 - 聚类2
        3: '#A6D9C0',  # 绿色 - 聚类3
        4: '#71A7D2'   # 蓝色 - 聚类4
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
                color=colors.get(cluster_id, '#7f7f7f'),  # 使用预定义颜色，默认为灰色
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
    
    # 更新布局以适应科研风格
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

# 标准化SMILES
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

# 计算SMILES相似度
def calculate_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 and mol2:
            fp1 = FingerprintMol(mol1)
            fp2 = FingerprintMol(mol2)
            return FingerprintSimilarity(fp1, fp2)
        return 0
    except:
        return 0

# 路由定义
@app.route('/')
def index():
    return render_template('index.html')

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
        image_path = get_image_path(row['SMILES'], '/Users/guozhenning/Desktop/SRT/WSEs_0827')
        molecules.append({
            'SMILES': row['SMILES'],
            'Image': image_path,
            'Relative binding energy (eV)': round(row.get('Es-Ea (eV)', 0), 2),
            'LUMO (eV)': round(row.get('LUMO_sol (eV)', 0), 2),
            'HOMO (eV)': round(row.get('HOMO_sol (eV)', 0), 2),
            'Dielectric constant': round(row.get('Dielectric constant of solvents', 0), 2),
            'Cluster': int(row.get('Cluster', 1)),  # 默认值改为1
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
    
    # 标准化查询SMILES
    standardized_query = standardize_smiles(query)
    
    if not standardized_query:
        return jsonify({'error': 'Invalid SMILES string'}), 400
    
    # 加载数据
    df = load_cluster_data()
    
    # 计算相似度
    results = []
    for _, row in df.iterrows():
        similarity = calculate_similarity(standardized_query, row['SMILES'])
        if similarity >= threshold:
            image_path = get_image_path(row['SMILES'], '/Users/guozhenning/Desktop/SRT/WSEs_0827')
            results.append({
                'SMILES': row['SMILES'],
                'Image': image_path,
                'Relative binding energy (eV)': round(row.get('Es-Ea (eV)', 0), 2),
                'LUMO (eV)': round(row.get('LUMO_sol (eV)', 0), 2),
                'HOMO (eV)': round(row.get('HOMO_sol (eV)', 0), 2),
                'Dielectric constant': round(row.get('Dielectric constant of solvents', 0), 2),
                'Similarity': round(similarity, 4),
                'Cluster': int(row.get('Cluster', 1)),  # 默认值改为1
                'PCA1': float(row.get('PC1', 0)),
                'PCA2': float(row.get('PC2', 0)),
                'PCA3': float(row.get('PC3', 0))
            })
    
    # 按相似度排序
    results.sort(key=lambda x: x['Similarity'], reverse=True)
    
    return jsonify({'results': results})

# API端点：获取集群统计信息
@app.route('/api/cluster_stats')
def get_cluster_stats():
    df = load_cluster_data()
    
    # 计算集群统计
    cluster_counts = df['Cluster'].value_counts().to_dict()
    total_molecules = len(df)
    
    # 将numpy类型转换为Python内置类型
    cluster_counts = {int(k): int(v) for k, v in cluster_counts.items()}
    
    # 计算属性统计
    properties = ['Es-Ea (eV)', 'LUMO_sol (eV)', 'HOMO_sol (eV)', 'Dielectric constant of solvents']
    stats = {}
    
    for prop in properties:
        if prop in df.columns:
            # 确保所有值都是Python内置类型
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
    # 确保静态目录存在
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    
    # 创建占位图片
    placeholder_path = os.path.join('static', 'images', 'placeholder.png')
    if not os.path.exists(placeholder_path):
        create_placeholder_image()
    
    # 确保数据已加载
    df = load_cluster_data()
    
    # 原始图像目录
    original_image_dir = '/Users/guozhenning/Desktop/SRT/WSEs_0827'
    
    # 为所有分子处理图像
    for _, row in df.iterrows():
        smiles = row['SMILES']
        
        # 首先尝试在原始目录中查找匹配的图像
        matched_image = None
        if os.path.exists(original_image_dir):
            for filename in os.listdir(original_image_dir):
                # 从文件名中提取SMILES
                extracted_smiles = extract_smiles_from_filename(filename)
                if extracted_smiles:
                    # 标准化SMILES进行比较
                    std_extracted = standardize_smiles(extracted_smiles)
                    std_smiles = standardize_smiles(smiles)
                    if std_extracted and std_smiles and std_extracted == std_smiles:
                        matched_image = filename
                        break
        
        # 确定目标图像路径
        if matched_image:
            # 如果找到匹配的图像，复制到静态目录
            src_path = os.path.join(original_image_dir, matched_image)
            dst_path = os.path.join('static', 'images', matched_image)
            
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        else:
            # 如果没有找到匹配的图像，使用RDKit生成
            img_filename = f"{abs(hash(smiles))}.png"
            img_path = os.path.join('static', 'images', img_filename)
            
            if not os.path.exists(img_path):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    img.save(img_path)
    
    app.run(debug=True, port=5000)



