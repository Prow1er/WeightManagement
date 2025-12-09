"""
Optimized network.py – memory‑efficient version with restored logs and visualization
Includes:
✓ Chunked cosine similarity
✓ float32 embeddings
✓ Smaller batch size
✓ No full N×N matrices
✓ Restored detailed logging and visualization
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import logging
from config import *
import matplotlib.pyplot as plt
import os
import platform
import matplotlib.font_manager as fm

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class SemanticNetworkBuilder:
    def __init__(self, similarity_threshold=0.85, chunk_size=512):
        self.threshold = similarity_threshold
        self.embedding_model = None
        self.networks = {}
        self.global_network = None
        self.chunk_size = chunk_size

    def load_embedding_model(self):
        logger.info("加载嵌入模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"  使用设备: {device}")  # 恢复设备信息日志

        self.embedding_model = SentenceTransformer(
            NLP_CONFIG['embedding_model'],
            cache_folder=HF_CACHE,
            device=device
        )
        self.embedding_model.max_seq_length = NLP_CONFIG['max_seq_length']
        logger.info(f"  模型加载完成: {NLP_CONFIG['embedding_model']}")  # 恢复模型名称日志

    def compute_embeddings(self, texts, batch_size=32):
        logger.info(f"计算文本嵌入,文档数: {len(texts)}")
        logger.info(f"  批次大小: {batch_size}")  # 恢复批次大小日志

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(np.float32)  # 保留内存优化
        logger.info(f"  嵌入维度: {embeddings.shape}, dtype={embeddings.dtype}")  # 恢复详细维度日志
        return embeddings

    # 分块计算余弦相似度（保留优化）
    def generate_edges_chunked(self, embeddings):
        n = embeddings.shape[0]
        edges = []
        csize = self.chunk_size

        logger.info(f"使用分块相似度计算: chunk_size={csize}")

        for start in range(0, n, csize):
            end = min(start + csize, n)


            sim_chunk = cosine_similarity(embeddings[start:end], embeddings)

            for i in range(end - start):
                global_i = start + i
                row = sim_chunk[i]

                # 阈值过滤（避免生成完整矩阵）
                mask = row >= self.threshold
                mask[:global_i + 1] = False  # 避免自环和重复边

                js = np.nonzero(mask)[0]
                for j in js:
                    edges.append((global_i, j, float(row[j])))

        logger.info(f"  分块相似度生成边数: {len(edges)}")
        return edges

    def build_network_for_window(self, df_window, embeddings, window_id):
        logger.info(f"构建时间窗口 {window_id} 的网络…")

        n_docs = len(df_window)
        logger.info(f"  文档数: {n_docs}")  # 恢复文档数日志

        G = nx.Graph()
        indices = df_window.index.tolist()

        # 添加节点
        strategy_col = 'meta_strategy' if 'meta_strategy' in df_window.columns else 'strategy'
        for idx in indices:
            G.add_node(
                idx,
                strategy=df_window.loc[idx, strategy_col],
                sentiment=df_window.loc[idx, 'sentiment'],
                arousal=df_window.loc[idx, 'arousal']
            )

        # 分块生成边
        edges = self.generate_edges_chunked(embeddings)

        # 添加边
        for local_i, local_j, w in edges:
            G.add_edge(indices[local_i], indices[local_j], weight=w)

        # 恢复详细网络统计日志
        logger.info(f"  节点数: {G.number_of_nodes()}")
        logger.info(f"  边数: {G.number_of_edges()}")

        metrics = self.calculate_network_metrics(G)
        # 恢复网络指标日志
        logger.info(f"  密度: {metrics['density']:.4f}")
        logger.info(f"  平均聚类系数: {metrics['avg_clustering']:.4f}")

        return G, metrics

    def calculate_network_metrics(self, G):
        metrics = {}

        # 基本指标
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)

        # 聚类系数
        if G.number_of_edges() > 0:
            metrics['avg_clustering'] = nx.average_clustering(G)
        else:
            metrics['avg_clustering'] = 0

        # 连通分量
        components = list(nx.connected_components(G))
        metrics['n_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len)) if components else 0

        # 按策略分组的节点数
        strategy_counts = {}
        for node, data in G.nodes(data=True):
            strategy = data.get('strategy', 'Unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        metrics['strategy_distribution'] = strategy_counts

        return metrics

    def analyze_echo_chambers(self, G):
        logger.info("分析回音室效应...")

        # 动态获取所有策略类型
        strategies = set()
        for node, data in G.nodes(data=True):
            strategy = data.get('strategy', 'Unknown')
            if strategy != 'Unknown' and strategy != 'MS_Outlier':
                strategies.add(strategy)

        # 初始化连接统计
        strategy_connections = {s: {'internal': 0, 'external': 0} for s in strategies}

        for u, v in G.edges():
            strategy_u = G.nodes[u].get('strategy', 'Unknown')
            strategy_v = G.nodes[v].get('strategy', 'Unknown')

            if strategy_u == strategy_v and strategy_u in strategy_connections:
                strategy_connections[strategy_u]['internal'] += 1
            elif strategy_u in strategy_connections:
                strategy_connections[strategy_u]['external'] += 1
            elif strategy_v in strategy_connections:
                strategy_connections[strategy_v]['external'] += 1

        # 计算回音室指数 (内部连接 / 总连接)
        echo_chamber_index = {}
        for strategy, counts in strategy_connections.items():
            total = counts['internal'] + counts['external']
            if total > 0:
                echo_chamber_index[strategy] = counts['internal'] / total
            else:
                echo_chamber_index[strategy] = 0

        # 恢复回音室指数详情日志
        logger.info("  回音室指数:")
        for strategy, index in echo_chamber_index.items():
            logger.info(f"    {strategy}: {index:.3f}")

        return echo_chamber_index

    def process_pipeline(self, df):
        logger.info("=" * 60)  # 恢复分隔线日志
        logger.info("开始阶段三: 语义网络构建")
        logger.info("=" * 60)

        # 1. 加载模型
        self.load_embedding_model()

        # 2. 计算所有文档的嵌入
        texts = df['全文内容_cleaned'].fillna('').astype(str).tolist()
        all_embeddings = self.compute_embeddings(texts)

        # 3. 为每个时间窗口构建网络
        if 'time_window' in df.columns:
            windows = sorted(df['time_window'].unique())
            logger.info(f"\n共有 {len(windows)} 个时间窗口")  # 恢复时间窗口数量日志

            all_metrics = []

            for window in windows:
                df_window = df[df['time_window'] == window].copy()
                window_indices = df_window.index.tolist()

                # 提取该窗口的嵌入
                embeddings_window = all_embeddings[[
                    i for i, idx in enumerate(df.index) if idx in window_indices
                ]]

                # 构建网络
                G, metrics = self.build_network_for_window(
                    df_window, embeddings_window, window
                )

                # 分析回音室
                echo_index = self.analyze_echo_chambers(G)
                metrics['echo_chamber'] = echo_index

                # 保存网络
                self.networks[window] = G
                metrics['window'] = window
                all_metrics.append(metrics)

            # 4. 构建全局网络
            logger.info("\n构建全局网络...")
            self.global_network, global_metrics = self.build_network_for_window(
                df, all_embeddings, 'global'
            )
            global_echo = self.analyze_echo_chambers(self.global_network)

            # 5. 保存结果
            self.save_results(all_metrics, global_metrics, global_echo)

            return self.networks, all_metrics
        else:
            logger.warning("未找到时间窗口信息,只构建全局网络")  # 恢复警告日志
            self.global_network, global_metrics = self.build_network_for_window(
                df, all_embeddings, 'global'
            )
            return {}, [global_metrics]

    def save_results(self, window_metrics, global_metrics, global_echo):
        logger.info("\n保存网络分析结果...")

        # 保存窗口级指标
        metrics_df = pd.DataFrame(window_metrics)
        metrics_file = os.path.join(OUTPUT_FOLDER, 'stage3_network_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
        logger.info(f"  网络指标已保存: {metrics_file}")  # 恢复保存路径日志

        # 保存网络对象
        for window, G in self.networks.items():
            network_file = os.path.join(OUTPUT_FOLDER, f'stage3_network_{window}.gexf')
            nx.write_gexf(G, network_file)
        logger.info(f"  网络文件已保存 (GEXF格式) 至 {OUTPUT_FOLDER}")  # 恢复保存路径日志

        # 可视化网络演化
        self.visualize_network_evolution(window_metrics)

    # 恢复完整的可视化逻辑（包括字体设置）
    def visualize_network_evolution(self, metrics_list):
        logger.info("生成网络演化可视化...")

        def set_simsunb_font():
            """优先设置simsun（宋体加粗）字体"""
            # 定义simsun字体的可能路径
            simsun_paths = [
                './SIMSUN.TTC',
                './simsun.ttc',
                '../SIMSUN.TTC',
                '../simsun.ttc',
                '/root/weight2/SIMSUN.TTC',
                '/root/weight2/simsun.ttc',
            ]

            # 扩展路径：处理相对路径和用户目录
            expanded_paths = []
            for path in simsun_paths:
                expanded_path = os.path.expanduser(path)  # 处理 ~ 符号
                expanded_paths.append(expanded_path)
                # 添加大小写组合路径
                if 'SIMSUN' in path:
                    expanded_paths.append(expanded_path.replace('SIMSUN', 'simsun'))
                else:
                    expanded_paths.append(expanded_path.replace('simsun', 'SIMSUN'))

            # 去重
            expanded_paths = list(set(expanded_paths))

            # 查找simsun字体
            simsun_font_path = None
            for path in expanded_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    if path.lower().endswith('.ttc'):
                        simsun_font_path = path
                        break

            # 加载字体
            if simsun_font_path:
                try:
                    font_prop = fm.FontProperties(fname=simsun_font_path)
                    fm.fontManager.addfont(simsun_font_path)
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'Arial Unicode MS']
                    plt.rcParams['axes.unicode_minus'] = False
                    plt.rcParams['font.size'] = 10
                    logger.info(f"✅ 成功加载 simsun 字体: {simsun_font_path}")
                    return simsun_font_path
                except Exception as e:
                    logger.error(f"⚠️ simsun字体加载失败: {e}")
                    return None
            else:
                logger.warning("⚠️ 未找到 simsun.ttc 字体文件，尝试查找其他中文字体...")
                return None

        def find_chinese_font():
            """备用：自动查找系统中可用的中文字体"""
            system = platform.system()

            # 优先查找常见中文字体
            common_chinese_fonts = [
                'SimHei', 'Microsoft YaHei', 'Microsoft JhengHei',
                'PingFang SC', 'Heiti SC', 'Songti SC',
                'SimSun', 'FangSong', 'KaiTi'
            ]

            # 检查已安装字体
            installed_fonts = {f.name for f in fm.fontManager.ttflist}
            for font_name in common_chinese_fonts:
                if font_name in installed_fonts:
                    logger.info(f"✅ 找到系统中文字体: {font_name}")
                    return font_name

            # 查找字体文件
            fontpaths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            for fontpath in fontpaths:
                font_name = fm.FontProperties(fname=fontpath).get_name()
                if any(keyword in font_name.lower() for keyword in
                       ['sim', 'hei', 'song', 'fang', 'kai', 'pingfang', 'heiti']):
                    logger.info(f"✅ 通过系统查找找到中文字体: {fontpath}")
                    return fontpath

            logger.warning("⚠️ 未找到理想中文字体，尝试使用默认字体")
            return None

        # 设置中文字体
        chinese_font_path = set_simsunb_font()
        if not chinese_font_path:
            chinese_font = find_chinese_font()
            if chinese_font:
                if os.path.isfile(chinese_font):
                    font_prop = fm.FontProperties(fname=chinese_font)
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'Arial Unicode MS']
                else:
                    plt.rcParams['font.sans-serif'] = [chinese_font, 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False

        if len(metrics_list) < 2:
            logger.warning("时间窗口不足,跳过演化可视化")
            return

        metrics_df = pd.DataFrame(metrics_list)

        # 绘制4个子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 密度演化
        axes[0, 0].plot(range(len(metrics_df)), metrics_df['density'], marker='o')
        axes[0, 0].set_title('网络密度演化')
        axes[0, 0].set_xlabel('时间窗口')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 聚类系数演化
        axes[0, 1].plot(range(len(metrics_df)), metrics_df['avg_clustering'], marker='s', color='orange')
        axes[0, 1].set_title('平均聚类系数演化')
        axes[0, 1].set_xlabel('时间窗口')
        axes[0, 1].set_ylabel('聚类系数')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 节点数演化
        axes[1, 0].plot(range(len(metrics_df)), metrics_df['n_nodes'], marker='^', color='green')
        axes[1, 0].set_title('网络规模演化')
        axes[1, 0].set_xlabel('时间窗口')
        axes[1, 0].set_ylabel('节点数')
        axes[1, 0].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_FOLDER, 'stage3_network_evolution.png')
        plt.savefig(
            output_path,
            dpi=VISUALIZATION_CONFIG['figure_dpi']
        )
        plt.close()
        logger.info(f"  可视化已保存: {output_path}")  # 恢复可视化保存日志


def main():
    """主函数"""
    # 加载阶段二的数据
    input_file = os.path.join(OUTPUT_FOLDER, 'stage2_framework_data.csv')
    logger.info(f"加载数据: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # 运行阶段三
    builder = SemanticNetworkBuilder(
        similarity_threshold=NETWORK_CONFIG['similarity_threshold']
    )
    networks, metrics = builder.process_pipeline(df)

    logger.info("\n" + "=" * 60)
    logger.info("阶段三总结")
    logger.info("=" * 60)
    logger.info(f"构建网络数: {len(networks)}")  # 恢复总结日志


if __name__ == "__main__":
    main()