"""
阶段二: 认知框架识别与情感量化 (方案C: 分层聚类动态识别元策略)
export HF_ENDPOINT=https://hf-mirror.com
"""
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.cluster import KMeans
import logging
from config import *

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def safe_convert(obj):

    if isinstance(obj, dict):
        return {safe_convert(k): safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class CognitiveFrameworkExtractor:
    """认知框架提取器 - 动态元策略识别"""

    def __init__(self):
        self.embedding_model = None
        self.topic_model = None
        self.topic_to_meta_strategy = {}  # 细粒度主题 -> 元策略映射
        self.meta_strategy_info = {}      # 元策略详细信息
        self.n_meta_strategies = 0        # 自动发现的元策略数量

    def load_embedding_model(self):
        """加载嵌入模型"""
        logger.info("加载中文嵌入模型...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")

        self.embedding_model = SentenceTransformer(
            NLP_CONFIG['embedding_model'],
            cache_folder=HF_CACHE,
            device=device
        )
        self.embedding_model.max_seq_length = NLP_CONFIG['max_seq_length']

        logger.info("嵌入模型加载完成")

    def build_topic_model(self):
        """构建BERTopic模型"""
        logger.info("构建BERTopic模型...")

        # 向量化器
        vectorizer = CountVectorizer(
            max_features=2000,
            min_df=3,
            ngram_range=(1, 2),
            stop_words=list(STOP_WORDS)
        )

        # BERTopic配置
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer,
            min_topic_size=BERTOPIC_CONFIG['min_topic_size'],
            nr_topics=BERTOPIC_CONFIG['nr_topics'],
            top_n_words=BERTOPIC_CONFIG['top_n_words'],
            calculate_probabilities=BERTOPIC_CONFIG['calculate_probabilities'],
            language='chinese',
            verbose=True
        )

        logger.info("BERTopic模型构建完成")

    def fit_topics(self, texts):
        """
        拟合主题模型 (包含强制拆分巨型主题的逻辑)
        """
        logger.info(f"开始主题建模,文档数: {len(texts)}")

        # 1. 初次训练模型
        topics, probs = self.topic_model.fit_transform(texts)

        # --- 新增: 强制拆分巨型主题逻辑 ---
        logger.info("检查是否存在过大的基础主题...")

        # 将 topics 转换为 numpy 数组方便操作
        new_topics = np.array(topics)
        total_docs = len(texts)
        unique_topics, counts = np.unique(new_topics, return_counts=True)

        # 定义巨型主题阈值 (例如: 单个主题超过总量的 30%)
        GIANT_TOPIC_THRESHOLD = 0.30

        # 标记是否发生了更新
        has_updates = False
        current_max_id = max(new_topics)  # 当前最大的主题ID

        for topic_id, count in zip(unique_topics, counts):
            # 跳过噪音类(-1)
            if topic_id == -1:
                continue

            ratio = count / total_docs
            if ratio > GIANT_TOPIC_THRESHOLD:
                logger.warning(f"  发现基础主题 #{topic_id} 过大 ({ratio:.1%}), 正在强制拆分...")

                # 1. 获取该主题下所有文档的索引
                indices = np.where(new_topics == topic_id)[0]
                subset_texts = [texts[i] for i in indices]

                # 2. 为子集计算嵌入 (为了准确聚类)
                # 虽然有点耗时，但针对子集通常很快
                subset_embeddings = self.embedding_model.encode(
                    subset_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                # 3. 使用 K-Means 强制拆分
                # 拆分数量策略: 如果特别大(>50%)拆4个，否则拆3个
                n_splits = 4 if ratio > 0.5 else 3

                kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                sub_labels = kmeans.fit_predict(subset_embeddings)

                # 4. 分配新的主题ID
                # sub_label=0 的保留原ID，其他的分配新ID
                for local_idx, sub_lab in enumerate(sub_labels):
                    global_idx = indices[local_idx]
                    if sub_lab > 0:
                        # 分配新 ID (current_max_id + 1, +2, ...)
                        new_id = current_max_id + sub_lab
                        new_topics[global_idx] = new_id

                # 更新 current_max_id 以防下一个巨型主题ID冲突
                current_max_id += (n_splits - 1)
                has_updates = True

                logger.info(f"    -> 主题 #{topic_id} 已拆分为 {n_splits} 个子主题")

        if has_updates:
            logger.info("应用拆分更新，重新计算主题关键词 (c-TF-IDF)...")
            # 这一步非常重要: 让 BERTopic 知道 ID 变了，需要重新计算关键词
            self.topic_model.update_topics(texts, new_topics)
            # 更新返回的 topics 列表
            topics = new_topics.tolist()

            # 重新统计一下
            n_topics_new = len(set(topics)) - 1
            logger.info(f"拆分后主题数量: {n_topics_new}")

        # --- 逻辑结束 ---

        n_topics = len(set(topics)) - 1
        outliers = sum(1 for t in topics if t == -1)

        logger.info(f"主题建模完成:")
        logger.info(f"  最终主题数量: {n_topics}")
        logger.info(f"  异常文档: {outliers} ({outliers / len(topics) * 100:.1f}%)")

        return topics, probs

    def extract_behavior_features(self, topic_id, texts):
        """
        为主题提取行为特征向量

        返回: numpy array, shape=(5,)
        """
        # 获取该主题的所有文本
        topic_texts = [texts[i] for i, t in enumerate(self.topic_model.topics_) if t == topic_id]

        if len(topic_texts) == 0:
            return np.zeros(8)

        # 合并为一个大文本用于分析
        combined_text = ' '.join(topic_texts[:1000])  # 取前100条避免太长

        # 特征1: 方法复杂度 (0-1, 越高越复杂)
        complexity_keywords = ['计算', '公式', '科学', '研究', '数据', '测量', '记录', '营养素', '宏量']
        simple_keywords = ['就是', '只要', '直接', '简单', '轻松', '一招', '秘诀']
        complexity = sum(1 for kw in complexity_keywords if kw in combined_text)
        simplicity = sum(1 for kw in simple_keywords if kw in combined_text)
        complexity_score = complexity / (complexity + simplicity + 1)

        # 特征2: 时间跨度 (0-1, 越高越长期)
        long_term_keywords = ['坚持', '长期', '习惯', '一辈子', '终身', '循序渐进', '慢慢', '稳定']
        short_term_keywords = ['快速', '三天', '一周', '速成', '暴瘦', '月瘦', '立竿见影', '马上']
        long_term = sum(1 for kw in long_term_keywords if kw in combined_text)
        short_term = sum(1 for kw in short_term_keywords if kw in combined_text)
        timespan_score = long_term / (long_term + short_term + 1)

        # 特征3: 产品依赖度 (0-1, 越高越依赖)
        product_keywords = ['药', '代餐', '产品', '购买', '链接', '店铺', '品牌', '推荐款',
                            '司美格鲁肽', '奥利司他', '酵素', '左旋肉碱', '仪器', '肽']
        product_count = sum(1 for kw in product_keywords if kw in combined_text)
        product_score = min(1.0, product_count / 10)

        # 特征4: 科学性 (0-1, 越高越科学)
        scientific_keywords = ['研究', '证明', '科学', '医生', '论文', '实验', '证据', '原理',
                               '热量', '蛋白质', '碳水', '代谢', '基础代谢', 'BMI', '体脂率']
        unscientific_keywords = ['排毒', '宿便', '清肠', '酸碱', '负卡', '燃脂', '神奇']
        scientific = sum(1 for kw in scientific_keywords if kw in combined_text)
        unscientific = sum(1 for kw in unscientific_keywords if kw in combined_text)
        scientific_score = scientific / (scientific + unscientific + 1)

        # 特征5: 态度倾向 (-1到1, 负面到正面)
        positive_keywords = ['成功', '有效', '推荐', '开心', '满意', '瘦了', '健康']
        negative_keywords = ['失败', '反弹', '放弃', '痛苦', '焦虑', '没用', '骗人']
        neutral_keywords = ['尝试', '考虑', '可能', '据说', '听说']
        positive = sum(1 for kw in positive_keywords if kw in combined_text)
        negative = sum(1 for kw in negative_keywords if kw in combined_text)
        attitude_score = (positive - negative) / (positive + negative + 1)

        # 新增特征6: 社交属性 (0-1, 越高越强调社交/打卡)
        social_keywords = ['打卡', '互相监督', '减肥搭子', '一起', '组队', '分享', '晒']
        social_score = sum(1 for kw in social_keywords if kw in combined_text) / 10

        # 新增特征7: 数据导向 (0-1, 越高越强调量化追踪)
        data_keywords = ['数据', '记录', 'App', '称重', '测量', '监测', '追踪']
        data_score = sum(1 for kw in data_keywords if kw in combined_text) / 10

        # 新增特征8: 专业指导 (0-1, 越高越依赖专业指导)
        prof_keywords = ['教练', '营养师', '医生', '专业', '指导', '咨询', '定制']
        prof_score = sum(1 for kw in prof_keywords if kw in combined_text) / 10

        # 归一化到 [0, 1]
        attitude_score = (attitude_score + 1) / 2

        return np.array([
            complexity_score, timespan_score, product_score, scientific_score,
            attitude_score, social_score, data_score, prof_score
        ])

    def map_to_meta_strategies(self, topics, texts, sentiments, arousals):
        """
        将细粒度主题映射到元策略 (方案C: 分层聚类 + 强制多样性约束)

        流程:
        1. 获取所有主题的关键词向量
        2. 计算主题间的语义相似度
        3. 使用层次聚类自动归纳为K个元策略
        4. 【新增】检查是否存在过大的聚类，进行强制拆分
        5. 分析每个元策略的语义特征
        """
        logger.info("=" * 60)
        logger.info("动态识别元策略 (分层聚类 + 强制多样性约束)")
        logger.info("=" * 60)

        # 获取主题信息
        topic_info = self.topic_model.get_topic_info()
        valid_topics = [t for t in topic_info['Topic'].values if t != -1]

        if len(valid_topics) < 2:
            logger.warning("有效主题数少于2个,无法进行元策略聚类")
            return {}

        logger.info(f"细粒度主题数: {len(valid_topics)}")

        # 步骤1: 构建主题的混合特征向量（关键词20% + 行为60% + 情感20%）
        logger.info("\n步骤1: 构建主题混合特征向量...")
        topic_feature_vectors = {}
        topic_keywords_text = {}

        # ... (保留原有的特征向量构建代码不变) ...
        # --- start copy from original ---
        topic_keyword_embeddings = {}
        for topic_id in valid_topics:
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue
            keywords = [word for word, _ in topic_words[:10]]
            keywords_text = ' '.join(keywords)
            topic_keywords_text[topic_id] = keywords_text
            keyword_emb = self.embedding_model.encode(keywords_text, convert_to_numpy=True)
            topic_keyword_embeddings[topic_id] = keyword_emb

        keyword_emb_array = np.array(list(topic_keyword_embeddings.values()))
        keyword_emb_mean = keyword_emb_array.mean(axis=0)
        keyword_emb_std = keyword_emb_array.std(axis=0) + 1e-8
        normalized_keyword_embs = {}
        for i, topic_id in enumerate(topic_keyword_embeddings.keys()):
            normalized_keyword_embs[topic_id] = (topic_keyword_embeddings[
                                                     topic_id] - keyword_emb_mean) / keyword_emb_std

        for topic_id in valid_topics:
            if topic_id not in normalized_keyword_embs:
                continue
            keyword_features = normalized_keyword_embs[topic_id] * 0.2
            behavior_features = self.extract_behavior_features(topic_id, texts) * 0.6
            topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
            if len(topic_indices) > 0:
                avg_sentiment = np.mean([sentiments[i] for i in topic_indices])
                avg_arousal = np.mean([arousals[i] for i in topic_indices])
            else:
                avg_sentiment, avg_arousal = 0, 0
            sentiment_norm = (avg_sentiment + 1) / 2
            emotion_features = np.array([sentiment_norm, avg_arousal]) * 0.2
            behavior_expanded = np.repeat(behavior_features, len(keyword_features) // 5 + 1)[:len(keyword_features)]
            emotion_expanded = np.repeat(emotion_features, len(keyword_features) // 2 + 1)[:len(keyword_features)]
            mixed_vector = keyword_features + behavior_expanded + emotion_expanded
            topic_feature_vectors[topic_id] = mixed_vector
        # --- end copy ---

        # 步骤2: 计算主题间相似度
        logger.info("\n步骤2: 计算主题间相似度...")
        topic_ids = list(topic_feature_vectors.keys())
        vectors = np.array([topic_feature_vectors[tid] for tid in topic_ids])

        similarity_matrix = cosine_similarity(vectors)

        # 预计算距离矩阵 (用于两阶段聚类)
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        distance_matrix = 1 - similarity_matrix
        distance_matrix[distance_matrix < 0] = 0
        np.fill_diagonal(distance_matrix, 0)

        # 步骤3: 层次聚类自动发现元策略 (初次聚类)
        logger.info("\n步骤3: 层次聚类识别元策略 (初筛)...")

        n_samples = len(topic_ids)
        min_k = 4
        max_k_safe = min(6, n_samples - 1)  # 稍微放宽上限

        if min_k > max_k_safe:
            best_n_clusters = max_k_safe
        else:
            best_n_clusters = self._determine_optimal_clusters(
                similarity_matrix, min_clusters=min_k, max_clusters=max_k_safe
            )

        clustering = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            metric='precomputed',
            linkage='average'
        )
        meta_strategy_labels = clustering.fit_predict(distance_matrix)

        # 步骤3.5: 强制多样性约束 (Splitting oversized clusters)
        logger.info("\n步骤3.5: 检查并拆分过大的聚类 (多样性约束)...")

        final_labels = meta_strategy_labels.copy()
        unique_labels, counts = np.unique(meta_strategy_labels, return_counts=True)

        # 定义阈值：如果一个策略包含了超过 40% 的主题，且该组内主题数 >= 4，则强制拆分
        MAX_CLUSTER_RATIO = 0.4
        next_label_id = max(unique_labels) + 1

        for label, count in zip(unique_labels, counts):
            ratio = count / n_samples

            if ratio > MAX_CLUSTER_RATIO :
                logger.warning(
                    f"  发现巨型聚类 (ID={label}): 包含 {count}/{n_samples} ({ratio:.1%}) 个主题，正在强制拆分...")

                # 获取该聚类内所有主题在距离矩阵中的索引
                indices_in_cluster = np.where(meta_strategy_labels == label)[0]

                # 提取子距离矩阵 (使用 np.ix_ 进行网格切片)
                sub_distance_matrix = distance_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]

                # 决定拆分成几个：如果非常大 (>10个主题) 拆成3个，否则拆成2个
                n_sub_clusters = 4 if count > 7 else 3
                n_sub_clusters = min(n_sub_clusters, count)
                n_sub_clusters = max(n_sub_clusters, 3)

                sub_clustering = AgglomerativeClustering(
                    n_clusters=n_sub_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                sub_labels = sub_clustering.fit_predict(sub_distance_matrix)

                # 更新全局标签
                # sub_labels 的顺序对应 indices_in_cluster 的顺序
                for i, sub_lab in enumerate(sub_labels):
                    original_idx = indices_in_cluster[i]
                    # 将子聚类 0 保留在原标签，其他子聚类分配新标签
                    if sub_lab > 0:
                        final_labels[original_idx] = next_label_id + (sub_lab - 1)

                logger.info(
                    f"    -> 已拆分为 {n_sub_clusters} 个子策略，新增标签 ID: {[label] + [next_label_id + i for i in range(n_sub_clusters - 1)]}")
                next_label_id += (n_sub_clusters - 1)

        # 重新整理标签，使其连续 (0, 1, 2...)
        unique_final_labels = np.unique(final_labels)
        label_map = {old: new for new, old in enumerate(unique_final_labels)}
        final_labels = np.array([label_map[x] for x in final_labels])
        best_n_clusters = len(unique_final_labels)

        logger.info(f"  最终元策略数量: {best_n_clusters}")

        # 步骤4: 构建主题-元策略映射
        logger.info("\n步骤4: 构建主题-元策略映射...")
        self.topic_to_meta_strategy = {}

        for idx, topic_id in enumerate(topic_ids):
            meta_strategy_id = f"MS{final_labels[idx]}"
            self.topic_to_meta_strategy[topic_id] = meta_strategy_id

        # 步骤5: 分析每个元策略的特征 (逻辑不变)
        # 步骤5: 分析每个元策略的特征（更新行为特征部分）
        logger.info("\n步骤5: 分析元策略语义特征...")
        self.meta_strategy_info = {}

        for meta_id in range(best_n_clusters):
            meta_strategy_name = f"MS{meta_id}"
            member_topics = [tid for tid, ms in self.topic_to_meta_strategy.items() if ms == meta_strategy_name]

            all_keywords = []
            for topic_id in member_topics:
                topic_words = self.topic_model.get_topic(topic_id)
                all_keywords.extend([word for word, _ in topic_words[:5]])

            from collections import Counter
            keyword_freq = Counter(all_keywords)
            top_keywords = [word for word, _ in keyword_freq.most_common(10)]

            # 提取新的6维度行为特征
            behavior_profiles = []
            for topic_id in member_topics:
                behavior_vec = self.extract_behavior_features(topic_id, texts)
                behavior_profiles.append(behavior_vec)

            if behavior_profiles:
                avg_behavior = np.mean(behavior_profiles, axis=0)
                behavior_profile = {
                    'exercise': float(avg_behavior[0]),  # 运动主导
                    'diet': float(avg_behavior[1]),  # 饮食控制
                    'medical': float(avg_behavior[2]),  # 医学干预
                    'special_group': float(avg_behavior[3]),  # 特殊人群
                    'tech_quant': float(avg_behavior[4]),  # 技术量化
                    'innovation': float(avg_behavior[5])  # 创新减肥
                }
            else:
                behavior_profile = None

            semantic_name = self._infer_meta_strategy_name(top_keywords, behavior_profile)

            self.meta_strategy_info[meta_strategy_name] = {
                'id': meta_strategy_name,
                'semantic_name': semantic_name,
                'member_topics': member_topics,
                'top_keywords': top_keywords,
                'n_topics': len(member_topics),
                'behavior_profile': behavior_profile
            }

            logger.info(f"\n  {meta_strategy_name} ({semantic_name}):")
            logger.info(f"    包含主题: {member_topics}")
            logger.info(f"    核心关键词: {', '.join(top_keywords)}")
            if behavior_profile:
                logger.info(f"    行为特征: 运动主导={behavior_profile['exercise']:.2f}, "
                            f"饮食控制={behavior_profile['diet']:.2f}, "
                            f"医学干预={behavior_profile['medical']:.2f}, "
                            f"特殊人群={behavior_profile['special_group']:.2f}, "
                            f"技术量化={behavior_profile['tech_quant']:.2f}, "
                            f"创新减肥={behavior_profile['innovation']:.2f}")
            # --- end copy ---

        outlier_indices = [i for i, t in enumerate(topics) if t == -1]

        if len(outlier_indices) > 0:
            logger.info(f"检测到 {len(outlier_indices)} 个 Outlier 文档，正在重新分配...")

            # 计算所有元策略的中心向量（对其 member_topics 的 feature vector 求均值）
            meta_centers = {}
            for ms_id, info in self.meta_strategy_info.items():
                member_topics = info['member_topics']
                member_vectors = [topic_feature_vectors[t] for t in member_topics]
                center_vec = np.mean(member_vectors, axis=0)
                meta_centers[ms_id] = center_vec

            # 为每个 outlier 文本计算嵌入
            outlier_texts = [texts[i] for i in outlier_indices]
            outlier_embs = self.embedding_model.encode(outlier_texts, convert_to_numpy=True)

            for idx, emb in zip(outlier_indices, outlier_embs):
                # 计算与每个元策略中心的 cosine similarity
                sims = {
                    ms: cosine_similarity(emb.reshape(1, -1), center.reshape(1, -1))[0][0]
                    for ms, center in meta_centers.items()
                }
                # 选择最相似的元策略
                best_ms = max(sims, key=sims.get)

                # 赋值给该文档
                self.topic_to_meta_strategy[-1] = best_ms  # Outlier 也视为该策略
                logger.debug(f"Outlier 文档 {idx} -> 分配至 {best_ms}")

        logger.info("所有 Outlier 已完成重新分配，不再保留 MS_Outlier")

        self.n_meta_strategies = best_n_clusters

        logger.info("\n" + "=" * 60)
        logger.info(f"元策略识别完成! 共发现 {self.n_meta_strategies} 个元策略")
        logger.info("=" * 60)

        return self.topic_to_meta_strategy

    def _determine_optimal_clusters(self, similarity_matrix, min_clusters=2, max_clusters=6):
        """
        确定最优聚类数
        使用轮廓系数 (Silhouette Score)
        """
        from sklearn.metrics import silhouette_score
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        distance_matrix = 1 - similarity_matrix
        distance_matrix[distance_matrix < 0] = 0
        np.fill_diagonal(distance_matrix, 0)
        best_score = -1
        best_n = min_clusters

        logger.info("  评估不同聚类数的效果...")

        for n in range(min_clusters, max_clusters + 1):
            clustering = AgglomerativeClustering(
                n_clusters=n,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

            # 计算轮廓系数
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            logger.info(f"    K={n}: 轮廓系数={score:.3f}")

            if score > best_score:
                best_score = score
                best_n = n

        logger.info(f"  最优聚类数: K={best_n} (轮廓系数={best_score:.3f})")
        return best_n

    def _infer_meta_strategy_name(self, keywords, behavior_profile=None):
        """根据关键词和新行为特征推断元策略的语义名称"""
        if behavior_profile:
            # 基于新行为特征的规则
            if behavior_profile['exercise'] > 0.6:
                return "运动主导型"
            elif behavior_profile['diet'] > 0.6:
                return "饮食控制型"
            elif behavior_profile['medical'] > 0.5:
                return "医学干预型"
            elif behavior_profile['special_group'] > 0.5:
                return "特殊人群适配型"
            elif behavior_profile['tech_quant'] > 0.5:
                return "技术量化型"
            elif behavior_profile['innovation'] > 0.5:
                return "创新方法型"
            # 混合类型判断
            elif behavior_profile['exercise'] > 0.4 and behavior_profile['diet'] > 0.4:
                return "运动饮食结合型"
            elif behavior_profile['tech_quant'] > 0.3 and behavior_profile['innovation'] > 0.3:
                return "科技驱动创新型"

        # 关键词匹配逻辑（作为补充）
        keywords_str = ' '.join(keywords).lower()
        patterns = {
            '运动主导型': [
                '运动', '健身', '跑步', '锻炼','健身房', '撸铁', '力量训练', '增肌减脂', 'HIIT', '有氧运动',
                '跑步', '游泳', '瑜伽', '普拉提', '私教', '训练计划','习惯养成',
                '运动消耗', '卡路里消耗', '运动打卡','管住嘴迈开腿', '运动'
            ],
            '饮食控制型': [
                '饮食', '节食', '卡路里', '热量','热量缺口', '轻断食', '生酮饮食', '低碳饮食', '碳水循环','均衡饮食',
                '饮食记录', '卡路里计算', '营养配比', '餐盘法则','长期坚持','习惯养成',
                '控糖', '低GI', '膳食纤维', '饱腹感', '戒糖', '控油','生活方式','饮食',
            ],
            '医学干预型': [
                '医生', '药物', '手术', '治疗','司美格鲁肽', 'GLP-1', '减肥针', '二甲双胍', '奥利司他',
                '医生建议', '内分泌科', '营养科', '肥胖门诊', '医学减重',
                '处方药', '药物辅助', '医疗监督', '肽'
            ],
            '特殊人群适配型': [
                '孕妇', '老人', '糖尿病', '术后','产后', '哺乳期', '中老年', '学生',
                '上班族', '久坐', 'PCOS', '甲减', '代谢慢', '易胖体质',
                '体质调理', '内分泌调节'
            ],
            '技术量化型': [
                'app', '数据', '监测', '打卡','BMI', '体脂率', '基础代谢', 'TDEE', '热量计算器',
                '体重秤', '智能秤', 'App记录', '数据追踪', '体重曲线',
                '围度测量', '体成分分析', '科学监测'
            ],
            '创新方法型': ['新方法', '创新', '研究发现', '个性化','创新体重管理', '创新', '新方式']
        }

        scores = {}
        for name, pattern_words in patterns.items():
            score = sum(1 for word in pattern_words if word in keywords_str)
            if score > 0:
                scores[name] = score

        return max(scores, key=scores.get) if scores else "综合策略型"

    def calculate_sentiment(self, texts):
        """计算情感得分"""
        logger.info("计算情感得分...")

        # 简化版情感分析 - 基于关键词
        positive_words = ['成功', '瘦了', '开心', '满意', '有效', '推荐', '坚持']
        negative_words = ['失败', '反弹', '放弃', '痛苦', '饿', '没用', '骗人']

        sentiments = []
        arousals = []

        for text in texts:
            text = str(text).lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            # 极性: -1到1
            polarity = (pos_count - neg_count) / max(1, pos_count + neg_count)

            # 唤醒度: 基于情绪词密度
            arousal = (pos_count + neg_count) / max(1, len(text) / 100)

            sentiments.append(polarity)
            arousals.append(min(1.0, arousal))  # 归一化到[0,1]

        logger.info(f"  平均极性: {np.mean(sentiments):.3f}")
        logger.info(f"  平均唤醒度: {np.mean(arousals):.3f}")

        return sentiments, arousals

    def build_observation_matrix(self, df, topics, sentiments, arousals):
        """构建观点-时间分布矩阵 (支持动态策略数)"""
        logger.info("构建观点-时间分布矩阵...")

        # 添加主题和元策略标签
        df['topic'] = topics
        df['meta_strategy'] = df['topic'].map(self.topic_to_meta_strategy)

        # 处理异常主题(-1)
        df['meta_strategy'] = df['meta_strategy'].fillna('MS_Outlier')

        df['sentiment'] = sentiments
        df['arousal'] = arousals

        # 统计每个时间窗口内各元策略的文档数和平均情感
        if 'time_window' in df.columns:
            # 计数矩阵
            count_matrix = df.groupby(['time_window', 'meta_strategy']).size().unstack(fill_value=0)

            # 情感矩阵
            sentiment_agg = df.groupby(['time_window', 'meta_strategy'])['sentiment'].mean()
            sentiment_matrix = sentiment_agg.unstack(fill_value=0)

            # 唤醒度矩阵
            arousal_agg = df.groupby(['time_window', 'meta_strategy'])['arousal'].mean()
            arousal_matrix = arousal_agg.unstack(fill_value=0)

            logger.info(f"  矩阵维度: {count_matrix.shape} (时间窗口 × 元策略)")
            logger.info(f"  包含元策略: {list(count_matrix.columns)}")

            # 计算每个时间窗口的策略流行度(归一化)
            popularity_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

            return count_matrix, sentiment_matrix, arousal_matrix, popularity_matrix
        else:
            logger.warning("未找到时间窗口信息")
            return None, None, None, None

    def process_pipeline(self, df, text_col='全文内容_cleaned'):
        """完整处理流程 (方案C: 动态元策略)"""
        logger.info("=" * 60)
        logger.info("开始阶段二: 认知框架识别 (动态元策略)")
        logger.info("=" * 60)

        # 1. 加载模型
        self.load_embedding_model()
        self.build_topic_model()

        # 2. 准备文本
        texts = df[text_col].fillna('').astype(str).tolist()
        logger.info(f"文本数量: {len(texts)}")

        # 3. 第一层: 细粒度主题建模
        logger.info("\n【第一层】细粒度主题建模...")
        topics, probs = self.fit_topics(texts)

        # 4. 情感量化（提前到这里）
        logger.info("\n【情感分析】量化情感和唤醒度...")
        sentiments, arousals = self.calculate_sentiment(texts)

        # 5. 第二层: 动态识别元策略（现在可以使用sentiments和arousals）
        logger.info("\n【第二层】动态识别元策略...")
        topic_to_meta = self.map_to_meta_strategies(topics, texts, sentiments, arousals)  # 添加参数

        # 6. 第三层: 构建时序观察矩阵
        logger.info("\n【第三层】构建时序观察矩阵...")
        count_matrix, sentiment_matrix, arousal_matrix, popularity_matrix = \
            self.build_observation_matrix(df.copy(), topics, sentiments, arousals)

        # 7. 保存结果
        df_result = df.copy()
        df_result['topic'] = topics
        df_result['topic_probability'] = [probs[i].max() if probs is not None and len(probs.shape) > 1
                                          else 0 for i in range(len(topics))]
        df_result['meta_strategy'] = [topic_to_meta.get(t, 'MS_Outlier') for t in topics]
        df_result['sentiment'] = sentiments
        df_result['arousal'] = arousals

        # 添加元策略的语义名称
        df_result['meta_strategy_name'] = df_result['meta_strategy'].map(
            lambda x: self.meta_strategy_info.get(x, {}).get('semantic_name', '异常')
        )

        output_file = os.path.join(OUTPUT_FOLDER, 'stage2_framework_data.csv')
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n数据已保存: {output_file}")

        # 保存元策略信息
        meta_info_file = os.path.join(OUTPUT_FOLDER, 'stage2_meta_strategy_info.json')
        import json
        with open(meta_info_file, 'w', encoding='utf-8') as f:
            json.dump(safe_convert(self.meta_strategy_info), f, ensure_ascii=False, indent=2)
        logger.info(f"元策略信息已保存: {meta_info_file}")

        # 保存主题到元策略的映射
        mapping_file = os.path.join(OUTPUT_FOLDER, 'stage2_topic_to_meta_mapping.csv')
        mapping_df = pd.DataFrame([
            {
                'topic_id': tid,
                'meta_strategy': ms,
                'meta_strategy_name': self.meta_strategy_info.get(ms, {}).get('semantic_name', '未知')
            }
            for tid, ms in self.topic_to_meta_strategy.items()
        ])
        mapping_df.to_csv(mapping_file, index=False, encoding='utf-8-sig')
        logger.info(f"主题映射已保存: {mapping_file}")

        # 保存矩阵
        if count_matrix is not None:
            count_matrix.to_csv(
                os.path.join(OUTPUT_FOLDER, 'stage2_count_matrix.csv'),
                encoding='utf-8-sig'
            )
            sentiment_matrix.to_csv(
                os.path.join(OUTPUT_FOLDER, 'stage2_sentiment_matrix.csv'),
                encoding='utf-8-sig'
            )
            arousal_matrix.to_csv(
                os.path.join(OUTPUT_FOLDER, 'stage2_arousal_matrix.csv'),
                encoding='utf-8-sig'
            )
            popularity_matrix.to_csv(
                os.path.join(OUTPUT_FOLDER, 'stage2_popularity_matrix.csv'),
                encoding='utf-8-sig'
            )
            logger.info(f"观察矩阵已保存")

        # 可视化元策略演化
        if count_matrix is not None:
            self.visualize_meta_strategy_evolution(count_matrix, popularity_matrix)

        logger.info("\n" + "=" * 60)
        logger.info("阶段二完成!")
        logger.info("=" * 60)

        return df_result, count_matrix, sentiment_matrix, arousal_matrix

    def visualize_meta_strategy_evolution(self, count_matrix, popularity_matrix):
        """可视化元策略在时间上的演化"""
        import matplotlib.pyplot as plt

        import matplotlib.font_manager as fm
        import os
        import platform

        def set_simsunb_font():
            """优先设置simsun（宋体加粗）字体"""
            # 定义simsun字体的可能路径（覆盖更多常见位置）
            simsun_paths = [
                './SIMSUN.TTC',
                './simsun.ttc',
                '../SIMSUN.TTC',
                '../simsun.ttc',
                'weight2/SIMSUN.TTC',
                '/root/weight2/SIMSUN.TTC',
                '/root/weight2/simsun.ttc',
            ]

            # 查找simsun字体
            simsun_font_path = None
            for path in simsun_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    # 验证是否为TrueType字体
                    if path.lower().endswith('.ttc'):
                        simsun_font_path = path
                        break

            # 如果找到simsun字体，直接设置
            if simsun_font_path:
                try:
                    # 注册字体
                    font_prop = fm.FontProperties(fname=simsun_font_path)
                    fm.fontManager.addfont(simsun_font_path)

                    # 设置matplotlib全局字体
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'Arial Unicode MS']
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    plt.rcParams['font.size'] = 10  # 设置默认字体大小

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

            # 先检查系统中已安装的字体
            installed_fonts = {f.name for f in fm.fontManager.ttflist}
            for font_name in common_chinese_fonts:
                if font_name in installed_fonts:
                    logger.info(f"✅ 找到系统中文字体: {font_name}")
                    return font_name

            # 如果没有找到已安装字体，查找字体文件
            fontpaths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            for fontpath in fontpaths:
                font_name = fm.FontProperties(fname=fontpath).get_name()
                if any(keyword in font_name.lower() for keyword in
                       ['sim', 'hei', 'song', 'fang', 'kai', 'pingfang', 'heiti']):
                    logger.info(f"✅ 通过系统查找找到中文字体: {fontpath}")
                    return fontpath

            logger.warning("⚠️ 未找到理想中文字体，尝试使用默认字体")
            return None

        # 调用字体设置
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

        logger.info("生成元策略演化可视化...")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 1. 绝对数量演化
        count_matrix.plot(ax=axes[0], marker='o')
        axes[0].set_title('元策略文档数量演化', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间窗口')
        axes[0].set_ylabel('文档数量')
        axes[0].legend(title='元策略', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 2. 流行度(归一化)演化
        popularity_matrix.plot(ax=axes[1], marker='s', linewidth=2)
        axes[1].set_title('元策略流行度演化 (归一化)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('时间窗口')
        axes[1].set_ylabel('流行度 (占比)')
        axes[1].legend(title='元策略', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_FOLDER, 'stage2_meta_strategy_evolution.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        logger.info("  可视化已保存")


def main():
    """主函数"""
    # 加载阶段一的数据
    input_file = os.path.join(OUTPUT_FOLDER, 'stage1_preprocessed_data.csv')
    logger.info(f"加载数据: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # 运行阶段二
    extractor = CognitiveFrameworkExtractor()
    df_result, count_mat, sent_mat, arousal_mat = extractor.process_pipeline(df)

    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info("阶段二总结")
    logger.info("=" * 60)
    logger.info(f"识别出 {extractor.n_meta_strategies} 个元策略:")

    for meta_id, info in extractor.meta_strategy_info.items():
        strategy_dist = df_result[df_result['meta_strategy'] == meta_id]
        count = len(strategy_dist)
        percentage = count / len(df_result) * 100

        logger.info(f"\n  {meta_id} ({info['semantic_name']}):")
        logger.info(f"    文档数: {count} ({percentage:.1f}%)")
        logger.info(f"    包含主题: {info['member_topics']}")
        logger.info(f"    核心关键词: {', '.join(info['top_keywords'])}")

    # 显示元策略在时间上的分布
    if 'time_window' in df_result.columns:
        logger.info("\n元策略跨时间窗口分布:")
        time_strategy = df_result.groupby(['time_window', 'meta_strategy']).size().unstack(fill_value=0)
        logger.info(f"\n{time_strategy}")


if __name__ == "__main__":
    main()