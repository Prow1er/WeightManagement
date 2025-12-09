"""
配置文件 - 全局参数设置
"""
import os

# ========== 路径配置 ==========
DATA_FOLDER = "../autodl-fs/filtered_data"
OUTPUT_FOLDER = "./evolution_results"
HF_CACHE = './hf_cache'

# 创建必要目录
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# ========== 数据文件 ==========
INPUT_FILE = os.path.join(DATA_FOLDER, "all_weight_related_combined.csv")

# ========== 时间窗口配置 ==========
TIME_WINDOW_CONFIG = {
    'unit': 'M',  # 'D': 天, 'W': 周, 'M': 月
    'window_size': 1,  # 窗口大小
    'min_docs_per_window': 100,  # 每个窗口最少文档数
}

# ========== NLP模型配置 ==========
NLP_CONFIG = {
    'embedding_model': 'BAAI/bge-small-zh-v1.5',
    'max_seq_length': 256,
    'batch_size': 128,
    'device': 'cuda',  # 'cuda' or 'cpu'
}

# ========== BERTopic配置 ==========
BERTOPIC_CONFIG = {
    'min_topic_size': 200,  # 降低阈值,允许更细粒度的主题
    'nr_topics': 15,  # 指定主题数量,避免过度合并
    'top_n_words': 15,  # 增加关键词数量以更好区分
    'calculate_probabilities': True,
    'diversity': 0.3,  # 添加多样性参数,增加主题区分度
}

# ========== 策略定义 (已废弃 - 改为动态识别) ==========
# 注意: 这些关键词现在只用于辅助语义推断,不再预设策略
STRATEGY_HINT_KEYWORDS = {
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

# ========== 情感分析配置 ==========
SENTIMENT_CONFIG = {
    'model_name': 'uer/roberta-base-finetuned-jd-binary-chinese',  # 或其他中文情感模型
    'batch_size': 64,
}

# ========== 网络构建配置 ==========
NETWORK_CONFIG = {
    'similarity_threshold': 0.95,  # 余弦相似度阈值
    'min_edges': 0,  # 节点最少连边数
    'use_sliding_window': True,  # 是否使用滑动时间窗口
}

# ========== 演化博弈配置 ==========
GAME_CONFIG = {
    'alpha': 0.45,  # 流行度权重
    'beta': 0.45,   # 情感权重
    'cost': {      # 认知成本
        'S1_科学正统': 0.80,
        'S2_捷径流行': 0.05,
        'S3_躺平反抗': 0.15,
    },
    'dt': 0.01,    # 时间步长
    'max_iterations': 1000,  # 最大迭代次数
    'convergence_threshold': 1e-4,  # 收敛阈值
}

# ========== 可视化配置 ==========
VISUALIZATION_CONFIG = {
    'figure_dpi': 600,
    'figure_format': 'png',
    'font_size': 15,
    'color_scheme': 'Set3',
}

# ========== 停用词 ==========
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这', '但', '这个',
    '可以', '这样', '啊', '吗', '呢', '吧', '哈哈'
])

# ========== 日志配置 ==========
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(OUTPUT_FOLDER, 'pipeline.log'),
    'filemode': 'a'
}