"""
阶段一: 时序数据切片与预处理
"""
import pandas as pd
import numpy as np
import re
import jieba
from datetime import datetime
from collections import Counter
import logging
from config import *

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class TemporalDataProcessor:
    """时序数据处理器"""
    
    def __init__(self, config=TIME_WINDOW_CONFIG):
        self.config = config
        self.windows = []
        self.window_data = {}
        
    def load_data(self, file_path):
        """加载数据"""
        logger.info(f"加载数据: {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        logger.info(f"原始数据量: {len(df)} 条")
        return df
    
    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 去除URL
        text = re.sub(r'http[s]?://\S+', '', text)
        # 去除@用户
        text = re.sub(r'@\S+', '', text)
        # 去除话题标签
        text = re.sub(r'#\S+#', '', text)
        # 去除表情符号
        text = re.sub(r'\[.*?\]', '', text)
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,。!?;:""''\s]', '', text)
        
        return text.strip()
    
    def filter_short_texts(self, df, text_col, min_length=5):
        """保留短文本 - 在共识研究中短语携带强共识信号"""
        original_len = len(df)
        # 计算文本长度
        df['text_length'] = df[text_col].fillna('').astype(str).apply(len)
        # 只过滤完全空的文本
        df = df[df['text_length'] >= min_length].copy()
        logger.info(f"过滤空文本: {original_len} -> {len(df)} 条")
        return df.drop('text_length', axis=1)
    
    def detect_bots(self, df, user_col='用户ID'):
        """简单的机器人检测"""
        if user_col not in df.columns:
            logger.warning(f"未找到用户列 {user_col}, 跳过机器人检测")
            return df
        
        # 统计每个用户的发帖数和平均时间间隔
        user_stats = df.groupby(user_col).agg({
            user_col: 'count'
        }).rename(columns={user_col: 'post_count'})
        
        # 标记发帖异常频繁的用户 (简单规则)
        suspected_bots = user_stats[user_stats['post_count'] > 100].index
        
        original_len = len(df)
        df = df[~df[user_col].isin(suspected_bots)].copy()
        logger.info(f"过滤疑似机器人: {original_len} -> {len(df)} 条")
        
        return df
    
    def create_temporal_windows(self, df, time_col='日期'):
        """创建时间窗口"""
        logger.info("创建时间窗口...")
        
        # 确保时间列是datetime格式
        if time_col not in df.columns:
            logger.error(f"未找到时间列: {time_col}")
            # 尝试查找其他可能的时间列
            possible_cols = ['发布时间', 'time', 'date', '时间']
            for col in possible_cols:
                if col in df.columns:
                    time_col = col
                    logger.info(f"使用替代时间列: {time_col}")
                    break
            else:
                logger.warning("未找到时间列,将所有数据视为单一时间窗口")
                df['time_window'] = 0
                self.windows = [0]
                self.window_data[0] = df
                return df
        
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # 删除无效时间
        df = df.dropna(subset=[time_col])
        
        # 根据配置创建时间窗口
        unit = self.config['unit']
        if unit == 'M':
            df['time_window'] = df[time_col].dt.to_period('M')
        elif unit == 'W':
            df['time_window'] = df[time_col].dt.to_period('W')
        elif unit == 'D':
            df['time_window'] = df[time_col].dt.to_period('D')
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
        
        # 转换为字符串便于处理
        df['time_window'] = df['time_window'].astype(str)
        
        # 统计每个窗口的文档数
        window_counts = df['time_window'].value_counts().sort_index()
        
        # 过滤文档数过少的窗口
        min_docs = self.config['min_docs_per_window']
        valid_windows = window_counts[window_counts >= min_docs].index.tolist()
        
        logger.info(f"总时间窗口: {len(window_counts)}")
        logger.info(f"有效窗口 (>={min_docs}条): {len(valid_windows)}")
        
        # 只保留有效窗口的数据
        df = df[df['time_window'].isin(valid_windows)].copy()
        
        # 存储窗口信息
        self.windows = sorted(valid_windows)
        for window in self.windows:
            self.window_data[window] = df[df['time_window'] == window].copy()
            logger.info(f"  窗口 {window}: {len(self.window_data[window])} 条")
        
        return df
    
    def extract_jargon(self, df, text_col, top_n=50):
        """提取高频术语(Jargon)"""
        logger.info("提取高频术语...")
        
        # 加载自定义词典
        custom_dict = """
        体重管理 5
        减肥 5
        减脂 5
        瘦身 5
        BMI 3
        体脂率 3
        平台期 3
        热量缺口 3
        基础代谢 3
        轻断食 3
        生酮 3
        碳水循环 3
        """.strip().split('\n')
        
        for word in custom_dict:
            if word.strip():
                jieba.add_word(word.strip().split()[0])
        
        # 分词并统计
        all_words = []
        for text in df[text_col].fillna(''):
            words = jieba.lcut(str(text))
            # 过滤停用词和短词
            words = [w for w in words if len(w) >= 2 and w not in STOP_WORDS]
            all_words.extend(words)
        
        # 统计词频
        word_freq = Counter(all_words)
        top_jargon = word_freq.most_common(top_n)
        
        logger.info(f"提取到 {len(top_jargon)} 个高频术语")
        logger.info(f"示例: {', '.join([w for w, _ in top_jargon[:10]])}")
        
        return top_jargon
    
    def process_pipeline(self, df, text_col='全文内容'):
        """完整预处理流程"""
        logger.info("=" * 60)
        logger.info("开始阶段一: 数据预处理")
        logger.info("=" * 60)
        
        # 1. 基础清洗
        logger.info("\n步骤1: 文本清洗")
        df[f'{text_col}_cleaned'] = df[text_col].apply(self.clean_text)
        
        # 2. 过滤短文本 (只过滤空文本)
        logger.info("\n步骤2: 过滤空文本")
        df = self.filter_short_texts(df, f'{text_col}_cleaned', min_length=5)
        
        # 3. 机器人检测 (可选)
        logger.info("\n步骤3: 机器人检测")
        if 'user_id' in df.columns or '用户ID' in df.columns:
            df = self.detect_bots(df)
        
        # 4. 时间窗口划分
        logger.info("\n步骤4: 时间窗口划分")
        df = self.create_temporal_windows(df)
        
        # 5. 提取共识符号
        logger.info("\n步骤5: 提取共识符号")
        jargon = self.extract_jargon(df, f'{text_col}_cleaned')
        
        # 保存结果
        output_file = os.path.join(OUTPUT_FOLDER, 'stage1_preprocessed_data.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n阶段一完成! 数据已保存: {output_file}")
        
        # 保存高频术语
        jargon_file = os.path.join(OUTPUT_FOLDER, 'stage1_jargon.txt')
        with open(jargon_file, 'w', encoding='utf-8') as f:
            for word, freq in jargon:
                f.write(f"{word}\t{freq}\n")
        logger.info(f"高频术语已保存: {jargon_file}")
        
        return df, jargon


def main():
    """主函数"""
    processor = TemporalDataProcessor()
    
    # 加载数据
    df = processor.load_data(INPUT_FILE)
    
    # 确定文本列
    text_col = None
    for col in ['全文内容', '标题_微博内容', '原微博内容', '内容']:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        raise ValueError("未找到合适的文本列")
    
    logger.info(f"使用文本列: {text_col}")
    
    # 运行预处理
    processed_df, jargon = processor.process_pipeline(df, text_col)
    
    logger.info("\n" + "=" * 60)
    logger.info("阶段一总结")
    logger.info("=" * 60)
    logger.info(f"处理后数据量: {len(processed_df)} 条")
    logger.info(f"时间窗口数: {len(processor.windows)}")
    logger.info(f"高频术语数: {len(jargon)}")


if __name__ == "__main__":
    main()
