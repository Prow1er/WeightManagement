"""
主流程: 体重管理观点演化分析完整Pipeline

export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT  # 输出设置的镜像地址即生效
"""
import argparse
import logging
import time
from datetime import datetime
from config import *
import pandas as pd
import os

# 导入各阶段模块
from preprocessing import TemporalDataProcessor
from framework import CognitiveFrameworkExtractor
from network import SemanticNetworkBuilder
from game import MultiWindowGameSimulator
from analysis import ComprehensiveAnalyzer

# 配置日志
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class PipelineManager:
    """Pipeline管理器"""
    
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        
    def print_banner(self):
        """打印启动横幅"""
        banner = """
        ╔════════════════════════════════════════════════════════════╗
        ║   体重管理观点与行为策略的演化机制研究                     ║
        ║   基于社交媒体评论数据的复杂网络与演化博弈分析             ║
        ╚════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info(f"Pipeline启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_stage1(self):
        """运行阶段一: 数据预处理"""
        logger.info("\n" + "🔄 " * 30)
        logger.info("STAGE 1: 时序演化的数据增强与预处理")
        logger.info("🔄 " * 30)
        
        stage_start = time.time()
        
        processor = TemporalDataProcessor()
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
        
        self.stage_times['stage1'] = time.time() - stage_start
        logger.info(f"✅ 阶段一完成,耗时: {self.stage_times['stage1']:.2f}秒")
        
        return processed_df
    
    def run_stage2(self):
        """运行阶段二: 认知框架识别"""
        logger.info("\n" + "🔄 " * 30)
        logger.info("STAGE 2: 认知框架识别与情感量化")
        logger.info("🔄 " * 30)
        
        stage_start = time.time()
        
        # 加载阶段一的数据
        input_file = os.path.join(OUTPUT_FOLDER, 'stage1_preprocessed_data.csv')
        logger.info(f"加载数据: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # 运行阶段二
        extractor = CognitiveFrameworkExtractor()
        df_result, count_mat, sent_mat, arousal_mat = extractor.process_pipeline(df)
        
        self.stage_times['stage2'] = time.time() - stage_start
        logger.info(f"✅ 阶段二完成,耗时: {self.stage_times['stage2']:.2f}秒")
        
        return df_result
    
    def run_stage3(self):
        """运行阶段三: 语义网络构建"""
        logger.info("\n" + "🔄 " * 30)
        logger.info("STAGE 3: 基于语义共现的隐性网络构建")
        logger.info("🔄 " * 30)
        
        stage_start = time.time()
        
        # 加载阶段二的数据
        input_file = os.path.join(OUTPUT_FOLDER, 'stage2_framework_data.csv')
        logger.info(f"加载数据: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # 运行阶段三
        builder = SemanticNetworkBuilder(
            similarity_threshold=NETWORK_CONFIG['similarity_threshold']
        )
        networks, metrics = builder.process_pipeline(df)
        
        self.stage_times['stage3'] = time.time() - stage_start
        logger.info(f"✅ 阶段三完成,耗时: {self.stage_times['stage3']:.2f}秒")
        
        return networks
    
    def run_stage4(self):
        """运行阶段四: 演化博弈模拟"""
        logger.info("\n" + "🔄 " * 30)
        logger.info("STAGE 4: 基于平均场的演化博弈")
        logger.info("🔄 " * 30)
        
        stage_start = time.time()
        
        # 加载阶段二的矩阵数据
        count_matrix_file = os.path.join(OUTPUT_FOLDER, 'stage2_count_matrix.csv')
        sentiment_matrix_file = os.path.join(OUTPUT_FOLDER, 'stage2_sentiment_matrix.csv')
        
        logger.info(f"加载矩阵数据...")
        count_matrix = pd.read_csv(count_matrix_file, index_col=0, encoding='utf-8-sig')
        sentiment_matrix = pd.read_csv(sentiment_matrix_file, index_col=0, encoding='utf-8-sig')
        
        # 运行多窗口模拟
        simulator = MultiWindowGameSimulator()
        simulator.simulate_all_windows(count_matrix, sentiment_matrix)
        simulator.save_results()
        
        self.stage_times['stage4'] = time.time() - stage_start
        logger.info(f"✅ 阶段四完成,耗时: {self.stage_times['stage4']:.2f}秒")
        
        return simulator
    
    def run_stage5(self):
        """运行阶段五: 综合分析"""
        logger.info("\n" + "🔄 " * 30)
        logger.info("STAGE 5: 结果分析与解释")
        logger.info("🔄 " * 30)
        
        stage_start = time.time()
        
        # 运行综合分析
        analyzer = ComprehensiveAnalyzer()
        analyzer.run_full_analysis()
        
        self.stage_times['stage5'] = time.time() - stage_start
        logger.info(f"✅ 阶段五完成,耗时: {self.stage_times['stage5']:.2f}秒")
        
        return analyzer
    
    def print_summary(self):
        """打印总结信息"""
        total_time = sum(self.stage_times.values())
        
        summary = f"""
        ╔════════════════════════════════════════════════════════════╗
        ║                    Pipeline执行总结                         ║
        ╚════════════════════════════════════════════════════════════╝
        
        各阶段耗时:
        """
        
        for stage, duration in self.stage_times.items():
            summary += f"\n          {stage}: {duration:.2f}秒 ({duration/total_time*100:.1f}%)"
        
        summary += f"""
        
        总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)
        
        输出文件位置: {OUTPUT_FOLDER}
        
        主要输出文件:
          - stage1_preprocessed_data.csv       (预处理后数据)
          - stage2_framework_data.csv          (认知框架数据)
          - stage3_network_metrics.csv         (网络指标)
          - stage4_game_results.csv            (博弈模拟结果)
          - stage5_comprehensive_report.md     (综合分析报告)
          - stage5_comprehensive_dashboard.png (可视化仪表板)
        
        ╔════════════════════════════════════════════════════════════╗
        ║                 🎉 Pipeline执行完成! 🎉                    ║
        ╚════════════════════════════════════════════════════════════╝
        """
        
        print(summary)
        logger.info("Pipeline完成")
    
    def run_pipeline(self, stages=None, skip_stages=None):
        """
        运行Pipeline
        
        参数:
            stages: 要运行的阶段列表,如 [1, 2, 3]
            skip_stages: 要跳过的阶段列表
        """
        self.start_time = time.time()
        self.print_banner()
        
        # 默认运行所有阶段
        if stages is None:
            stages = [1, 2, 3, 4, 5]
        
        # 处理跳过阶段
        if skip_stages:
            stages = [s for s in stages if s not in skip_stages]
        
        logger.info(f"将要运行的阶段: {stages}")
        
        try:
            # 按顺序执行各阶段
            if 1 in stages:
                self.run_stage1()
            
            if 2 in stages:
                self.run_stage2()
            
            if 3 in stages:
                self.run_stage3()
            
            if 4 in stages:
                self.run_stage4()
            
            if 5 in stages:
                self.run_stage5()
            
            # 打印总结
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline执行出错: {str(e)}", exc_info=True)
            raise


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='体重管理观点演化分析Pipeline'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有阶段(默认)'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        help='指定要运行的阶段,如: 1 或 1,2,3'
    )
    
    parser.add_argument(
        '--skip-stage',
        type=str,
        help='指定要跳过的阶段,如: 1 或 1,2'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 解析要运行的阶段
    stages = None
    skip_stages = None
    
    if args.stage:
        stages = [int(s.strip()) for s in args.stage.split(',')]
    
    if args.skip_stage:
        skip_stages = [int(s.strip()) for s in args.skip_stage.split(',')]
    
    # 创建并运行Pipeline
    pipeline = PipelineManager()
    pipeline.run_pipeline(stages=stages, skip_stages=skip_stages)


if __name__ == "__main__":
    main()
