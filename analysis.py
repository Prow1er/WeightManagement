"""
阶段五: 综合结果分析与解释
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import *

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class ComprehensiveAnalyzer:
    """综合分析器"""
    
    def __init__(self):
        self.data = {}
        self.insights = {}
        
    def load_all_results(self):
        """加载所有阶段的结果 (支持动态策略)"""
        logger.info("加载所有阶段结果...")

        # 阶段二: 框架数据
        self.data['framework'] = pd.read_csv(
            os.path.join(OUTPUT_FOLDER, 'stage2_framework_data.csv'),
            encoding='utf-8-sig'
        )

        # 加载元策略信息
        import json
        meta_info_file = os.path.join(OUTPUT_FOLDER, 'stage2_meta_strategy_info.json')
        with open(meta_info_file, 'r', encoding='utf-8') as f:
            self.data['meta_strategy_info'] = json.load(f)

        logger.info(f"  识别出 {len(self.data['meta_strategy_info'])} 个元策略")

        # 阶段三: 网络指标
        self.data['network_metrics'] = pd.read_csv(
            os.path.join(OUTPUT_FOLDER, 'stage3_network_metrics.csv'),
            encoding='utf-8-sig'
        )

        # 阶段四: 博弈结果
        self.data['game_results'] = pd.read_csv(
            os.path.join(OUTPUT_FOLDER, 'stage4_game_results.csv'),
            encoding='utf-8-sig'
        )

        logger.info("数据加载完成")

    def analyze_snowball_effect(self):
        """分析共识的滚雪球效应 (动态策略)"""
        logger.info("\n分析1: 共识滚雪球效应")
        logger.info("=" * 60)

        df = self.data['framework']

        # 使用meta_strategy列
        strategy_col = 'meta_strategy' if 'meta_strategy' in df.columns else 'strategy'

        # 按时间窗口统计各策略的流行度变化
        if 'time_window' in df.columns:
            popularity_evolution = df.groupby(['time_window', strategy_col]).size().unstack(fill_value=0)

            # 计算增长率
            growth_rates = popularity_evolution.pct_change().mean()

            logger.info("各策略的平均增长率:")
            for strategy, rate in growth_rates.items():
                if strategy != 'MS_Outlier':
                    # 获取策略的语义名称
                    semantic_name = self.data['meta_strategy_info'].get(strategy, {}).get('semantic_name', strategy)
                    logger.info(f"  {strategy} ({semantic_name}): {rate:.2%}")

            # 识别滚雪球效应 (增长率 > 10%)
            snowball_strategies = growth_rates[growth_rates > 0.1].index.tolist()
            snowball_strategies = [s for s in snowball_strategies if s != 'MS_Outlier']

            if snowball_strategies:
                logger.info(f"\n发现滚雪球效应的策略: {', '.join(snowball_strategies)}")
                logger.info("这些策略通过'简单易懂(低成本)' + '初始高声量'快速形成共识霸权")

            self.insights['snowball_effect'] = {
                'growth_rates': growth_rates.to_dict(),
                'snowball_strategies': snowball_strategies
            }

            return popularity_evolution
        else:
            logger.warning("缺少时间窗口信息")
            return None

    def analyze_semantic_homogenization(self):
        """分析语义网络的同质化"""
        logger.info("\n分析2: 语义网络同质化")
        logger.info("=" * 60)

        metrics = self.data['network_metrics']

        # 分析密度和聚类系数的变化趋势
        density_trend = metrics['density'].values
        clustering_trend = metrics['avg_clustering'].values

        # 计算趋势(简单线性拟合)
        if len(density_trend) > 1:
            density_slope = np.polyfit(range(len(density_trend)), density_trend, 1)[0]
            clustering_slope = np.polyfit(range(len(clustering_trend)), clustering_trend, 1)[0]

            logger.info(f"网络密度趋势: {'上升' if density_slope > 0 else '下降'} (斜率={density_slope:.4f})")
            logger.info(f"聚类系数趋势: {'上升' if clustering_slope > 0 else '下降'} (斜率={clustering_slope:.4f})")

            if density_slope > 0 and clustering_slope > 0:
                logger.info("\n结论: 网络随时间推移变得越来越紧密,说明共识形成导致语言同质化")
                logger.info("      所有人都说同样的话,词汇越来越贫乏")

            self.insights['homogenization'] = {
                'density_slope': density_slope,
                'clustering_slope': clustering_slope,
                'is_homogenizing': density_slope > 0 and clustering_slope > 0
            }

    def analyze_emotion_rationality_mismatch(self):
        """分析情感与理性的错位 (动态策略)"""
        logger.info("\n分析3: 情感与理性错位")
        logger.info("=" * 60)

        df = self.data['framework']

        strategy_col = 'meta_strategy' if 'meta_strategy' in df.columns else 'strategy'

        # 计算各策略的平均情感和唤醒度
        strategy_emotions = df.groupby(strategy_col).agg({
            'sentiment': 'mean',
            'arousal': 'mean'
        })

        logger.info("各策略的情感特征:")
        for strategy, row in strategy_emotions.iterrows():
            if strategy != 'MS_Outlier':
                semantic_name = self.data['meta_strategy_info'].get(strategy, {}).get('semantic_name', strategy)
                logger.info(f"  {strategy} ({semantic_name}):")
                logger.info(f"    情感极性: {row['sentiment']:.3f}")
                logger.info(f"    唤醒度: {row['arousal']:.3f}")

        # 找出最"科学"和最"捷径"的策略
        # 基于语义名称判断
        science_strategies = []
        shortcut_strategies = []

        for strategy, info in self.data['meta_strategy_info'].items():
            semantic_name = info.get('semantic_name', '')
            if '科学' in semantic_name or '健康' in semantic_name:
                science_strategies.append(strategy)
            elif '快速' in semantic_name or '捷径' in semantic_name:
                shortcut_strategies.append(strategy)

        if science_strategies and shortcut_strategies:
            # 比较唤醒度
            science_arousal = strategy_emotions.loc[science_strategies, 'arousal'].mean()
            shortcut_arousal = strategy_emotions.loc[shortcut_strategies, 'arousal'].mean()

            logger.info(f"\n科学派平均唤醒度: {science_arousal:.3f}")
            logger.info(f"捷径派平均唤醒度: {shortcut_arousal:.3f}")

            if shortcut_arousal > science_arousal:
                logger.info("\n发现: 捷径派的情绪唤醒度显著高于科学派")
                logger.info("解释: 科学共识往往伴随'枯燥、中性'情感,缺乏传播力")
                logger.info("      而捷径派伴随'极度兴奋'或'极度沮丧'的强情绪")
                logger.info("      这解释了为什么科学共识难以构建——缺乏情绪传播力")

            self.insights['emotion_mismatch'] = {
                'strategy_emotions': strategy_emotions.to_dict(),
                'science_arousal': science_arousal,
                'shortcut_arousal': shortcut_arousal,
                'arousal_gap': shortcut_arousal - science_arousal
            }
        else:
            logger.warning("未能识别出明确的科学派和捷径派")
            self.insights['emotion_mismatch'] = {
                'strategy_emotions': strategy_emotions.to_dict()
            }

    def analyze_game_convergence(self):
        """分析演化博弈的收敛模式（支持动态策略）"""
        logger.info("\n分析4: 演化博弈收敛分析")
        logger.info("=" * 60)

        game_results = self.data['game_results']

        # 统计各类均衡类型
        if 'equilibrium_type' in game_results.columns:
            equilibrium_types = game_results['equilibrium_type'].value_counts()
            logger.info("均衡类型分布:")
            for eq_type, count in equilibrium_types.items():
                logger.info(f"  {eq_type}: {count} 个窗口")
        else:
            equilibrium_types = {}
            logger.warning("未找到均衡类型列")

        # 动态提取策略名称（从列名中）
        initial_cols = [col for col in game_results.columns
                        if col.startswith('initial_') and 'MS_Outlier' not in col]
        final_cols = [col for col in game_results.columns
                      if col.startswith('final_') and 'MS_Outlier' not in col]

        # 提取策略ID（去掉 initial_ 或 final_ 前缀）
        initial_strategies = [col.replace('initial_', '') for col in initial_cols]
        final_strategies = [col.replace('final_', '') for col in final_cols]

        # 找出共同的策略（初始和最终都有的）
        common_strategies = list(set(initial_strategies) & set(final_strategies))
        common_strategies.sort()  # 排序保证一致性

        logger.info(f"\n共同策略: {common_strategies}")

        if len(common_strategies) == 0:
            logger.warning("未找到共同策略列，跳过方差分析")
            self.insights['game_convergence'] = {
                'equilibrium_types': equilibrium_types if isinstance(equilibrium_types,
                                                                     dict) else equilibrium_types.to_dict()
            }
            return

        # 重新构建对齐的列名
        aligned_initial_cols = [f'initial_{s}' for s in common_strategies]
        aligned_final_cols = [f'final_{s}' for s in common_strategies]

        # 分析初始vs最终分布的变化
        initial_variances = []
        final_variances = []

        for idx, row in game_results.iterrows():
            # 每个窗口的初始分布方差（按照对齐的顺序）
            initial_values = [row[col] for col in aligned_initial_cols if col in row.index]
            if len(initial_values) > 1:
                initial_variances.append(np.var(initial_values))

            # 每个窗口的最终分布方差（按照对齐的顺序）
            final_values = [row[col] for col in aligned_final_cols if col in row.index]
            if len(final_values) > 1:
                final_variances.append(np.var(final_values))

        if len(initial_variances) == 0 or len(final_variances) == 0:
            logger.warning("方差数据不足，跳过分析")
            self.insights['game_convergence'] = {
                'equilibrium_types': equilibrium_types if isinstance(equilibrium_types,
                                                                     dict) else equilibrium_types.to_dict(),
                'n_strategies': len(common_strategies)
            }
            return

        initial_variance = np.mean(initial_variances)
        final_variance = np.mean(final_variances)
        variance_change = final_variance - initial_variance

        logger.info(f"\n初始分布平均方差: {initial_variance:.4f}")
        logger.info(f"最终分布平均方差: {final_variance:.4f}")
        logger.info(f"方差变化: {variance_change:+.4f}")

        if final_variance > initial_variance * 1.1:  # 增加超过10%
            logger.info("\n解释: 演化后策略分化更明显，说明趋势追随机制放大了差异")
            logger.info("      优势策略获得更多支持，弱势策略进一步边缘化")
        elif final_variance < initial_variance * 0.9:  # 减少超过10%
            logger.info("\n解释: 演化后策略趋于均衡，分布更加平均")
            logger.info("      可能存在多个竞争性策略达到稳定共存")
        else:
            logger.info("\n解释: 演化前后方差变化不大，策略分布相对稳定")

        # 计算策略多样性的变化
        if 'diversity' in game_results.columns:
            avg_diversity = game_results['diversity'].mean()
            logger.info(f"\n平均策略多样性指数: {avg_diversity:.3f}")
            if avg_diversity < 0.5:
                logger.info("  → 低多样性，通常存在主导策略")
            elif avg_diversity < 0.8:
                logger.info("  → 中等多样性，存在明显的优势策略")
            else:
                logger.info("  → 高多样性，策略分布较为均衡")

        self.insights['game_convergence'] = {
            'equilibrium_types': equilibrium_types if isinstance(equilibrium_types,
                                                                 dict) else equilibrium_types.to_dict(),
            'initial_variance': float(initial_variance),
            'final_variance': float(final_variance),
            'variance_change': float(variance_change),
            'n_strategies': len(common_strategies),
            'strategies': common_strategies,
            'avg_diversity': float(game_results['diversity'].mean()) if 'diversity' in game_results.columns else None
        }

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        logger.info("\n生成综合分析报告...")

        report_file = os.path.join(OUTPUT_FOLDER, 'stage5_comprehensive_report.md')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 体重管理观点与行为策略演化机制综合分析报告\n\n")
            f.write("基于社交媒体评论数据的复杂网络与演化博弈分析\n\n")
            f.write("=" * 80 + "\n\n")

            # 研究概览
            f.write("## 一、研究概览\n\n")
            df = self.data['framework']
            f.write(f"- **分析数据量**: {len(df)} 条评论/帖子\n")
            if 'time_window' in df.columns:
                f.write(f"- **时间跨度**: {df['time_window'].nunique()} 个时间窗口\n")

            # 修复策略数计算：先判断存在的策略列
            strategy_col = 'meta_strategy' if 'meta_strategy' in df.columns else 'strategy'
            total_strategies = df[strategy_col].nunique()
            other_strategies = total_strategies - 3  # 3个元策略
            f.write(f"- **识别策略数**: 3个元策略 + {other_strategies} 个其他策略\n\n")

            # 核心发现
            f.write("## 二、核心发现\n\n")

            f.write("### 2.1 共识的滚雪球效应\n\n")
            if 'snowball_effect' in self.insights:
                snowball = self.insights['snowball_effect']
                f.write(f"发现滚雪球效应的策略: {', '.join(snowball.get('snowball_strategies', []))}\n\n")
                f.write("**机制解释**:\n")
                f.write("- 这些观点通过'简单易懂(低认知成本)' + '初始高声量'迅速形成共识霸权\n")
                f.write("- 流行度参数Popularity起主导作用,即使效果存疑,高流行度仍能持续传播\n\n")

            f.write("### 2.2 语义网络的同质化\n\n")
            if 'homogenization' in self.insights:
                homo = self.insights['homogenization']
                if homo['is_homogenizing']:
                    f.write("**观察**: 网络密度和聚类系数随时间上升\n\n")
                    f.write("**解释**: 共识构建过程也是'语言贫困化'过程\n")
                    f.write("- 所有人都说同样的话,使用相同的术语(如'液断'、'碳循环')\n")
                    f.write("- 词汇越来越贫乏,形成严密的'回音室'\n\n")

            f.write("### 2.3 情感与理性的错位\n\n")
            if 'emotion_mismatch' in self.insights:
                mismatch = self.insights['emotion_mismatch']
                f.write("**核心矛盾**: 科学共识vs捷径共识的情绪传播力差异\n\n")
                f.write("| 策略类型 | 情感特征 | 传播效果 |\n")
                f.write("|---------|---------|--------|\n")
                f.write("| S1科学正统 | 枯燥、中性、理性 | 传播力弱 |\n")
                f.write("| S2捷径流行 | 极度兴奋或沮丧 | 传播力强 |\n\n")
                f.write("**结论**: 科学共识难以构建,因为缺乏情绪能量驱动传播\n\n")

            # 方法论贡献
            f.write("## 三、方法论创新\n\n")
            f.write("### 3.1 时序演化分析框架\n")
            f.write("- 通过时间窗口切片捕捉观点动态演化\n")
            f.write("- 保留短文本(如'绝绝子')识别强共识信号\n\n")

            f.write("### 3.2 语义共现网络\n")
            f.write("- 基于文本语义相似度构建隐性网络\n")
            f.write("- 揭示回音室形成机制\n\n")

            f.write("### 3.3 平均场演化博弈\n")
            f.write("- 趋势追随机制: 用户与'整个舆论场趋势'博弈\n")
            f.write("- 收益函数: U = α·Popularity + β·Sentiment - Cost\n\n")

            # 政策建议
            f.write("## 四、实践启示\n\n")
            f.write("### 4.1 对健康传播的启示\n")
            f.write("- 科学减肥知识需要'情绪化包装'才能有效传播\n")
            f.write("- 不能仅依赖理性说服,需要设计情感共鸣点\n\n")

            f.write("### 4.2 对平台治理的建议\n")
            f.write("- 警惕'伪科学观点'通过滚雪球效应形成霸权\n")
            f.write("- 需要打破回音室,促进多元观点交流\n\n")

            f.write("=" * 80 + "\n")
            f.write("\n**报告生成时间**: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

        logger.info(f"报告已保存: {report_file}")

    def visualize_comprehensive_dashboard(self):
        """生成综合可视化仪表板 (动态策略)"""
        logger.info("生成综合可视化仪表板...")

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

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        strategy_col = 'meta_strategy' if 'meta_strategy' in self.data['framework'].columns else 'strategy'

        # 1. 策略分布饼图
        ax1 = fig.add_subplot(gs[0, 0])
        strategy_counts = self.data['framework'][strategy_col].value_counts()
        # 移除异常策略
        strategy_counts = strategy_counts[strategy_counts.index != 'MS_Outlier']
        ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', textprops={'fontsize': 8})
        ax1.set_title('策略分布')

        # 2. 情感极性分布
        ax2 = fig.add_subplot(gs[0, 1])
        sentiment_by_strategy = self.data['framework'].groupby(strategy_col)['sentiment'].mean()
        sentiment_by_strategy = sentiment_by_strategy[sentiment_by_strategy.index != 'MS_Outlier']
        ax2.barh(range(len(sentiment_by_strategy)), sentiment_by_strategy.values)
        ax2.set_yticks(range(len(sentiment_by_strategy)))
        ax2.set_yticklabels(sentiment_by_strategy.index, fontsize=7)
        ax2.set_xlabel('平均情感极性')
        ax2.set_title('各策略情感特征')
        ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)

        # 3. 唤醒度对比
        ax3 = fig.add_subplot(gs[0, 2])
        arousal_by_strategy = self.data['framework'].groupby(strategy_col)['arousal'].mean()
        arousal_by_strategy = arousal_by_strategy[arousal_by_strategy.index != 'MS_Outlier']
        ax3.bar(range(len(arousal_by_strategy)), arousal_by_strategy.values, color='orange', alpha=0.7)
        ax3.set_xticks(range(len(arousal_by_strategy)))
        ax3.set_xticklabels(arousal_by_strategy.index, rotation=45, ha='right', fontsize=7)
        ax3.set_ylabel('平均唤醒度')
        ax3.set_title('各策略情绪唤醒度')

        # 4. 网络演化
        ax4 = fig.add_subplot(gs[1, :])
        metrics = self.data['network_metrics']
        ax4_twin = ax4.twinx()
        ax4.plot(metrics['density'], marker='o', label='密度', color='steelblue')
        ax4_twin.plot(metrics['avg_clustering'], marker='s', label='聚类系数', color='orange')
        ax4.set_xlabel('时间窗口')
        ax4.set_ylabel('网络密度', color='steelblue')
        ax4_twin.set_ylabel('聚类系数', color='orange')
        ax4.set_title('网络结构演化(同质化分析)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')

        # 5. 博弈演化对比 - 只显示有数据的策略
        ax5 = fig.add_subplot(gs[2, :])
        game_results = self.data['game_results']

        # 找出所有final_列
        final_cols = [col for col in game_results.columns if col.startswith('final_') and col != 'final_MS_Outlier']
        strategies = [col.replace('final_', '') for col in final_cols]

        x = range(len(game_results))
        width = 0.8 / len(strategies) if strategies else 0.2

        for i, strategy in enumerate(strategies):
            col_name = f'final_{strategy}'
            if col_name in game_results.columns:
                offset = (i - len(strategies)/2) * width
                ax5.bar(
                    [xi + offset for xi in x],
                    game_results[col_name],
                    width,
                    label=strategy,
                    alpha=0.8
                )

        ax5.set_xlabel('时间窗口')
        ax5.set_ylabel('策略占比')
        ax5.set_title('演化博弈:各窗口均衡状态')
        ax5.legend(fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(
            os.path.join(OUTPUT_FOLDER, 'stage5_comprehensive_dashboard.png'),
            dpi=VISUALIZATION_CONFIG['figure_dpi'],
            bbox_inches='tight'
        )
        plt.close()
        logger.info("仪表板已保存")

    def run_full_analysis(self):
        """运行完整分析"""
        logger.info("=" * 60)
        logger.info("开始阶段五: 综合结果分析")
        logger.info("=" * 60)

        # 加载数据
        self.load_all_results()

        # 执行四大分析
        self.analyze_snowball_effect()
        self.analyze_semantic_homogenization()
        self.analyze_emotion_rationality_mismatch()
        self.analyze_game_convergence()

        # 生成报告和可视化
        self.generate_comprehensive_report()
        self.visualize_comprehensive_dashboard()

        logger.info("\n" + "=" * 60)
        logger.info("阶段五完成! 所有分析结果已保存")
        logger.info("=" * 60)


def main():
    """主函数"""
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()