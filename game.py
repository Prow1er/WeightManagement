"""
阶段四: 基于平均场的演化博弈
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from config import *

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class EvolutionaryGameModel:
    """演化博弈模型 (支持动态策略数)"""

    def __init__(self, costs, alpha=0.7, beta=0.2, default_cost=0.3):
        """
        参数:
            alpha: 流行度权重
            beta: 情感权重
            default_cost: 默认认知成本 (对于未知策略)
        """
        self.alpha = alpha
        self.beta = beta
        self.default_cost = default_cost

        self.strategies = []  # 动态策略列表
        self.costs = {}      # 动态成本字典
        self.history = []     # 记录演化历史

    def calculate_utility(self, strategy, popularity, sentiment, cost):
        """
        计算收益函数
        U_i(S) = α * Popularity(S,t) + β * Self_Sentiment(S) - C(S)
        """
        utility = self.alpha * popularity + self.beta * sentiment - cost
        return utility

    def replicator_dynamics(self, x, utilities):
        """
        复制者动态方程
        dx_S/dt = x_S * (U(S) - U_avg)

        参数:
            x: 当前策略分布 [x_S1, x_S2, x_S3]
            utilities: 各策略收益 [U_S1, U_S2, U_S3]
        """
        # 平均收益
        avg_utility = np.dot(x, utilities)

        # 计算变化率
        dx = x * (utilities - avg_utility)

        return dx

    def infer_strategy_cost(self, strategy_name):
        """
        根据策略名称推断认知成本
        """
        name_lower = strategy_name.lower()

        # 科学健康类 - 高成本
        if any(word in name_lower for word in ['科学', '健康', '营养', '均衡']):
            return 0.5

        # 快速捷径类 - 低成本
        if any(word in name_lower for word in ['快速', '捷径', '药']):
            return 0.1

        # 心理调适类 - 中等成本
        if any(word in name_lower for word in ['心理', '焦虑', '调适']):
            return 0.3

        # 放弃躺平类 - 低成本
        if any(word in name_lower for word in ['放弃', '躺平', '佛系']):
            return 0.2

        # 运动类 - 中高成本
        if any(word in name_lower for word in ['运动', '健身', '锻炼']):
            return 0.4

        # 默认中等成本
        return self.default_cost

    def simulate_evolution(self, initial_state, sentiment_dict, strategy_names,
                          dt=0.01, max_iter=1000, convergence_threshold=1e-4):
        """
        模拟演化过程 (支持动态策略数)

        参数:
            initial_state: 初始策略分布 dict {strategy_name: proportion}
            sentiment_dict: 各策略的平均情感 dict
            strategy_names: 策略名称列表
            dt: 时间步长
            max_iter: 最大迭代次数
            convergence_threshold: 收敛阈值
        """
        logger.info("开始演化博弈模拟...")

        # 动态设置策略
        self.strategies = strategy_names
        n_strategies = len(self.strategies)

        logger.info(f"  策略数量: {n_strategies}")
        logger.info(f"  策略列表: {self.strategies}")

        # 推断各策略的认知成本
        self.costs = {}
        for strategy in self.strategies:
            self.costs[strategy] = self.infer_strategy_cost(strategy)
            logger.info(f"    {strategy}: 成本={self.costs[strategy]:.2f}")

        # 初始化状态向量
        x = np.array([initial_state.get(s, 0) for s in self.strategies])

        # 归一化
        if x.sum() > 0:
            x = x / x.sum()
        else:
            x = np.ones(n_strategies) / n_strategies

        history = [x.copy()]

        for iteration in range(max_iter):
            # 计算各策略收益
            utilities = np.array([
                self.calculate_utility(
                    strategy,
                    popularity=x[i],
                    sentiment=sentiment_dict.get(strategy, 0),
                    cost=self.costs.get(strategy, self.default_cost)
                )
                for i, strategy in enumerate(self.strategies)
            ])

            # 计算变化
            dx = self.replicator_dynamics(x, utilities)

            # 更新状态
            x_new = x + dt * dx

            # 确保非负且和为1
            x_new = np.maximum(x_new, 0)
            if x_new.sum() > 0:
                x_new = x_new / x_new.sum()

            # 检查收敛
            if np.linalg.norm(x_new - x) < convergence_threshold:
                logger.info(f"  在第 {iteration} 次迭代收敛")
                break

            x = x_new
            history.append(x.copy())

        self.history = np.array(history)

        logger.info("模拟完成")
        result_str = ", ".join([f"{s}={x[i]:.3f}" for i, s in enumerate(self.strategies)])
        logger.info(f"  最终分布: {result_str}")

        return x, self.history

    def analyze_equilibrium(self, final_state):
        """分析均衡状态"""
        logger.info("\n分析演化均衡:")

        # 识别主导策略
        dominant_idx = np.argmax(final_state)
        dominant_strategy = self.strategies[dominant_idx]

        logger.info(f"  主导策略: {dominant_strategy} ({final_state[dominant_idx]:.1%})")

        # 多样性指数 (Shannon entropy)
        # 避免log(0)
        p = final_state[final_state > 0]
        diversity = -np.sum(p * np.log(p))

        logger.info(f"  策略多样性: {diversity:.3f}")

        # 判断均衡类型
        if final_state[dominant_idx] > 0.8:
            equilibrium_type = "主导均衡"
        elif diversity > 0.8:
            equilibrium_type = "混合均衡"
        else:
            equilibrium_type = "部分主导"

        logger.info(f"  均衡类型: {equilibrium_type}")

        return {
            'dominant_strategy': dominant_strategy,
            'dominant_ratio': final_state[dominant_idx],
            'diversity': diversity,
            'equilibrium_type': equilibrium_type
        }

    def visualize_evolution(self, save_path=None):
        """可视化演化过程"""
        if len(self.history) == 0:
            logger.warning("没有历史数据可视化")
            return

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
                '/root/weight/SIMSUN.TTC',
                '/root/weight/simsun.ttc',
            ]

            # 扩展路径：处理相对路径和用户目录
            expanded_paths = []
            for path in simsun_paths:
                expanded_path = os.path.expanduser(path)  # 处理 ~ 符号
                expanded_paths.append(expanded_path)
                # 也添加大写小写组合的路径
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
                    # 验证是否为TrueType字体
                    if path.lower().endswith('.ttf'):
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

        plt.figure(figsize=(12, 6))

        for i, strategy in enumerate(self.strategies):
            plt.plot(self.history[:, i], label=strategy, linewidth=2)

        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('策略占比', fontsize=12)
        plt.title('演化博弈:策略分布动态演化', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['figure_dpi'])
            logger.info(f"  可视化已保存: {save_path}")

        plt.close()


class MultiWindowGameSimulator:
    """多时间窗口博弈模拟器"""

    def __init__(self):
        self.game_model = EvolutionaryGameModel(
            alpha=GAME_CONFIG['alpha'],
            beta=GAME_CONFIG['beta'],
            costs=GAME_CONFIG['cost']
        )
        self.window_results = []

    def simulate_all_windows(self, count_matrix, sentiment_matrix):
        """
        模拟所有时间窗口 (支持动态策略)

        参数:
            count_matrix: 策略计数矩阵 (时间窗口 × 策略)
            sentiment_matrix: 情感矩阵 (时间窗口 × 策略)
        """
        logger.info("=" * 60)
        logger.info("开始多窗口演化博弈模拟 (动态策略)")
        logger.info("=" * 60)

        windows = count_matrix.index.tolist()
        all_strategies = count_matrix.columns.tolist()

        # 过滤掉异常策略
        all_strategies = [s for s in all_strategies if s != 'MS_Outlier']

        logger.info(f"全局策略集合: {all_strategies}")
        logger.info(f"策略数量: {len(all_strategies)}")

        for window in windows:
            logger.info(f"\n{'='*60}")
            logger.info(f"时间窗口: {window}")
            logger.info(f"{'='*60}")

            # 获取该窗口的数据
            counts = count_matrix.loc[window]
            sentiments = sentiment_matrix.loc[window]

            # 找出该窗口实际存在的策略 (count > 0)
            active_strategies = [s for s in all_strategies if counts.get(s, 0) > 0]

            if len(active_strategies) == 0:
                logger.warning(f"  窗口 {window} 无有效策略,跳过")
                continue

            logger.info(f"  活跃策略: {active_strategies} ({len(active_strategies)}个)")

            # 计算初始分布 (只考虑活跃策略)
            total = sum(counts.get(s, 0) for s in active_strategies)
            if total == 0:
                logger.warning(f"  窗口 {window} 策略计数为0,跳过")
                continue

            initial_state = {
                strategy: counts.get(strategy, 0) / total
                for strategy in active_strategies
            }

            # 计算平均情感 (只考虑活跃策略)
            sentiment_dict = {
                strategy: sentiments.get(strategy, 0)
                for strategy in active_strategies
            }

            # 运行模拟
            final_state, history = self.game_model.simulate_evolution(
                initial_state,
                sentiment_dict,
                active_strategies,
                dt=GAME_CONFIG['dt'],
                max_iter=GAME_CONFIG['max_iterations'],
                convergence_threshold=GAME_CONFIG['convergence_threshold']
            )

            # 分析均衡
            equilibrium = self.game_model.analyze_equilibrium(final_state)

            # 记录结果 (动态构建结果字典)
            result = {'window': window}

            # 记录所有全局策略的初始和最终值 (不存在的策略为0)
            for strategy in all_strategies:
                result[f'initial_{strategy}'] = initial_state.get(strategy, 0)

            for i, strategy in enumerate(active_strategies):
                result[f'final_{strategy}'] = final_state[i]

            # 不活跃的策略最终值为0
            for strategy in all_strategies:
                if strategy not in active_strategies:
                    result[f'final_{strategy}'] = 0

            result['dominant_strategy'] = equilibrium['dominant_strategy']
            result['diversity'] = equilibrium['diversity']
            result['equilibrium_type'] = equilibrium['equilibrium_type']
            result['n_active_strategies'] = len(active_strategies)

            self.window_results.append(result)

        logger.info("\n所有窗口模拟完成")

    def save_results(self):
        """保存模拟结果"""
        logger.info("\n保存演化博弈结果...")

        # 保存结果表
        results_df = pd.DataFrame(self.window_results)
        results_file = os.path.join(OUTPUT_FOLDER, 'stage4_game_results.csv')
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        logger.info(f"  结果已保存: {results_file}")

        # 可视化跨时间窗口的演化
        self.visualize_cross_window_evolution(results_df)

    def visualize_cross_window_evolution(self, results_df):
        """可视化跨时间窗口的演化 (动态策略)"""
        logger.info("生成跨窗口演化可视化...")

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
        # 找出所有策略列
        strategy_cols = [col for col in results_df.columns if col.startswith('initial_') or col.startswith('final_')]
        strategies = list(set([col.replace('initial_', '').replace('final_', '') for col in strategy_cols]))
        strategies = [s for s in strategies if s != 'MS_Outlier']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        x = range(len(results_df))
        n_strategies = len(strategies)
        width = 0.8 / n_strategies

        # 初始分布
        for i, strategy in enumerate(strategies):
            col_name = f'initial_{strategy}'
            if col_name in results_df.columns:
                offset = (i - n_strategies/2) * width
                axes[0].bar(
                    [xi + offset for xi in x],
                    results_df[col_name],
                    width,
                    label=strategy,
                    alpha=0.7
                )

        axes[0].set_title('各时间窗口初始策略分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('时间窗口')
        axes[0].set_ylabel('策略占比')
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3, axis='y')

        # 最终分布
        for i, strategy in enumerate(strategies):
            col_name = f'final_{strategy}'
            if col_name in results_df.columns:
                offset = (i - n_strategies/2) * width
                axes[1].bar(
                    [xi + offset for xi in x],
                    results_df[col_name],
                    width,
                    label=strategy,
                    alpha=0.8
                )

        axes[1].set_title('各时间窗口演化后均衡分布', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('时间窗口')
        axes[1].set_ylabel('策略占比')
        axes[1].legend(fontsize=8, ncol=2)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_FOLDER, 'stage4_cross_window_evolution.png'),
            dpi=VISUALIZATION_CONFIG['figure_dpi']
        )
        plt.close()
        logger.info("  可视化已保存")


def main():
    """主函数"""
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

    logger.info("\n" + "=" * 60)
    logger.info("阶段四总结")
    logger.info("=" * 60)
    logger.info(f"模拟窗口数: {len(simulator.window_results)}")

    # 统计主导策略分布
    dominant_strategies = [r['dominant_strategy'] for r in simulator.window_results]
    from collections import Counter
    strategy_counts = Counter(dominant_strategies)
    logger.info("各时间窗口的主导策略分布:")
    for strategy, count in strategy_counts.items():
        logger.info(f"  {strategy}: {count} 个窗口")


if __name__ == "__main__":
    main()