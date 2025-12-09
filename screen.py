import pandas as pd
import numpy as np
import os
import glob
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import platform

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 自动查找中文字体 ==========
def find_chinese_font():
    """自动查找系统中可用的中文字体"""
    system = platform.system()

    # 扩展字体搜索路径
    possible_fonts = [
        'weight/SIMSUN.TTC',
        '/root/weight/simsun.ttc',
        './simsun.ttc',
        '/root/weight/SIMSUN.TTC',
        './SIMSUN.TTC'
    ]

    for font in possible_fonts:
        if os.path.exists(font):
            print(f"✅ 找到中文字体: {font}")
            return font

    print("⚠️ 未找到理想中文字体，尝试使用默认字体")
    return None


chinese_font_path = find_chinese_font()
if chinese_font_path is None:
    print("⚠️ 警告：未找到中文字体，词云可能无法显示中文！")
else:
    print(f"✅ 使用中文字体: {chinese_font_path}")

# ========== 参数配置 ==========
input_folder = "../autodl-fs/cleaned_data"
output_folder = "../autodl-fs/filtered_data"
os.makedirs(output_folder, exist_ok=True)

# 需要删除的特定词汇列表
WORDS_TO_REMOVE = [
    "封面新闻记者", "许敏", "编辑", "红星新闻记者", "责编", "李彬彬", "雷军", "我们尊重原创也注重分享", '李桐杉','赵鹏', '图片来源', '发布于广东省'
    "the end", "来源中国疾控中心", "杨一兵司向高欣", "责任编辑", "胡伊文", "版权原作者所有如在使用信息时侵犯了您的利益请及时联系我们将在24小时之内删除",
    "来源中国新闻社综合自中国新闻网红星新闻21世纪经济报道九派新闻大众日报", "哪吒", "来源中国疾控中心", '邮箱地址jpbl', '杨一兵司向高欣',
    "新政府或将提前释放囚犯", "3上半年进出口规模历史同期首次突破20万亿元我国外贸展现较强韧性", "4德国宣布移除5g网络中国通信企业组件我使馆坚决反对"
]

# ========== 新增：文本清洗函数 - 删除指定词汇 ==========
def remove_specific_words(text):
    """删除文本中指定的词汇"""
    if pd.isna(text):
        return ""
    
    text_str = str(text)
    # 按顺序删除词汇（长词汇优先，避免短词汇影响长词汇）
    for word in sorted(WORDS_TO_REMOVE, key=len, reverse=True):
        # 使用正则替换，确保完整匹配词汇
        text_str = re.sub(re.escape(word), ' ', text_str)
    # 清理多余空格
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str

# ========== 1. 定义体重管理相关关键词及需排除的关键词 ==========
weight_keywords = {
    '核心词': ['体重', '减肥', '瘦身', '减重', '塑形', '健身', '体脂', 'BMI', '身材管理', '超重', '肥胖', '食养',
               '减脂', '增肌', '体型', '身材', '体重控制', '体重管理', '健康体重', '理想体重', '控制体重', '标准体重'],

    '方法词': ['节食', '断食', '轻断食', '生酮', '低碳', '卡路里', '热量', '代餐',
               '运动', '跑步', '瑜伽', '健身房', '力量训练', '有氧', '无氧',
               '饮食控制', '管住嘴', '迈开腿', '营养餐', '健康餐', '燃脂',
               'HIIT', '燃脂', '消耗', '代谢', '基础代谢', '热量缺口', '热量盈余',
               '饮食计划', '训练计划', '健身计划', '减脂期', '增肌期', '平台期突破',
               '间歇性断食', '碳水循环', '低GI饮食', '高蛋白', '膳食纤维', '粗粮',
               '控糖', '少油', '清淡饮食', '均衡饮食', '饱腹感'],

    '产品词': ['减肥药', '代餐粉', '奶昔', '酵素', '左旋肉碱', '蛋白粉', '燃脂',
               '瘦身霜', '塑身衣', '体重秤', '智能秤', '体脂秤',
               '减肥茶', '瘦身包', '减肥贴', '瘦腿袜', '束腰', '健身器材',
               '运动补剂', 'BCAA', 'CLA', '绿茶提取物', '藤黄果', '白芸豆',
               '减肥仪器', '射频', '冷冻减脂', '抽脂', '吸脂', '代餐奶昔', '胶原蛋白肽',
               '瘦身咖啡', '膳食纤维粉', '益生菌', '减肥糖果', '体重管理茶', '司美格鲁肽', '肽'],

    '效果词': ['瘦了', '减了', '掉秤', '平台期', '反弹', '暴瘦', '月瘦',
               '减肥', '瘦身', '变瘦', '体重下降', '瘦下来',
               '减掉', '瘦身成功', '体重减轻', '体脂下降', '围度减少',
               '马甲线', '人鱼线', '腹肌', '翘臀', '瘦腿', '瘦腰',
               '体重回升', '复胖', '体重波动', '代谢适应', '突破平台期',
               '小蛮腰', '穿衣显瘦', '视觉瘦', '瘦一圈', '瘦了10斤'],

    '相关词': ['肥胖', '超重', '发胖', '长胖', '增重', '体重增加', '变胖',
               '腰围', '腹部', '小肚子', '游泳圈', '拜拜肉', '大象腿',
               '双下巴', '蝴蝶袖', '富贵包', '梨形身材', '苹果形身材',
               '基础体重', '目标体重', '理想体重', '标准体重',
               '皮下脂肪', '内脏脂肪', '肌肉量', '水分率', '骨量',
               '体脂率', 'BMI指数', '腰臀比', '体重曲线', '体型变化'],

    '网络用语': ['打卡', '自律', '坚持', 'Day', 'day', '减脂餐', '健身餐',
                 '暴食', '欺骗餐', '放纵餐', '轻断食后', '空腹有氧',
                 '撸铁', '举铁', '有氧运动', '无氧运动', '核心训练',
                 '体态', '体态改善', '体脂率', '肌肉线条',
                 '减肥打卡', '健身打卡', '运动打卡', '饮食记录',
                 '胖友', '减友', '健身伙伴', '减肥搭子', '卷王',
                 '#减肥日记#', '#瘦身打卡#', '#健身日常#', '#体重管理#'],

    '健康指标': ['腰臀比', '体脂百分比', '肌肉率', '基础代谢率', 'BMI指数',
                 '肥胖程度', '体重指数', '健康风险', '代谢综合征',
                 '胰岛素抵抗', '血脂', '血糖', '血压', '肝功能',
                 '心血管健康', '胆固醇', '甘油三酯', '空腹血糖'],

    '心理行为': ['食欲', '饥饿感', '饱腹感', '食物渴望', '情绪性进食', '摆烂'
                                                                       '压力肥', '熬夜胖', '代谢慢', '易胖体质',
                 '喝水都胖',
                 '自律', '坚持', '动力', '目标', '成果', '变化',
                 '减肥心态', '身材焦虑', '自信心', '形象管理', '健康意识'],
    # ========== 新增：对立观点关键词 ==========
    '反减肥词': [
        # 身体接纳
        '接受自己', '爱自己', '身材自由', '不必瘦', '胖也可以美',
        '大码女孩', '大码', 'body positive', '身体积极',
        
        # 反对焦虑
        '拒绝焦虑', '反对身材焦虑', '不要被绑架', '容貌焦虑', '外貌焦虑',
        '反PUA', '反内卷', '身材内卷',
        
        # 放弃/失败叙事
        '放弃减肥', '不减了', '躺平', '佛系', '随缘',
        '减肥失败', '又胖了', '反弹', '复胖', '瘦不下来',
        
        # 健康优先
        '健康第一', '不伤身', '安全最重要', '别伤身体',
        '过度减肥', '减肥危害', '厌食症', '暴食症',
    ],
    
    '心理健康词': [
        '心理', '心态', '情绪', '压力', '抑郁', '自信',
        '心理咨询', '心理医生', '心理建设', '心理调节',
        '自我认同', '自我价值', '自我接纳', '自尊',
        '完美主义', '强迫', '控制欲', '补偿心理',
    ],
    
    '医疗干预词': [
        # 药物
        '司美格鲁肽', 'Semaglutide', '利拉鲁肽', 'Wegovy', 'Ozempic',
        '奥利司他', '二甲双胍', '减肥针', 'GLP-1',
        
        # 手术
        '吸脂', '抽脂', '溶脂', '缩胃', '袖胃', '胃旁路',
        '减重手术', '代谢手术', '医美', '整形',
        
        # 医疗机构
        '医院', '医生建议', '内分泌科', '营养科', '肥胖门诊',
    ],
    
    '社会文化词': [
        '审美', '标准', '媒体', '明星', '网红', '博主',
        '白幼瘦', 'A4腰', '反手摸肚脐', '锁骨放硬币',
        '直男审美', '凝视', '物化', '评价',
        '社会压力', '同辈压力', '家人施压', '被催',
    ],
}

# 扩展需要排除的关键词列表
exclude_keywords = [
    # 医疗相关
    '医疗器械', '医疗设备', '患者',
    '医疗器械注册', 
    # 无关领域
    '招聘', '就业', '求职', '薪资', '面试', '简历', '职位', '工作', '感情戏',
    '宠物', '猫', '狗', '动物', '宠物食品', '兽医', '考试', '战役纪念', '爱国主义', 
    '骨质疏松症', '骨密度', '补钙', '骨骼健康', '骨骼', '乡村', '创业', '早安', '学无止境'
    # 金融相关
    '贷款', '信用卡', '投资', '理财', '保险', '股票', '基金', '证券',
    '银行', '支付', '转账', '收款', '提现', '退款', '押金', '保证金'
]

# 为不同关键词类别设置不同权重
keyword_weights = {
    '核心词': 1.5,
    '方法词': 1.5,
    '效果词': 3.0,
    '相关词': 2.5,
    '网络用语': 3.0,
    '产品词': 3.0,
    '健康指标': 1.5,
    '心理行为': 3.0,
    '反减肥词': 2.5,      
    '心理健康词': 2.2,    
    '医疗干预词': 2.0,
    '社会文化词': 1.8,
}

# 计算加权关键词列表
weighted_keywords = {}
for category, keywords in weight_keywords.items():
    weight = keyword_weights.get(category, 1.0)
    for keyword in keywords:
        weighted_keywords[keyword] = weight

all_keywords = list(weighted_keywords.keys())
all_weights = list(weighted_keywords.values())

print(f"共定义了 {len(all_keywords)} 个体重管理相关关键词")
print(f"关键词权重范围: {min(all_weights)} - {max(all_weights)}")
print(f"需排除的关键词数量: {len(exclude_keywords)}")
print(f"体重关键词示例: {all_keywords[:10]}...")


def length_factor(text):
    """根据文本长度调整置信度（避免短文本误杀、长文本刷分）"""
    if pd.isna(text):
        return 0.7

    L = len(str(text))

    if L < 20:
        return 1.1
    elif L < 50:
        return 1.05
    elif L < 200:
        return 1
    elif L < 500:
        return 0.9
    else:
        return 0.7  # 上限限制，避免长文刷分


# ========== 2. 数据筛选函数（增强版） ==========
def calculate_keyword_score(text, keywords, weights, exclude_words):
    """
    计算文本关键词得分（加权）+ 文本长度因子修正
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0, []

    # 先删除指定词汇
    cleaned_text = remove_specific_words(text)
    text_lower = cleaned_text.lower()

    # 排除词检查
    for exclude in exclude_words:
        if exclude in text_lower:
            return 0.0, []

    score = 0.0
    matched_keywords = []

    # 关键词匹配得分
    for i, keyword in enumerate(keywords):
        if keyword in text_lower:
            score += weights[i]
            matched_keywords.append((keyword, weights[i]))

    # 长度置信度加成
    score *= length_factor(cleaned_text)

    return score, matched_keywords


def is_weight_related_enhanced(row, text_columns, keywords, weights, exclude_words,
                               min_score=3.5, min_core_keywords=2, min_decision_keywords=0):
    """
    增强版判断函数，新增决策关键词匹配要求
    min_decision_keywords: 至少匹配1个决策相关关键词
    """
    total_score = 0.0
    all_matches = []
    core_keyword_count = 0
    decision_keyword_count = 0  # 新增：决策关键词计数器

    # 先获取所有决策相关关键词（用于判断）
    decision_keywords = weight_keywords.get('决策选择', [])
    anti_diet_keywords = weight_keywords.get('反减肥词', [])

    # 检查所有文本列
    for col in text_columns:
        if col in row and not pd.isna(row[col]):
            # 先删除指定词汇
            text = remove_specific_words(row[col])
            text_lower = text.lower()

            # 检查排除词
            for exclude in exclude_words:
                if exclude in text_lower:
                    return False, 0.0, []

            # 计算得分
            score, matches = calculate_keyword_score(text, keywords, weights, exclude_words)
            total_score += score
            all_matches.extend(matches)

            # 统计核心关键词和决策关键词
            for keyword, weight in matches:
                if weight >= 2.5:
                    core_keyword_count += 1
                if keyword in decision_keywords:  # 新增：统计决策关键词匹配数
                    decision_keyword_count += 1
                if keyword in anti_diet_keywords:  # ⭐ 新增
                    anti_diet_keyword_count += 1

    # 应用多重判断标准（新增决策关键词要求）
    is_related = (
        (total_score >= min_score and core_keyword_count >= min_core_keywords) or
        (anti_diet_keyword_count >= 2)  # ⭐ 新增：有2个以上反减肥词就算相关
    )

    return is_related, total_score, all_matches


def filter_weight_data_enhanced(df, keywords, weights, exclude_words, min_score=6.0):
    """
    增强版筛选函数
    """
    # 需要检查的文本字段
    text_columns = ['标题_微博内容', '全文内容', '原微博内容', '根微博标题',
                     '话题', '用户简介', '内容', '图文识别']

    # 找出实际存在的文本字段
    existing_text_cols = [col for col in text_columns if col in df.columns]

    if not existing_text_cols:
        print("⚠️ 未找到可用的文本字段")
        return df.iloc[:0], df, []  # 返回空的相关数据和全部无关数据

    print(f"🔍 使用以下文本列进行筛选: {existing_text_cols}")

    # 先对所有文本列执行指定词汇删除
    for col in existing_text_cols:
        df[col] = df[col].apply(remove_specific_words)

    # 添加评分和匹配关键词列
    df['weight_score'] = 0.0
    df['matched_keywords'] = ''
    df['is_weight_related'] = False

    # 逐行处理
    print("🔄 正在筛选体重管理相关数据，请稍候...")
    matched_keywords_list = []

    for idx, row in df.iterrows():
        is_related, score, matches = is_weight_related_enhanced(
            row, existing_text_cols, keywords, weights, exclude_words,
            min_score=min_score, min_core_keywords=1
        )

        df.at[idx, 'weight_score'] = score
        df.at[idx, 'is_weight_related'] = is_related
        df.at[idx, 'matched_keywords'] = '; '.join([f"{kw}({w})" for kw, w in matches[:10]])  # 保存最多10个匹配

        if matches:
            matched_keywords_list.extend([kw for kw, _ in matches])

    # 分离相关和无关数据
    related_df = df[df['is_weight_related']].copy()
    unrelated_df = df[~df['is_weight_related']].copy()

    # 删除临时列
    for df_temp in [related_df, unrelated_df]:
        for col in ['weight_score', 'matched_keywords', 'is_weight_related']:
            if col in df_temp.columns:
                df_temp.drop(col, axis=1, inplace=True)

    # 分析匹配关键词频率
    keyword_freq = Counter(matched_keywords_list)

    return related_df, unrelated_df, keyword_freq.most_common(20)


# ========== 3. 文本质量检查 ==========
def check_text_quality(text):
    """
    检查文本质量，过滤低质量内容
    """
    # 先删除指定词汇再检查质量
    cleaned_text = remove_specific_words(text)
    
    if pd.isna(cleaned_text) or str(cleaned_text).strip() == '':
        return False, "空文本"

    text = str(cleaned_text).strip()

    # 检查文本长度
    if len(text) < 15:  # 太短
        return False, "文本过短"

    # 检查重复字符
    if re.search(r'(.)\1{5,}', text):  # 同一字符重复6次以上
        return False, "重复字符过多"

    # 检查URL比例
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if len(urls) > 3:  # URL过多
        return False, "URL过多"

    # 检查特殊字符比例
    special_chars = re.findall(r'[^\w\s\u4e00-\u9fff]', text)
    if len(special_chars) / len(text) > 0.4:  # 特殊字符比例过高
        return False, "特殊字符过多"

    return True, "高质量"


# ========== 4. 批量处理清洗后的文件 ==========
all_files = glob.glob(os.path.join(input_folder, "cleaned_*.csv"))

if not all_files:
    raise FileNotFoundError(f"⚠️ 未在 {input_folder} 中找到清洗后的 CSV 文件")

print(f"\n📊 找到 {len(all_files)} 个清洗后的文件，开始筛选...")

all_related = []
all_unrelated = []
all_keyword_freq = Counter()

# 处理每个文件
for f in all_files:
    try:
        filename = os.path.basename(f)
        print(f"\n📄 处理文件: {filename}")

        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
        print(f"   原始数据量: {len(df)} 条")

        # ========== 新增：对所有列应用指定词汇删除 ==========
        print(f"   对所有列执行指定词汇删除...")
        for col in df.columns:  # 遍历所有列
            df[col] = df[col].apply(remove_specific_words)  # 应用删除函数

        # 确定主要文本列（用于后续筛选逻辑）
        text_col = None
        for col in ['全文内容', '标题_微博内容', '原微博内容', '内容']:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            print(f"⚠️ 未找到合适的文本列，跳过文件 {filename}")
            continue

        print(f"   使用文本列: {text_col}")

        # 应用增强版筛选
        related_df, unrelated_df, keyword_freq = filter_weight_data_enhanced(
            df, all_keywords, all_weights, exclude_keywords, min_score=3.5
        )

        # 应用文本质量过滤
        if not related_df.empty and text_col in related_df.columns:
            quality_mask = related_df[text_col].apply(lambda x: check_text_quality(x)[0])
            filtered_count = len(related_df) - sum(quality_mask)
            related_df = related_df[quality_mask]
            print(f"   文本质量过滤后: {len(related_df)} 条 (移除 {filtered_count} 条低质量数据)")

        # 更新关键词频率统计
        for kw, count in keyword_freq:
            all_keyword_freq[kw] += count

        # 保存相关数据
        if len(related_df) > 0:
            original_filename = filename.replace('cleaned_', '')
            related_output = os.path.join(output_folder, f"weight_related_{original_filename}")
            related_df.to_csv(related_output, index=False, encoding='utf-8-sig')
            all_related.append(related_df)
            print(f"✅ {original_filename}: 相关数据 {len(related_df)} 条")

        # 保存部分无关数据用于分析
        if len(unrelated_df) > 0:
            sample_size = min(1000, len(unrelated_df))  # 只保存1000条样本
            unrelated_sample = unrelated_df.sample(n=sample_size, random_state=42)
            original_filename = filename.replace('cleaned_', '')
            unrelated_output = os.path.join(output_folder, f"unrelated_sample_{original_filename}")
            unrelated_sample.to_csv(unrelated_output, index=False, encoding='utf-8-sig')
            all_unrelated.append(unrelated_sample)
            print(f"   无关数据样本 {sample_size} 条 (总计 {len(unrelated_df)} 条)")

    except Exception as e:
        print(f"❌ 处理文件 {f} 时出错: {str(e)}")

# ========== 5. 合并所有相关数据 + 去重 ==========
if all_related:
    combined_related = pd.concat(all_related, ignore_index=True)
    initial_count = len(combined_related)
    print(f"\n🔍 初始合并数据量: {initial_count} 条")


    # ====== 新增：全局去重处理 ======
    def remove_duplicates(df):
        """智能去重：优先ID去重，其次内容相似度去重"""
        print("\n🔄 开始去重处理...")
        dup_count = 0

        # 1. 尝试用唯一ID去重 (如微博ID)
        id_cols = ['微博ID', 'id', 'ID', '文章ID']
        id_col = next((col for col in id_cols if col in df.columns), None)

        if id_col:
            before = len(df)
            df = df.drop_duplicates(subset=[id_col], keep='first')
            after = len(df)
            dup_count += (before - after)
            print(f"   ✅ 基于唯一ID去重: 移除 {before - after} 条重复数据 (剩余 {after} 条)")

        # 2. 内容相似度去重 (处理相同内容不同ID的情况)
        content_col = None
        for col in ['全文内容', '标题_微博内容', '原微博内容', '内容']:
            if col in df.columns:
                content_col = col
                break

        if content_col and not df.empty:
            # 创建内容指纹 (标准化处理)
            df['content_fingerprint'] = df[content_col].str.lower().str.strip()
            df['content_fingerprint'] = df['content_fingerprint'].str.replace(r'\s+', ' ', regex=True)

            # 优先保留高互动量的记录
            interaction_cols = ['点赞数', '评论数', '转发数', '阅读数_浏览热度']
            interaction_col = next((col for col in interaction_cols if col in df.columns), None)

            if interaction_col:
                df = df.sort_values(by=interaction_col, ascending=False)

            # 执行去重
            before = len(df)
            df = df.drop_duplicates(subset=['content_fingerprint'], keep='first')
            after = len(df)
            dup_count += (before - after)
            print(f"   ✅ 基于内容指纹去重: 移除 {before - after} 条重复数据 (剩余 {after} 条)")

            # 清理临时列
            df.drop('content_fingerprint', axis=1, inplace=True)

        print(f"   🧹 总共移除重复数据: {dup_count} 条")
        return df, dup_count


    # 执行去重
    combined_related, total_duplicates = remove_duplicates(combined_related)

    # ====== 文本质量二次过滤 (在去重后执行) ======
    final_text_col = None
    for col in ['全文内容', '标题_微博内容', '原微博内容', '内容']:
        if col in combined_related.columns:
            final_text_col = col
            break

    if final_text_col:
        quality_mask = combined_related[final_text_col].apply(lambda x: check_text_quality(x)[0])
        filtered_count = len(combined_related) - sum(quality_mask)
        combined_related = combined_related[quality_mask]
        print(f"\n🔧 文本质量过滤: 移除 {filtered_count} 条低质量数据 (剩余 {len(combined_related)} 条)")
    else:
        print("\n⚠️ 未找到合适文本列进行质量过滤")

    # 保存合并数据
    combined_output = os.path.join(output_folder, "all_weight_related_combined.csv")
    combined_related.to_csv(combined_output, index=False, encoding='utf-8-sig')
    print(f"\n✅✅ 去重+过滤后数据: {len(combined_related)} 条 (原始合并量: {initial_count})")
    print(f"   保存位置: {combined_output}")

    # 生成关键词频率报告
    keyword_report = os.path.join(output_folder, "keyword_frequency_report.txt")
    with open(keyword_report, 'w', encoding='utf-8') as f:
        f.write("体重管理关键词频率分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"初始合并数据量: {initial_count} 条\n")
        f.write(f"去重移除量: {total_duplicates} 条\n")
        f.write(f"文本质量过滤移除量: {filtered_count} 条\n")
        f.write(f"最终有效数据量: {len(combined_related)} 条\n\n")
        f.write("最常匹配的关键词 (Top 50):\n")
        f.write("-" * 30 + "\n")
        for i, (kw, count) in enumerate(all_keyword_freq.most_common(50), 1):
            f.write(f"{i:2d}. {kw}: {count} 次匹配\n")
    print(f"✅ 关键词频率报告已生成: {keyword_report}")

else:
    print("\n⚠️⚠️ 未找到任何体重管理相关数据")

# ========== 6. 基础数据分析 ==========
if all_related:
    print("\n" + "=" * 60)
    print("📊 数据概览分析")
    print("=" * 60)

    df_analysis = combined_related.copy()

    # 6.1 数据量统计
    print(f"\n✅ 最终有效数据量: {len(df_analysis)} 条")

    # 6.2 时间分布分析
    if '日期' in df_analysis.columns:
        df_analysis['日期'] = pd.to_datetime(df_analysis['日期'], errors='coerce')
        df_analysis['年月'] = df_analysis['日期'].dt.to_period('M')
        valid_dates = df_analysis['年月'].notna().sum()

        if valid_dates > 0:
            time_dist = df_analysis['年月'].value_counts().sort_index()
            print(f"\n📅 时间分布 (前10个月):")
            print(time_dist.head(10))

            # 可视化时间分布
            plt.figure(figsize=(12, 6))
            time_dist.head(12).plot(kind='bar')
            plt.title('体重管理内容月度分布 (Top 12个月)')
            plt.xlabel('年月')
            plt.ylabel('文档数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "time_distribution.png"), dpi=300)
            plt.close()
            print("✅ 时间分布图已保存")

    # 6.3 情绪分析
    if '微博情绪' in df_analysis.columns:
        emotion_dist = df_analysis['微博情绪'].value_counts()
        print(f"\n😊 情绪分布:")
        print(emotion_dist)

        # 可视化情绪分布
        plt.figure(figsize=(10, 6))
        emotion_dist.plot(kind='pie', autopct='%1.1f%%')
        plt.title('体重管理内容情绪分布')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "emotion_distribution.png"), dpi=300)
        plt.close()
        print("✅ 情绪分布图已保存")

    # 6.4 互动数据统计
    interaction_cols = ['转发数', '评论数', '点赞数', '阅读数_浏览热度']
    existing_interaction = [col for col in interaction_cols if col in df_analysis.columns]

    if existing_interaction:
        print(f"\n💬 互动数据统计:")
        print(df_analysis[existing_interaction].describe())

        # 互动量分布可视化
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(existing_interaction[:3], 1):  # 只显示前3个
            plt.subplot(2, 2, i)
            df_analysis[col].hist(bins=30)
            plt.title(f'{col}分布')
            plt.xlabel(col)
            plt.ylabel('频率')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "interaction_distribution.png"), dpi=300)
        plt.close()
        print("✅ 互动数据分布图已保存")

# ========== 7. 准备BERTopic分析的文本数据 ==========
if all_related:
    # 重新确定文本列
    bertopic_text_col = None
    for col in ['全文内容', '标题_微博内容', '原微博内容', '内容']:
        if col in df_analysis.columns:
            bertopic_text_col = col
            break

    if bertopic_text_col:
        # 准备文本数据（已经过指定词汇删除）
        texts = df_analysis[bertopic_text_col].fillna('').astype(str)
        texts = texts[texts.str.strip() != '']  # 过滤空文本

    # 保存用于BERTopic的文本
    bertopic_output = os.path.join(output_folder, "texts_for_bertopic.txt")
    with open(bertopic_output, 'w', encoding='utf-8') as f:
        for text in texts:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if len(clean_text) >= 20:  # 只保存足够长的文本
                f.write(clean_text + '\n')

    print(f"\n✅ 已准备 {len(texts)} 条文本用于BERTopic分析")
    print(f"   文本文件: {bertopic_output}")

    # 保存带索引的完整数据用于后续关联
    bertopic_data = df_analysis.copy()
    bertopic_data = bertopic_data[bertopic_data[text_col].notna()]
    bertopic_data = bertopic_data[bertopic_data[text_col].str.strip() != '']
    bertopic_data_output = os.path.join(output_folder, "data_for_bertopic.csv")
    bertopic_data.to_csv(bertopic_data_output, index=False, encoding='utf-8-sig')
    print(f"   数据文件: {bertopic_data_output}")

# ========== 8. 生成筛选报告 ==========
report_file = os.path.join(output_folder, "data_filtering_report.md")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 体重管理数据筛选报告\n\n")
    f.write(f"## 筛选概览\n\n")
    f.write(f"- **筛选时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(
        f"- **原始数据总量**: {sum(len(pd.read_csv(f, encoding='utf-8-sig', low_memory=False)) for f in all_files)} 条\n")
    f.write(f"- **相关数据总量**: {len(combined_related) if all_related else 0} 条\n")
    f.write(
        f"- **筛选比例**: {len(combined_related) / sum(len(pd.read_csv(f, encoding='utf-8-sig', low_memory=False)) for f in all_files) * 100:.1f}%\n")
    f.write(f"- **关键词权重策略**: 核心词(3.0) > 方法词(2.5) > 效果词(2.0) > 心理行为(1.8)\n")
    f.write(f"- **排除关键词数量**: {len(exclude_keywords)} 个\n")
    f.write(f"- **额外删除词汇**: {', '.join(WORDS_TO_REMOVE)}\n\n")  # 新增：记录删除的词汇

    f.write("## 筛选参数\n\n")
    f.write("- **最小相关分数**: 7.0\n")
    f.write("- **最少核心关键词**: 1个\n")
    f.write("- **文本质量要求**: 长度15-1000字符，特殊字符<40%\n\n")

    f.write("## 高频匹配关键词 (Top 20)\n\n")
    f.write("| 排名 | 关键词 | 匹配次数 |\n")
    f.write("|-----|-------|---------|\n")
    for i, (kw, count) in enumerate(all_keyword_freq.most_common(20), 1):
        f.write(f"| {i} | {kw} | {count} |\n")

    f.write("\n## 建议\n\n")
    f.write("- 相关数据量充足，可直接进行主题建模分析\n")
    f.write("- 建议在BERTopic分析中使用较低的MIN_TOPIC_SIZE(300-500)以减少异常主题\n")
    f.write("- 关注'减肥'、'瘦身'、'平台期'等高频关键词相关主题\n")

print(f"\n✅ 筛选报告已生成: {report_file}")

print("\n" + "=" * 60)
print("✅✅ 数据筛选完成总结")
print("=" * 60)
if all_related:
    print(f"✅ 相关数据总计: {len(combined_related)} 条")
    print(
        f"   数据质量: {len(combined_related) / sum(len(pd.read_csv(f, encoding='utf-8-sig', low_memory=False)) for f in all_files) * 100:.1f}%")
print(f"📁 输出文件夹: {output_folder}")
print("\n➡️ 下一步: 运行BERTopic分析脚本进行主题聚类")