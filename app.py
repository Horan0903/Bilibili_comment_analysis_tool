import streamlit as st
import pandas as pd
from utils.data_loader import load_data_from_csv, load_data_from_api, load_bilibili_comments
from utils.text_analysis import perform_sentiment_analysis, extract_keywords
from utils.visualization import generate_wordcloud, plot_time_trend, plot_interaction_stats
import plotly.express as px
import openai
from datetime import datetime
import os
from openai import OpenAI
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import streamlit.components.v1 as components
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 设置页面配置
st.set_page_config(
    page_title="B站评论分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 添加停用词加函数
def load_stopwords():
    """加载停用词"""
    try:
        with open('LDA/Stopword.txt', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
        return stopwords
    except FileNotFoundError:
        st.warning("停用词文件未找到，将使用空的停用词集")
        return set()

# 添加文本预处理函数
def preprocess_text(text, stopwords):
    """文本预处理：分词并去除停用词"""
    if not isinstance(text, str):
        text = str(text) if isinstance(text, (float, int)) else ''
    
    seg = jieba.cut(text)
    return ' '.join([i for i in seg if i not in stopwords])

# 修改perform_lda_analysis函数
@st.cache_data(show_spinner=False)
def perform_lda_analysis(_texts, num_topics=5):
    """
    执行 LDA 主题建模分析
    
    Args:
        _texts: 评论文本列表
        num_topics: 主题数量
    Returns:
        lda_model: LDA模型
        corpus: 文档-词项矩阵
        dictionary: 词典
    """
    # 加载停用词
    stopwords = load_stopwords()
    
    # 加载自定义词典
    try:
        user_dicts = [
            "SogouLabDic.txt", "dict_baidu_utf8.txt", "dict_pangu.txt",
            "dict_sougou_utf8.txt", "dict_tencent_utf8.txt", "my_dict.txt"
        ]
        for dict_path in user_dicts:
            jieba.load_userdict(f"LDA/{dict_path}")
    except FileNotFoundError:
        st.warning("部分自定义词典未找到")
    
    # 文本预处理
    processed_texts = [preprocess_text(text, stopwords) for text in _texts]
    
    # 使用CountVectorizer构建文档-词项矩阵
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    
    # 转换为gensim格式
    corpus = Sparse2Corpus(doc_term_matrix, documents_columns=False)
    dictionary = corpora.Dictionary.from_corpus(
        corpus, 
        id2word=dict(enumerate(vectorizer.get_feature_names_out()))
    )
    
    # 训练LDA模型
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, dictionary

# 修改词云生成函数的缓存装饰器
@st.cache_data(show_spinner=False)
def generate_topic_wordcloud(_lda_model, _dictionary):
    """为每个主题生成词云"""
    wordclouds = []
    for idx, topic in enumerate(_lda_model.get_topics()):
        word_freq = dict(zip(_dictionary.values(), topic))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            max_words=50,
            font_path='LDA/Songti.ttc'  # 确保字体文件存在
        ).generate_from_frequencies(word_freq)
        
        # 将词云图保存到内存
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'主题 {idx + 1}')
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        wordclouds.append(buf)
    return wordclouds

def main():
    # 侧边栏
    st.sidebar.title("B站评论分析工具")
    
    # 数据输入选择
    data_source = st.sidebar.radio(
        "选择数据来源",
        ("本地CSV文件", "B站API")
    )
    
    # 主要内容区域
    if data_source == "本地CSV文件":
        uploaded_file = st.sidebar.file_uploader(
            "上传数据文件", 
            type=['csv', 'xlsx', 'xls'],
            help="支持的文件格式：CSV、Excel (xlsx/xls)"
        )
        if uploaded_file is not None:
            try:
                with st.spinner('正在加载数据...'):
                    df = load_data_from_csv(uploaded_file)
                
                if df is not None:  # 只有在用户确认列映射后才继续
                    if len(df) > 0:
                        st.success(f'成功加载 {len(df)} 条评论数据')
                        
                        # 添加数据预览和基本统计
                        with st.expander("数据预览与基本统计"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.dataframe(
                                    df.head(),
                                    use_container_width=True
                                )
                            with col2:
                                st.write("基本统计信息：")
                                st.write(f"- 评论总数：{len(df):,}")
                                st.write(f"- 评论时间范围：")
                                st.write(f"  - 开始：{df['时间'].min().strftime('%Y-%m-%d')}")
                                st.write(f"  - 结束：{df['时间'].max().strftime('%Y-%m-%d')}")
                                st.write(f"- 总点赞数：{df['点赞数'].sum():,}")
                                st.write(f"- 总回复数：{df['回复数'].sum():,}")
                        
                        # 添加数据过滤选项
                        with st.expander("数据过滤选项"):
                            col1, col2 = st.columns(2)
                            with col1:
                                min_date = st.date_input(
                                    "开始日期",
                                    value=df['时间'].min().date(),
                                    min_value=df['时间'].min().date(),
                                    max_value=df['时间'].max().date()
                                )
                            with col2:
                                max_date = st.date_input(
                                    "结束日期",
                                    value=df['时间'].max().date(),
                                    min_value=df['时间'].min().date(),
                                    max_value=df['时间'].max().date()
                                )
                            
                            # 根据日期过滤数据
                            mask = (df['时间'].dt.date >= min_date) & (df['时间'].dt.date <= max_date)
                            filtered_df = df.loc[mask]
                            
                            if len(filtered_df) != len(df):
                                st.info(f"已过滤，当前显示 {len(filtered_df)} 条评论（共 {len(df)} 条）")
                        
                        # 使用过滤后的数据进行分析
                        display_analysis(filtered_df)
                        
                        # 添加数据导出选项
                        with st.expander("导出分析结果"):
                            col1, col2 = st.columns(2)
                            with col1:
                                # 导出原始数据
                                csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    "下载过滤后的数据(CSV)",
                                    csv,
                                    "bilibili_comments_filtered.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                            with col2:
                                # 导出分析结果
                                analysis_results = {
                                    '评论总数': len(filtered_df),
                                    '时间范围': f"{min_date} 至 {max_date}",
                                    '平均点赞数': filtered_df['点赞数'].mean(),
                                    '平均回复数': filtered_df['回复数'].mean(),
                                    # 可以添加更多分析结果
                                }
                                st.json(analysis_results)
                                
                    else:
                        st.warning('未找到有效的评论数据')
            except Exception as e:
                st.error(f'加载文件时出错: {str(e)}')
    else:
        video_id = st.sidebar.text_input(
            "输入视频BV号或链接",
            help="支持BV号或完整视频链接"
        )
        
        max_comments = st.sidebar.number_input(
            "最大评论获取数量",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            help="设置为获取的最大评论数量，包括回复"
        )
        
        if video_id:
            try:
                with st.spinner('正在从B站获取评论数据...'):
                    df = load_bilibili_comments(video_id, max_comments)
                    
                if df is not None and len(df) > 0:
                    st.success(f'成功获取 {len(df)} 条评论数据')
                    
                    # 添加数预览
                    with st.expander("数据预览"):
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # 显示额外的用户信息统计
                        if '用户等级' in df.columns:
                            st.subheader("用户信息统计")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                avg_level = df['用户等级'].mean()
                                st.metric("平均用户等级", f"{avg_level:.1f}")
                                
                            with col2:
                                vip_count = (df['会员类型'] > 0).sum()
                                st.metric("大会员数量", vip_count)
                                
                            with col3:
                                gender_dist = df['性别'].value_counts()
                                st.write("性别分布：")
                                st.write(gender_dist)
                    
                    # 继续显示分析结果
                    display_analysis(df)
                else:
                    st.warning('未找到评论数据，请检查视频BV号是否正确')
            except Exception as e:
                st.error(f'获取数据时出错: {str(e)}')
                st.error('请确保输入的BV号正确，并且视频存在且可访问')

    # 添加页脚
    st.sidebar.markdown('---')
    st.sidebar.markdown('### 用说明')
    st.sidebar.markdown('''
    1. 选择据来源（本地CSV或B站API）
    2. 上传文件或输入BV号
    3. 等待数据加载完成
    4. 在不同标签页查看分析结果
    ''')

def display_analysis(df):
    # 添加页面标题和描述
    st.title("B站评论数据分析")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        📊 实时分析视频评论数据，深入了解用户反馈和互动情况。
    </div>
    """, unsafe_allow_html=True)
    
    # 创建多个标签页，添加AI分析标签
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 评论概览", "😊 情感分析", "🔍 关键词分析", "👥 互动数据", "🤖 AI分析"
    ])
    
    with tab1:
        st.markdown("""
        <div class='stCardContainer'>
            <h2>评论概览</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 概览指标
        metrics_container = st.container()
        with metrics_container:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,}</div>
                    <div class='metric-label'>总评论数</div>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
            with m2:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,}</div>
                    <div class='metric-label'>总点赞数</div>
                </div>
                """.format(df['点赞数'].sum()), unsafe_allow_html=True)
            with m3:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,}</div>
                    <div class='metric-label'>总回复数</div>
                </div>
                """.format(df['回复数'].sum()), unsafe_allow_html=True)
            with m4:
                avg_likes = df['点赞数'].mean()
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:.1f}</div>
                    <div class='metric-label'>平均点赞数</div>
                </div>
                """.format(avg_likes), unsafe_allow_html=True)
        
        # 时间趋势和词云
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(plot_time_trend(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("词云展示")
            wordcloud_img = generate_wordcloud(df)
            st.image(wordcloud_img)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.header("情感分析")
        
        # 算情感分析结果
        with st.spinner('正在进行情感分析...'):
            sentiment_analysis = perform_sentiment_analysis(df)
            df['情感得分'] = sentiment_analysis['sentiment_scores']
        
        # 定义情感分类
        df['情感类别'] = pd.cut(
            df['情感得分'], 
            bins=[-0.1, 0.3, 0.7, 1.1], 
            labels=['消极', '中性', '积极']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 情感分布饼图
            sentiment_dist = df['情感类别'].value_counts()
            fig_pie = px.pie(
                values=sentiment_dist.values,
                names=sentiment_dist.index,
                title="评论情感分布",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 情感得分分布直方图
            fig_hist = px.histogram(
                df,
                x='情感得分',
                nbins=30,
                title="情感得分分布",
                color_discrete_sequence=['#2ecc71']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # 展示典型评论
        st.subheader("典型评论示例")
        
        sentiment_tabs = st.tabs(["积极评论", "中性评论", "消极评论"])
        
        with sentiment_tabs[0]:
            positive_comments = df[df['情感类别'] == '积极'].nlargest(5, '点赞数')
            if not positive_comments.empty:
                for _, comment in positive_comments.iterrows():
                    st.info(f"👤 {comment['用户名']}: {comment['评论内容']}\n\n"
                           f"👍 {comment['点赞数']} | 💬 {comment['回复数']} | "
                           f"😊 情感得分: {comment['情感得分']:.2f}")
            else:
                st.write("暂无积极评论")
        
        with sentiment_tabs[1]:
            neutral_comments = df[df['情感类别'] == '中性'].nlargest(5, '点赞数')
            if not neutral_comments.empty:
                for _, comment in neutral_comments.iterrows():
                    st.info(f"👤 {comment['用户名']}: {comment['评论内容']}\n\n"
                           f"👍 {comment['点赞数']} | 💬 {comment['回复数']} | "
                           f"😐 情感得分: {comment['情感得分']:.2f}")
            else:
                st.write("暂无中性评论")
        
        with sentiment_tabs[2]:
            negative_comments = df[df['情感类别'] == '消极'].nlargest(5, '点赞数')
            if not negative_comments.empty:
                for _, comment in negative_comments.iterrows():
                    st.info(f"👤 {comment['用户名']}: {comment['评论内容']}\n\n"
                           f"👍 {comment['点赞数']} | 💬 {comment['回复数']} | "
                           f"😔 情感得分: {comment['情感得分']:.2f}")
            else:
                st.write("暂无消极评论")
        
        # 添加情感分析统计信息
        st.subheader("情感分析统计")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            avg_sentiment = df['情感得分'].mean()
            st.metric(
                "平均情感得分",
                f"{avg_sentiment:.2f}",
                delta=None,
                help="得分范围0-1，越高表示越积极"
            )
            
        with stats_col2:
            positive_rate = (df['情感类别'] == '积极').mean() * 100
            st.metric(
                "积极评论占比",
                f"{positive_rate:.1f}%",
                delta=None
            )
            
        with stats_col3:
            negative_rate = (df['情感类别'] == '消极').mean() * 100
            st.metric(
                "消极评论占比",
                f"{negative_rate:.1f}%",
                delta=None
            )
    
    with tab3:
        st.header("关键词分析")
        
        # 获取关键词
        keywords = extract_keywords(df)
        
        # 转换为DataFrame以便绘图
        keywords_df = pd.DataFrame(keywords, columns=['关键词', '出现次数'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 关键词柱状图
            fig_bar = px.bar(
                keywords_df.head(15),
                x='关键词',
                y='出现次数',
                title="热门关键词TOP15",
                color='出现次数',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # 关键词表格
            st.dataframe(
                keywords_df.head(20).style.background_gradient(cmap='YlOrRd'),
                use_container_width=True
            )
        
        # 添加LDA主题建模部分
        st.subheader("LDA主题建模分析")
        
        # 准备数据
        column_name = "评论内容"  # 使用正确的列名
        if column_name not in df.columns:
            st.error(f"找不到列 '{column_name}'，请确保数据包含评论内容列")
            return
            
        texts = df[column_name].tolist()
        
        # 添加主题数量选择器
        num_topics = st.slider("选择主题数量", min_value=2, max_value=10, value=5)
        
        if st.button("开始LDA分析"):
            with st.spinner("正在进行LDA主题建模分析..."):
                # 执行LDA分析
                lda_model, corpus, dictionary = perform_lda_analysis(texts, num_topics)
                
                # 显示主题词云
                st.write("### 主题词云")
                wordclouds = generate_topic_wordcloud(lda_model, dictionary)
                for buf in wordclouds:
                    st.image(buf, use_container_width=True)
                
                # 显示主题词分布
                st.write("### 主题词分布")
                cols = st.columns(num_topics)  # 创建与主题数量相同的列
                for idx, (topic_num, topic) in enumerate(lda_model.print_topics(-1)):
                    with cols[idx]:  # 在每个列中显示一个主题
                        st.write(f'主题 {idx + 1}:')
                        words = [(word.split('*')[1].strip().replace('"', ''), 
                                 float(word.split('*')[0])) 
                                for word in topic.split(' + ')]
                        topic_df = pd.DataFrame(words, columns=['词语', '权重'])
                        st.dataframe(topic_df)
                
                # 生成交互式可视化
                vis_data = pyLDAvis.gensim_models.prepare(
                    lda_model, corpus, dictionary, mds='mmds')
                
                # 翻译并显示HTML
                html_string = pyLDAvis.prepared_data_to_html(vis_data)
                translations = {
                    "Topic": "主题",
                    "Lambda": "λ 值",
                    "Relevance": "关联度",
                    "Overall term frequency": "整体词频",
                    "Top 30 Most Salient Terms": "前30个最显著的词语",
                    "Most relevant words for topic": "主题的相关词语",
                }
                for english, chinese in translations.items():
                    html_string = html_string.replace(english, chinese)
                
                components.html(html_string, width=1300, height=800)
                
                # 主题分布热力图
                doc_topics = []
                for doc in corpus:
                    topic_probs = lda_model.get_document_topics(doc)
                    probs = [0] * num_topics
                    for topic_id, prob in topic_probs:
                        probs[topic_id] = prob
                    doc_topics.append(probs)
                
                topic_dist_df = pd.DataFrame(doc_topics)
                topic_dist_df.columns = [f'主题{i+1}' for i in range(num_topics)]
                
                fig = px.imshow(
                    topic_dist_df.T,
                    labels=dict(x="文档", y="主题", color="概率"),
                    title="文档-主题分布热力图"
                )
                st.plotly_chart(fig)
        
        # 关键词搜索功能
        st.subheader("关键词搜索")
        search_word = st.text_input("输入关键词搜索相关评论：")
        if search_word:
            filtered_comments = df[df['评论内容'].str.contains(search_word, na=False)]
            if not filtered_comments.empty:
                st.write(f"找到 {len(filtered_comments)} 条相关评论：")
                for _, comment in filtered_comments.iterrows():
                    st.info(f"👤 {comment['用户名']}: {comment['评论内容']}\n\n"
                           f"👍 {comment['点赞数']} | 💬 {comment['回复数']}")
            else:
                st.warning("未找到相关评论")
    
    with tab4:
        st.header("互动数据")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 点赞数分布
            fig_likes = px.histogram(
                df,
                x='点赞数',
                title="评论点赞数分布",
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig_likes, use_container_width=True)
        
        with col2:
            # 回复数分布
            fig_replies = px.histogram(
                df,
                x='回复数',
                title="评论回复数分布",
                color_discrete_sequence=['#e74c3c']
            )
            st.plotly_chart(fig_replies, use_container_width=True)
        
        # 互动最高的评论
        st.subheader("互动最高的评论")
        df['互动总数'] = df['点赞数'] + df['回复数']
        top_comments = df.nlargest(5, '互动总数')
        for _, comment in top_comments.iterrows():
            st.success(f"👤 {comment['用户名']}: {comment['评论内容']}\n\n"
                      f"👍 {comment['点赞数']} | 💬 {comment['回复数']}")
    
    with tab5:
        st.header("AI 评论分析")
        
        # API配置
        api_key = st.text_input("API 密钥", type="password")
        base_url = st.text_input("Base URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = st.text_input("模型名称", "qwen-turbo")
        
        # 系统提示语
        system_prompt = st.text_area(
            "系统提示语", 
            "You are a helpful assistant that analyzes video comments. Please analyze if each comment is related to the given keywords. Answer with only '是' or '否'."
        )
        
        # 关键词输入
        keywords = st.text_input(
            "输入关键词（用逗号分隔）",
            "数据分析,数据,可视化,八爪鱼,数据建模",
            help="输入你想分析的关键词，用逗号分隔"
        )
        
        # 构建提示词模板
        user_prompt_template = (
            "请分析以下评论是否与这些关键词相关：{keywords}\n"
            "只需回答'是'或'否'。评论内容：{comment}"
        )
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.8)
        with col2:
            top_p = st.slider("Top P", 0.0, 1.0, 0.8)
        
        # 选择要分析的评论数量
        num_comments = st.number_input(
            "分析评论数量", 
            min_value=1,
            max_value=min(len(df), 100),
            value=min(10, len(df))
        )
        
        # 确保所有输入都已填写
        if st.button("开始AI分析"):
            if not api_key:
                st.error("请提供 API 密钥")
            elif not base_url:
                st.error("请提供 Base URL")
            elif not model_name:
                st.error("请提供模型名称")
            else:
                try:
                    # 初始化OpenAI客户端
                    client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    
                    # 获取待分析的评论
                    top_comments = df.nlargest(num_comments, '点赞数')
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    log_window = st.empty()
                    
                    # 存储分类结果
                    classifications = []
                    
                    # 处理关键词
                    keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
                    keywords_str = '、'.join(keywords_list)
                    
                    for i, (_, comment) in enumerate(top_comments.iterrows()):
                        log_window.text(f"正在分析第 {i+1}/{num_comments} 条评论...")
                        
                        try:
                            # 调用API进行分析
                            completion = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {'role': 'system', 'content': system_prompt},
                                    {'role': 'user', 'content': user_prompt_template.format(
                                        keywords=keywords_str,
                                        comment=comment['评论内容']
                                    )}
                                ],
                                temperature=temperature,
                                top_p=top_p
                            )
                            
                            classification = completion.choices[0].message.content.strip()
                            classifications.append(classification)
                            
                        except Exception as e:
                            st.error(f"分析评论时出错: {str(e)}")
                            classifications.append("无法分类")
                        
                        # 更新进度
                        progress_bar.progress((i + 1) / num_comments)
                    
                    # 显示分析结果
                    st.success(f"分析完成！关键词：{keywords_str}")
                    
                    # 将分类结果添加到DataFrame
                    results_df = top_comments.copy()
                    results_df['分类结果'] = classifications
                    
                    # 计算统计信息
                    related_comments = results_df[results_df['分类结果'] == '是']
                    total_likes = results_df['点赞数'].sum()
                    related_likes = related_comments['点赞数'].sum()
                    
                    # 显示统计信息
                    st.subheader("分析统计")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("分析评论总数", f"{len(results_df)}")
                    with col2:
                        st.metric("相关评论数", f"{len(related_comments)}")
                    with col3:
                        weighted_ratio = related_likes / total_likes if total_likes > 0 else 0
                        st.metric("相关评论加权占比", f"{weighted_ratio:.2%}")
                    
                    # 显示详细结果
                    st.subheader("详细分析结果")
                    st.dataframe(
                        results_df[['用户名', '评论内容', '点赞数', '分类结果']],
                        use_container_width=True
                    )
                    
                    # 提供下载选项
                    csv = results_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "下载分析结果",
                        csv,
                        "comment_analysis_results.csv",
                        "text/csv",
                        key='download-analysis'
                    )
                    
                except Exception as e:
                    st.error(f"AI分析出错: {str(e)}")

if __name__ == "__main__":
    main() 