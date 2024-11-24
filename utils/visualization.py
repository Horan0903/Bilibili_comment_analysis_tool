import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
import jieba
from collections import Counter
from utils.text_analysis import STOP_WORDS  # 导入停用词
import os
import streamlit as st

def generate_wordcloud(df, keywords_df=None, mask_path=None):
    """生成增强版词云图"""
    # 准备词频数据
    text = ' '.join(df['评论内容'].astype(str))
    
    # 使用jieba分词
    words = jieba.cut(text)
    # 过滤停用词
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    word_freq = Counter(words)
    
    try:
        # 设置字体路径
        font_paths = [
            'fonts/SourceHanSansSC-Regular.otf',  # 优先使用项目内的思源黑体
            '/System/Library/Fonts/PingFang.ttc',  # macOS自带字体备选
            '/System/Library/Fonts/STHeiti Light.ttc',
        ]
        
        # 查找可用的字体
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path is None:
            raise ValueError("未找到可用的中文字体文件")
            
        # 配置词云参数
        wordcloud_params = {
            'font_path': font_path,  # 设置字体路径
            'width': 1200,           # 增加宽度
            'height': 800,           # 增加高度
            'background_color': 'white',
            'max_words': 150,        # 增加显示词数
            'max_font_size': 150,    # 增加最大字号
            'min_font_size': 12,     # 设置最小字号
            'random_state': 42,
            'colormap': 'viridis',   # 使用viridis配色
            'prefer_horizontal': 0.7,
            'scale': 2               # 增加清晰度
        }
        
        # 生成词云
        wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(word_freq)
        
        # 生成图片
        plt.figure(figsize=(16, 9), dpi=200)  # 增加图片尺寸和DPI
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # 保存到内存
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        return img_buf
        
    except Exception as e:
        st.error(f"生成词云图时出错: {str(e)}")
        # 返回一个带有错误信息的图片
        plt.figure(figsize=(16, 9))
        plt.text(0.5, 0.5, f'词云生成失败: {str(e)}', 
                ha='center', va='center', fontproperties=font_path)
        plt.axis('off')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        return img_buf

def plot_keyword_analysis(keywords_df):
    """绘制关键词分析图表"""
    # 综合得分图
    keywords_df['综合得分'] = (
        keywords_df['出现次数'] * 0.4 +
        keywords_df['TextRank权重'] * 0.3 +
        keywords_df['TF-IDF权重'] * 0.3
    )
    
    fig = go.Figure()
    
    # 添加柱状图
    fig.add_trace(go.Bar(
        x=keywords_df['关键词'],
        y=keywords_df['出现次数'],
        name='出现次数',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # 添加线图
    fig.add_trace(go.Scatter(
        x=keywords_df['关键词'],
        y=keywords_df['综合得分'],
        name='综合得分',
        yaxis='y2',
        line=dict(color='rgb(219, 64, 82)', width=3)
    ))
    
    # 更新布局
    fig.update_layout(
        title='关键词分析',
        xaxis_title='关键词',
        yaxis_title='出现次数',
        yaxis2=dict(
            title='综合得分',
            overlaying='y',
            side='right'
        ),
        barmode='group',
        height=500
    )
    
    return fig

def plot_time_trend(df):
    """绘制时间趋势图"""
    daily_stats = df.groupby(df['时间'].dt.date).agg({
        '评论内容': 'count',
        '点赞数': 'sum',
        '回复数': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # 添加评论数量线
    fig.add_trace(go.Scatter(
        x=daily_stats['时间'],
        y=daily_stats['评论内容'],
        name='评论数',
        line=dict(color='rgb(49, 130, 189)', width=2)
    ))
    
    # 添加移动平均线
    ma7 = daily_stats['评论内容'].rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=daily_stats['时间'],
        y=ma7,
        name='7日移动平均',
        line=dict(color='rgb(204, 204, 204)', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='评论数量趋势',
        xaxis_title='日期',
        yaxis_title='评论数量',
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_interaction_stats(df):
    """绘制互动统计图表"""
    fig = go.Figure()
    
    # 添加点赞数分布
    fig.add_trace(go.Histogram(
        x=df['点赞数'],
        name='点赞数分布',
        nbinsx=30,
        marker_color='rgb(55, 83, 109)'
    ))
    
    # 添加回复数分布
    fig.add_trace(go.Histogram(
        x=df['回复数'],
        name='回复数分布',
        nbinsx=30,
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title='互动数据分布',
        xaxis_title='数量',
        yaxis_title='频次',
        barmode='overlay',
        height=400
    )
    
    # 设置透明度
    fig.update_traces(opacity=0.75)
    
    return fig 