import jieba
import jieba.analyse
from snownlp import SnowNLP
from collections import Counter
import pandas as pd

# 添加自定义停用词
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就',
    '都', '而', '及', '与', '着', '或', '一个', '没有',
    '这个', '那个', '你们', '我们', '他们', '什么', '这样',
    '那样', '好', '看', '啊', '哦', '呢', '吧', '吗',
    '这', '那', '也', '很', '个', '之', '人', '上',
    '来', '去', '说', '要', '能', '会', '对', '到',
    '被', '还', '给', '等', '但', '并', '却', '以'
])

def perform_sentiment_analysis(df):
    """执行情感分析，返回情感得分和详细分析结果"""
    sentiments = []
    keywords = []
    sentences = []
    
    for comment in df['评论内容']:
        try:
            s = SnowNLP(str(comment))  # 确保输入是字符串
            sentiment_score = s.sentiments
            sentiments.append(sentiment_score)
            
            # 提取每条评论的关键词
            kw = jieba.analyse.textrank(str(comment), topK=3, withWeight=True)
            keywords.append([w for w, _ in kw])
            
            # 分句分析
            sentences.extend([(sent, SnowNLP(sent).sentiments) for sent in s.sentences])
        except Exception as e:
            # 如果处理失败，添加默认值
            sentiments.append(0.5)  # 默认中性
            keywords.append([])
            print(f"处理评论时出错: {str(e)}")
    
    return {
        'sentiment_scores': sentiments,
        'keywords': keywords,
        'sentence_analysis': sentences
    }

def extract_keywords(df, top_n=20, min_freq=2):
    """提取关键词并返回详细统计信息"""
    # 使用 TextRank 算法提取关键词
    text = ' '.join(df['评论内容'].astype(str))
    keywords_textrank = jieba.analyse.textrank(text, topK=top_n, withWeight=True)
    
    # 使用 TF-IDF 算法提取关键词
    keywords_tfidf = jieba.analyse.extract_tags(text, topK=top_n, withWeight=True)
    
    # 词频统计
    words = [w for w in jieba.cut(text) if w not in STOP_WORDS and len(w) > 1]
    word_freq = Counter(words)
    
    # 合并结果
    keywords_data = []
    for word, freq in word_freq.most_common(top_n):
        if freq >= min_freq:
            textrank_weight = next((w for k, w in keywords_textrank if k == word), 0)
            tfidf_weight = next((w for k, w in keywords_tfidf if k == word), 0)
            keywords_data.append({
                '关键词': word,
                '出现次数': freq,
                'TextRank权重': float(textrank_weight),
                'TF-IDF权重': float(tfidf_weight)
            })
    
    return pd.DataFrame(keywords_data)

def analyze_comment_trends(df):
    """分析评论趋势"""
    # 按时间统计评论数量
    df['日期'] = df['时间'].dt.date
    daily_stats = df.groupby('日期').agg({
        '评论内容': 'count',
        '点赞数': 'sum',
        '回复数': 'sum'
    }).reset_index()
    
    # 计算移动平均
    daily_stats['评论数_MA7'] = daily_stats['评论内容'].rolling(window=7).mean()
    
    return daily_stats