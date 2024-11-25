import streamlit as st
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import jieba
from jieba import analyse
import io


# 加载自定义词典和停用词
def load_resources():
    user_dicts = [
        "SogouLabDic.txt", "dict_baidu_utf8.txt", "dict_pangu.txt",
        "dict_sougou_utf8.txt", "dict_tencent_utf8.txt", "my_dict.txt"
    ]
    for dict_path in user_dicts:
        jieba.load_userdict(f"/Users/liuhaoran/LHR/PycharmProjects/Comment analysis/LDA/{dict_path}")

    with open('/Users/liuhaoran/LHR/PycharmProjects/Comment analysis/LDA/Stopword.txt') as f:
        stopwords = set(line.strip() for line in f)
    return stopwords


stopwords = load_resources()


# 自定义停用词功能
def get_stopwords(custom_input):
    stopwords = load_resources()  # 加载默认停用词
    if custom_input:
        custom_stopwords = set(custom_input.splitlines())
        stopwords.update(custom_stopwords)
    return stopwords


# 文本预处理
def preprocess_text(text, stopwords):
    if not isinstance(text, str):
        text = str(text) if isinstance(text, (float, int)) else ''

    seg = jieba.cut(text)
    return ' '.join([i for i in seg if i not in stopwords])


# 关键词提取
def extract_keywords(text):
    text = str(text) if isinstance(text, (float, int, str)) else ''
    keywords = analyse.extract_tags(text.strip(), allowPOS=(
    'ns', 'nr', 'nt', 'nz', 'nl', 'n', 'vn', 'vd', 'vg', 'v', 'vf', 'a', 'an', 'i'))
    return ' '.join(keywords)


# 主题建模，缓存结果以提高性能
@st.cache_data
def perform_topic_modeling_gensim(data, n_topics=5, stopwords=None):
    data = data.fillna('')

    vectorizer = CountVectorizer(preprocessor=lambda x: preprocess_text(x, stopwords), max_df=0.95, min_df=2,
                                 stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data)

    # 转换为gensim可用格式
    corpus = Sparse2Corpus(doc_term_matrix, documents_columns=False)
    id2word = Dictionary.from_corpus(corpus, id2word=dict(enumerate(vectorizer.get_feature_names_out())))

    lda = LdaModel(corpus=corpus, num_topics=n_topics, id2word=id2word, random_state=0)
    return lda, id2word, corpus


# 生成词云
@st.cache_data
def display_word_cloud(_lda, id2word):
    wordclouds = []
    for idx, topic in enumerate(_lda.get_topics()):
        word_freq = dict(zip(id2word.values(), topic))
        wordcloud = WordCloud(width=800, height=400, max_words=50,
                              font_path='/Users/liuhaoran/LHR/PycharmProjects/Comment analysis/LDA/Songti.ttc').generate_from_frequencies(
            word_freq)
        buf = io.BytesIO()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {idx + 1}')
        plt.savefig(buf, format="png")
        buf.seek(0)
        wordclouds.append(buf)
    return wordclouds


# 翻译LDA可视化文本为中文
def translate_html_to_chinese(html_content):
    translations = {
        "Topic": "主题", "Lambda": "λ 值", "Relevance": "关联度", "Overall term frequency": "整体词频",
        "Top 30 Most Salient Terms": "前30个最显著的词语", "Most relevant words for topic": "主题的相关词语",
    }
    for english, chinese in translations.items():
        html_content = html_content.replace(english, chinese)
    return html_content


# Streamlit应用
st.title("主题建模工具")

uploaded_file = st.file_uploader("上传数据表", type=["csv", "xlsx"])

# 允许用户输入自定义停用词
use_custom_stopwords = st.checkbox("启用自定义停用词")
custom_stopwords_input = st.text_area("输入自定义停用词，使用换行分隔每个词", "")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("数据预览：", df.head())

        stopwords = get_stopwords(custom_stopwords_input if use_custom_stopwords else None)

        selected_column = st.selectbox("选择用于分析的列", df.columns)
        if selected_column:
            df['关键词'] = df[selected_column].apply(extract_keywords)
            st.write("关键词提取结果：", df[[selected_column, '关键词']].head())

            n_topics = st.slider("选择主题数目", 2, 20, 5)
            lda_model, id2word, corpus = perform_topic_modeling_gensim(df[selected_column], n_topics=n_topics,
                                                                       stopwords=stopwords)

            st.write("LDA 模型可视化：")
            lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word)
            pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_vis_data)

            if st.button("保存 LDA 可视化结果"):
                translated_html = translate_html_to_chinese(pyLDAvis_html)
                with open('lda_visualization.html', 'w', encoding='utf-8') as f:
                    f.write(translated_html)
                st.success("LDA 可视化结果已保存为 'lda_visualization.html'")

            st.components.v1.html(pyLDAvis_html, width=1300, height=800, scrolling=True)

            st.write("主题词云：")
            wordclouds = display_word_cloud(lda_model, id2word)
            for buf in wordclouds:
                st.image(buf, use_column_width=True)
        else:
            st.error("请选择一个用于分析的列。")
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")