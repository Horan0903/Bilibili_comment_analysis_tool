import streamlit as st
from openai import OpenAI
import pandas as pd
import re

# 设置 Streamlit  标题
st.title("评论分析工具")

# 输入 API 密钥和 URL
api_key = st.text_input("API 密钥", type="password")
base_url = st.text_input("Base URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 上传 CSV 文件
uploaded_file = st.file_uploader("上传评论数据表", type="csv")

# 输入模型名称
model_name = st.text_input("模型名称", "qwen-turbo")

# 输入系统提示语
system_prompt = st.text_area("系统提示语", "You are a helpful assistant.")

# 输入用户提示语的模板
user_prompt_template = st.text_area("用户提示语模板",
                                    "以下内容出自网络视频评论区。我是这个视频的作者，本期的视觉设计是：数据分析、数据、可视化、八爪鱼、数据建模，"
                                    "期望了解观众是否有在关注视频中的画面信息（而不是关注配音或者口播内容）。请你帮我分类每一条评论是否与画面信息相关。"
                                    "只需回答‘是’or‘否’。中括号包裹的是表情包，可以忽略。\n\n评论：{comment}\n分类：")

# 输入其他参数
temperature = st.slider("Temperature", 0.0, 1.0, 0.8)
top_p = st.slider("Top P", 0.0, 1.0, 0.8)
max_comment_length = st.number_input("最大评论长度", value=1000, step=1)

# 输入输出文件名
output_filename = st.text_input("输出文件名", "classified_comments_with_likes.csv")

if uploaded_file is not None:
    # 读取上传的 CSV 文件1
    st.write("正在读取上传的 CSV 文件...")
    data = pd.read_csv(uploaded_file)

    # 让用户选择分析所在的列
    comment_column = st.selectbox("选择评论内容所在列", data.columns)
    likes_column = st.selectbox("选择点赞数所在列", data.columns)

    data[comment_column] = data[comment_column].fillna('')  # 用空字符串填充缺失值
    data[comment_column] = data[comment_column].astype(str)  # 转换为字符串
    comments = data[comment_column].tolist()  # 假设评论列名为用户选择的列
    likes = data[likes_column].tolist()  # 假设点赞数列名为用户选择的列

    st.write("CSV 文件读取完毕，预览数据：")
    st.dataframe(data.head())  # 显示前五行数据

    # 定义预处理函数
    def preprocess_comment(comment):
        if not isinstance(comment, str):
            comment = str(comment)
        comment = re.sub(r'[^\w\s,.:?!]', '', comment)  # 移除除了字母、数字、空格和基本标点符号外的所有字符
        comment = comment.strip()  # 去除前后空格
        return comment[:max_comment_length]

    # 定义分析函数
    def analyze_comment(client, comment):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt_template.format(comment=comment)}
        ]
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )
            classification = completion.choices[0].message.content.strip()
        except Exception as e:
            error_message = str(e)
            if "data_inspection_failed" in error_message:
                classification = "不适当内容"
            else:
                st.error(f"分析评论时出错: {e}")
                classification = "无法分类"
        return classification

    @st.cache_resource
    def init_client(api_key, base_url):
        return OpenAI(api_key=api_key, base_url=base_url)

    client = init_client(api_key, base_url)

    if st.button("运行分析"):
        if api_key and base_url:
            st.write("初始化 OpenAI 客户端...")

            # 处理评论并显示进度条
            classifications = []
            log_window = st.empty()  # 创建一个占位符窗口
            st.write("开始处理评论...")
            progress_bar = st.progress(0)

            for i, comment in enumerate(comments):
                if not comment.strip() or comment == ',,,,':  # 如果评论为空或仅包含逗号，则跳过
                    log_window.text(f"跳过空评论 {i + 1}/{len(comments)}")
                    classifications.append("未处理")
                    continue

                log_window.text(f"正在预处理评论 {i + 1}/{len(comments)}: {comment[:50]}...")
                clean_comment = preprocess_comment(comment)
                log_window.text(f"正在分析评论 {i + 1}/{len(comments)}: {clean_comment[:50]}...")
                classification = analyze_comment(client, clean_comment)
                log_window.text(f"评论 {i + 1}/{len(comments)} 的分类结果: {classification}")
                classifications.append(classification)
                progress_bar.progress((i + 1) / len(comments))

            st.write("评论处理完成，正在保存分类结果...")

            # 将分类结果添加到数据表中
            data['classification'] = classifications

            # 保存分类结果到新的数据表
            data.to_csv(output_filename, index=False)
            st.success(f"分类结果已保存到: {output_filename}")

            st.write("分类结果预览：")
            st.dataframe(data.head())  # 显示前五行分类结果数据

            # 筛选视觉类评论
            st.write("正在计算视觉类评论的加权占比...")
            visual_comments = data[
                (data['classification'] == '是') & (data[comment_column].str.strip() != '') & (data[comment_column] != ',,,,')]

            # 计算视觉类评论总点赞数
            visual_likes = visual_comments[likes_column].sum()

            # 计算所有评论的总点赞数
            filtered_data = data[(data[comment_column].str.strip() != '') & (data[comment_column] != ',,,,')]
            total_likes = filtered_data[likes_column].sum()

            # 计算视觉类评论加权占比
            weighted_visual_ratio = visual_likes / total_likes if total_likes != 0 else 0

            # 计算详细数据
            total_comments = len(filtered_data)
            visual_comments_count = len(visual_comments)

            # 显示详细数据
            st.write(f"总评论数: {total_comments}")
            st.write(f"视觉类评论数: {visual_comments_count}")
            st.write(f"总点赞数: {total_likes}")
            st.write(f"视觉类评论点赞数: {visual_likes}")
            st.write(f"视觉类评论加权占比: {weighted_visual_ratio:.2%}")

        else:
            st.error("请提供 API 密钥和 Base URL")
