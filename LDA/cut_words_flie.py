#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:51:48 2017
@author: Ming JIN
"""

import jieba
import pandas as pd

# 加载自定义词典
jieba.load_userdict("SogouLabDic.txt")
jieba.load_userdict("dict_baidu_utf8.txt")
jieba.load_userdict("dict_pangu.txt")
jieba.load_userdict("dict_sougou_utf8.txt")
jieba.load_userdict("dict_tencent_utf8.txt")
jieba.load_userdict("my_dict.txt")

# 加载停用词
stopwords = {}.fromkeys([line.rstrip() for line in open('Stopword.txt')])

def process_comments(file_path):
    # 读取表格数据
    data = pd.read_csv(file_path)  # 如果是Excel文件，用 pd.read_excel(file_path)

    print("正在解析数据...")

    for _, row in data.iterrows():
        result = []
        comment = row['评论内容']  # 假设表格中有一列名为 'comment'

        seg = jieba.cut(comment)

        for i in seg:
            if i not in stopwords:
                result.append(i)

        with open("data_full.dat", "a+", encoding='utf-8') as fo:
            fo.write(' '.join(result) + '\n')

    print("解析完成!")


if __name__ == '__main__':
    file_path = '/Users/liuhaoran/LHR/PycharmProjects/Weibo-Analyst/date/BV1Ui421h7v7_玉渊谭天_144秒看如何嗑成一个美国“紫薯”_评论.csv'  # 这里填写上传的表格文件路径
    print("进程开始...")
    process_comments(file_path)
    print("Done!")
