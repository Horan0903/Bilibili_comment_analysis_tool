# B站评论分析工具

一个基于 Streamlit 的 B站视频评论分析工具，支持多维度数据分析和可视化。

## 功能特点

- 📊 **数据来源灵活**
  - 支持本地 CSV 文件导入
  - 支持通过 B站 API 实时获取评论数据

- 🔍 **多维度分析**
  - 评论概览统计
  - 情感分析
  - 关键词提取与分析
  - LDA 主题建模
  - 用户互动数据分析

- 📈 **丰富的可视化**
  - 评论趋势图
  - 情感分布图
  - 关键词词云
  - 主题词云
  - 交互式 LDA 可视化
  - 用户互动热力图

- 🤖 **AI 分析**
  - 支持 OpenAI API 进行深度评论分析
  - 自定义分析维度

## 安装说明

1. 克隆项目

```
bash
git clone https://github.com/yourusername/bilibili-comment-analysis.git
cd bilibili-comment-analysis
```

2. 安装依赖

```
bash
pip install -r requirements.txt
```


2. 配置环境变量

```
bash
创建 .env 文件并添加 OpenAI API 密钥（可选，用于 AI 分析功能）
OPENAI_API_KEY=your_api_key_her
```



## 使用方法

1. 启动应用


2. 选择数据来源
   - 上传本地 CSV 文件
   - 或输入 B站视频 BV 号

3. 查看分析结果
   - 评论概览：查看基本统计信息和时间趋势
   - 情感分析：查看评论情感分布和典型评论
   - 关键词分析：查看热门关键词和 LDA 主题分析
   - 互动数据：查看点赞和回复数据分布
   - AI 分析：使用 AI 进行深度评论分析

## 项目结构

## 数据格式要求

如果使用本地 CSV 文件，需要包含以下列：
- 评论内容：评论的具体内容
- 用户名：发表评论的用户名
- 时间：评论发布时间
- 点赞数：评论获得的点赞数
- 回复数：评论获得的回复数
- 用户等级（可选）：发表评论的用户等级
- 会员类型（可选）：用户的会员状态
- 性别（可选）：用户性别信息

## 主要功能说明

### 评论概览
- 评论总数统计
- 点赞数和回复数分析
- 评论时间趋势分析
- 评论词云展示

### 情感分析
- 评论情感得分计算
- 情感分布可视化
- 典型积极/消极评论展示
- 情感趋势分析

### 关键词分析
- 热门关键词提取
- 关键词频率统计
- LDA 主题建模
- 主题词云展示
- 交互式主题可视化

### 互动数据
- 点赞数分布分析
- 回复数分布分析
- 高互动评论展示
- 用户活跃度分析

### AI 分析
- 基于 OpenAI API 的深度分析
- 自定义关键词相关性分析
- 评论分类与聚类
- 分析结果可视化

## 依赖项

主要依赖包括：
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.18.0
- jieba >= 0.42.1
- gensim >= 4.0.0
- pyLDAvis >= 3.3.0
- openai >= 1.3.0
- wordcloud >= 1.9.0
- matplotlib >= 3.7.0
- snownlp >= 0.12.3

详细依赖列表请查看 `requirements.txt`

## 注意事项

- Python 版本要求：Python 3.8 或更高版本
- 确保所有必要的依赖包都已正确安装
- 使用 AI 分析功能需要有效的 OpenAI API 密钥
- LDA 分析需要足够的评论数据才能得到好的效果
- 首次运行可能需要下载一些必要的资源文件
- 建议使用虚拟环境来管理项目依赖

## 常见问题

1. OpenAI API 相关问题
   - 确保已正确设置 API 密钥
   - 检查网络连接是否正常
   - 注意 API 调用限制

2. 数据加载问题
   - 确保 CSV 文件格式正确
   - 检查文件编码（推荐使用 UTF-8）
   - 验证必要的列名是否存在

3. LDA 分析相关问题
   - 确保停用词文件和词典文件存在
   - 调整主题数量获得更好的效果
   - 注意数据量的充足性

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。贡献时请注意：
- 遵循现有的代码风格
- 添加必要的注释和文档
- 确保代码通过测试
- 更新相关文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[your-email@example.com]

## 致谢

感谢以下开源项目的支持：
- Streamlit
- Gensim
- pyLDAvis
- Jieba
- SnowNLP
- 以及其他所有使用到的开源库

## 更新日志

### v1.0.0 (2024-01)
- 初始版本发布
- 实现基本的评论分析功能
- 添加 LDA 主题建模
- 集成 AI 分析功能