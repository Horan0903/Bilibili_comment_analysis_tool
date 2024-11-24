import pandas as pd
import requests
import time
from datetime import datetime
import streamlit as st
import asyncio
import aiohttp
import json
import sys
import re

def extract_bvid(url):
    """从URL或BV号中提取BVID"""
    # BV号直接匹配模式
    bv_pattern = r'BV[a-zA-Z0-9]+'
    match = re.search(bv_pattern, url)
    if match:
        return match.group()
    return None

async def fetch_comments(session, url, params):
    """异步获取评论数据"""
    try:
        async with session.get(url, params=params) as response:
            return await response.json()
    except Exception as e:
        st.error(f"请求失败: {str(e)}")
        return None

async def load_data_from_api(input_text, max_comments=None):
    """从B站API异步加载评论数据"""
    # 提取BV号
    bv_id = extract_bvid(input_text)
    if not bv_id:
        raise ValueError("无效的BV号或链接")

    # B站API基础URL
    api_base = "https://api.bilibili.com"
    view_url = f"{api_base}/x/web-interface/view"
    reply_url = f"{api_base}/x/v2/reply/main"
    reply_detail_url = f"{api_base}/x/v2/reply/reply"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com',
        'Cookie': "buvid3=2E00D3C2-0FEF-0BA6-1234-5678901234567890infoc"
    }

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            # 1. 获取视频基本信息
            params = {'bvid': bv_id}
            response = await fetch_comments(session, view_url, params)
            
            if not response or response['code'] != 0:
                raise ValueError(f"获取视频信息失败: {response.get('message', '未知错误')}")
            
            video_info = response['data']
            aid = video_info['aid']
            
            # 显示视频基本信息
            st.info(f"视频信息:\n"
                   f"标题: {video_info['title']}\n"
                   f"UP主: {video_info['owner']['name']}\n"
                   f"播放量: {video_info['stat']['view']:,}\n"
                   f"点赞数: {video_info['stat']['like']:,}\n"
                   f"投币数: {video_info['stat']['coin']:,}\n"
                   f"收藏数: {video_info['stat']['favorite']:,}\n"
                   f"弹幕数: {video_info['stat']['danmaku']:,}\n"
                   f"评论数: {video_info['stat']['reply']:,}")
            
            # 2. 获取评论
            comments_data = []
            page = 1
            page_size = 20  # 固定每页20条评论
            has_more = True
            total_count = video_info['stat']['reply']
            
            # 如果设置了最大评论数，则调整total_count
            if max_comments is not None:
                total_count = min(total_count, max_comments)
                st.info(f"将获取前 {total_count} 条评论")
            
            # 创建进度条占位符
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            while has_more:
                # 检查是否达到最大评论数
                if max_comments is not None and len(comments_data) >= max_comments:
                    break
                
                # 主评论请求参数
                params = {
                    'type': 1,
                    'oid': aid,
                    'pn': page,  # 页码
                    'ps': page_size,  # 每页数量固定为20
                    'sort': 2  # 按时间排序：2是时间倒序，3是热度排序
                }
                
                # 显示当前页码
                progress_text.text(f'正在获取第 {page} 页评论...')
                
                data = await fetch_comments(session, reply_url, params)
                
                if not data or data['code'] != 0:
                    st.warning(f"获取第 {page} 页评论失败: {data.get('message', '未知错误')}")
                    break
                
                replies = data['data'].get('replies', [])
                if not replies:
                    break
                
                # 处理评论数据
                for reply in replies:
                    # 检查是否达到最大评论数
                    if max_comments is not None and len(comments_data) >= max_comments:
                        has_more = False
                        break
                        
                    comment = {
                        '评论内容': reply['content']['message'],
                        '用户名': reply['member']['uname'],
                        '时间': datetime.fromtimestamp(reply['ctime']),
                        '点赞数': reply['like'],
                        '回复数': reply['count'],
                        '用户等级': reply['member']['level_info']['current_level'],
                        '性别': reply['member']['sex'],
                        '会员类型': reply['member']['vip']['vipType'],
                        'IP属地': reply.get('reply_control', {}).get('location', '未知')
                    }
                    comments_data.append(comment)
                    
                    # 获取评论的回复
                    if reply['count'] > 0 and (max_comments is None or len(comments_data) < max_comments):
                        reply_params = {
                            'type': 1,
                            'oid': aid,
                            'root': reply['rpid'],
                            'ps': min(10, max_comments - len(comments_data) if max_comments else 10)
                        }
                        
                        reply_data = await fetch_comments(session, reply_detail_url, reply_params)
                        if reply_data and reply_data['code'] == 0:
                            for sub_reply in reply_data['data'].get('replies', []):
                                if max_comments is not None and len(comments_data) >= max_comments:
                                    break
                                    
                                sub_comment = {
                                    '评论内容': sub_reply['content']['message'],
                                    '用户名': sub_reply['member']['uname'],
                                    '时间': datetime.fromtimestamp(sub_reply['ctime']),
                                    '点赞数': sub_reply['like'],
                                    '回复数': 0,
                                    '用户等级': sub_reply['member']['level_info']['current_level'],
                                    '性别': sub_reply['member']['sex'],
                                    '会员类型': sub_reply['member']['vip']['vipType'],
                                    'IP属地': sub_reply.get('reply_control', {}).get('location', '未知')
                                }
                                comments_data.append(sub_comment)
                
                # 更新进度
                progress = min(len(comments_data) / total_count, 1.0)
                progress_bar.progress(progress)
                progress_text.text(f'已获取 {len(comments_data)} 条评论 (第 {page} 页, 进度: {int(progress * 100)}%)')
                
                # 检查是否还有更多页
                if len(replies) < page_size:
                    has_more = False
                else:
                    page += 1
                    # 添加固定延迟
                    await asyncio.sleep(2)  # 每页之间延迟2秒
                    
                    # 每获取5页后增加额外迟
                    if page % 5 == 0:
                        await asyncio.sleep(3)  # 每5页后额外延迟3秒
                        st.info(f"已获取 {page} 页，短暂暂停以避免请求过快...")
    
    except Exception as e:
        st.error(f"获取评论数据时出错: {str(e)}")
        return pd.DataFrame()
    finally:
        # 清理进度显示
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'progress_text' in locals():
            progress_text.empty()
    
    if not comments_data:
        return pd.DataFrame(columns=['评论内容', '用户名', '时间', '点赞数', '回复数', 
                                   '用户等级', '性别', '会员类型', 'IP属地'])
    
    # 转换为DataFrame并返回
    df = pd.DataFrame(comments_data)
    
    # 添加统计信息
    st.success(f"成功获取 {len(df)} 条评论")
    
    return df

def load_bilibili_comments(input_text, max_comments=None):
    """包装异步函数以供同步调用"""
    try:
        # 在Windows系统上可能需要设置事件循环策略
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 运行异步函数
        return asyncio.run(load_data_from_api(input_text, max_comments))
    except Exception as e:
        st.error(f"加载评论失败: {str(e)}")
        return pd.DataFrame()

def load_data_from_csv(file):
    """从CSV或Excel文件加载评论数据"""
    # 获取文件扩展名
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        # 首先加载原始数据
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")
        
        # 使用会话状态保存列映射
        if 'column_mapping_confirmed' not in st.session_state:
            st.session_state['column_mapping_confirmed'] = False
        
        if not st.session_state['column_mapping_confirmed']:
            # 显示列映射选择器
            st.write("请选择对应的列名映射：")
            
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                content_col = st.selectbox(
                    "评论内容列", 
                    options=df.columns,
                    help="包含评论文本的列"
                )
                
                username_col = st.selectbox(
                    "用户名列", 
                    options=df.columns,
                    help="包含用户名的列"
                )
                
                time_col = st.selectbox(
                    "时间列", 
                    options=df.columns,
                    help="包含评论时间的列"
                )
                
            with col2:
                likes_col = st.selectbox(
                    "点赞数列", 
                    options=['无'] + list(df.columns),
                    help="包含点赞数的列，如果没有可选择'无'"
                )
                
                replies_col = st.selectbox(
                    "回复数列", 
                    options=['无'] + list(df.columns),
                    help="包含回复数的列，如果没有可选择'无'"
                )
            
            # 确认按钮
            if st.button("确认列映射"):
                # 保存列映射到���话状态
                st.session_state['column_mapping'] = {
                    'content_col': content_col,
                    'username_col': username_col,
                    'time_col': time_col,
                    'likes_col': likes_col,
                    'replies_col': replies_col
                }
                st.session_state['column_mapping_confirmed'] = True
                st.rerun()  # 使用新的 st.rerun() 替代 st.experimental_rerun()
        
        # 如果列映射已确认，使用映射加载数据
        if st.session_state['column_mapping_confirmed']:
            mapping = st.session_state['column_mapping']
            new_df = pd.DataFrame()
            
            # 复制必要的列
            new_df['评论内容'] = df[mapping['content_col']]
            new_df['用户名'] = df[mapping['username_col']]
            new_df['时间'] = pd.to_datetime(df[mapping['time_col']])
            
            # 处理可选列
            new_df['点赞数'] = df[mapping['likes_col']] if mapping['likes_col'] != '无' else 0
            new_df['回复数'] = df[mapping['replies_col']] if mapping['replies_col'] != '无' else 0
            
            # 确保数值列为数值类型
            new_df['点赞数'] = pd.to_numeric(new_df['点赞数'], errors='coerce').fillna(0)
            new_df['回复数'] = pd.to_numeric(new_df['回复数'], errors='coerce').fillna(0)
            
            return new_df
        
        return None
        
    except Exception as e:
        raise Exception(f"加载文件时出错: {str(e)}")