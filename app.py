import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import io

# --- 页面配置 ---
st.set_page_config(page_title="基金全指标分析器", page_icon="📈", layout="wide")

# --- 核心计算函数 (从你之前的脚本提取并优化) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_data(df, date_col, price_col):
    # 数据预处理
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col])
    
    prices = df[price_col]
    dates = df[date_col]
    
    # 1. 基础涨跌幅
    df['当前涨跌幅(%)'] = prices.pct_change() * 100
    
    # 2. 周期涨幅
    df['20日涨幅(%)'] = (prices / prices.shift(20) - 1) * 100
    df['60日涨幅(%)'] = (prices / prices.shift(60) - 1) * 100
    
    # 3. 波动率
    daily_ret = prices.pct_change()
    df['20日波动率(%)'] = daily_ret.rolling(window=20).std() * 100
    
    # 4. 动量比率
    momentum = df['20日涨幅(%)'] / 100
    volatility = df['20日波动率(%)'] / 100
    df['动量-波动率比率(20日)'] = np.where(volatility != 0, momentum / volatility, 0)
    
    # 5. RSI
    df['RSI(14)'] = calculate_rsi(prices, 14)
    
    # 6. 状态
    def get_status(rsi):
        if pd.isna(rsi): return "数据不足"
        if rsi > 70: return "超买"
        if rsi < 30: return "超卖"
        return "正常"
    df['状态'] = df['RSI(14)'].apply(get_status)
    
    # 7. 价格百分位
    df['价格百分位'] = (prices.expanding().rank(pct=True) * 100).round(2)
    
    # 8. YTD
    df['year'] = dates.dt.year
    first_in_year = df.groupby('year').head(1).index
    df['ytd_start'] = np.nan
    df.loc[first_in_year, 'ytd_start'] = prices.loc[first_in_year]
    df['ytd_start'] = df['ytd_start'].ffill()
    df['YTD涨幅(%)'] = (prices / df['ytd_start'] - 1) * 100
    df.drop(columns=['year', 'ytd_start'], inplace=True)
    
    # 9. 202409TD
    target_date = pd.Timestamp('2024-09-30')
    base_mask = dates >= target_date
    if base_mask.any():
        base_idx = base_mask.argmax()
        base_price_val = prices.iloc[base_idx]
        base_series = pd.Series(np.nan, index=df.index)
        base_series.iloc[base_idx:] = base_price_val
        df['202409TD涨幅(%)'] = (prices / base_series - 1) * 100
    else:
        df['202409TD涨幅(%)'] = np.nan
        
    return df

def plot_chart(df, date_col, price_col):
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 图1: 净值
    ax[0].plot(df[date_col], df[price_col], label='净值', color='#1f77b4', linewidth=1.5)
    ax[0].set_title('净值走势', fontsize=14, fontweight='bold')
    ax[0].legend(loc='upper left')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    
    # 图2: RSI
    ax[1].plot(df[date_col], df['RSI(14)'], label='RSI(14)', color='purple', linewidth=1.5)
    ax[1].axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax[1].axhline(30, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax[1].fill_between(df[date_col], 70, 100, color='red', alpha=0.1)
    ax[1].fill_between(df[date_col], 0, 30, color='green', alpha=0.1)
    ax[1].set_ylim(0, 100)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    # 图3: 20日涨幅
    ax[2].plot(df[date_col], df['20日涨幅(%)'], label='20日涨幅%', color='orange', linewidth=1.5)
    ax[2].axhline(0, color='black', linewidth=0.5)
    ax[2].legend(loc='upper right')
    ax[2].grid(True, linestyle='--', alpha=0.6)
    
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# --- 主界面逻辑 ---
st.title("📊 基金全历史指标分析器 (Web版)")
st.markdown("上传基金净值 CSV/Excel 文件，自动计算 RSI、波动率、动量比率等指标并生成图表。")

# 侧边栏
with st.sidebar:
    st.header("📂 文件上传")
    uploaded_file = st.file_uploader("选择文件", type=['csv', 'xlsx', 'xls'])
    
    st.info("💡 提示：文件需包含'日期'和'净值'(或'收盘')列。")

if uploaded_file is not None:
    try:
        # 1. 读取文件
        with st.spinner('正在读取数据...'):
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        
        # 2. 自动识别列
        date_col = None
        price_col = None
        for col in df_raw.columns:
            col_str = str(col).lower()
            if '日期' in col_str or 'date' in col_str:
                date_col = col
            if any(k in col_str for k in ['净值', '收盘', 'price', 'nav', '累计']):
                price_col = col
        
        if not date_col or not price_col:
            st.error(f"❌ 无法自动识别列。检测到的列名：{list(df_raw.columns)}")
            st.stop()
            
        st.success(f"✅ 识别成功：日期列='{date_col}', 价格列='{price_col}'")
        
        # 3. 执行计算
        with st.spinner('正在计算每日指标 (RSI, 波动率, 涨幅...) ...'):
            df_result = analyze_data(df_raw.copy(), date_col, price_col)
        
        # 4. 展示最新数据摘要
        st.subheader("📈 最新一日数据摘要")
        latest = df_result.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("当前价格", f"{latest[price_col]:.4f}")
        col2.metric("RSI(14)", f"{latest['RSI(14)']:.2f}", delta_color="inverse")
        col3.metric("状态", latest['状态'])
        col4.metric("20日涨幅", f"{latest['20日涨幅(%)']:.2f}%")
        col5.metric("价格百分位", f"{latest['价格百分位']:.1f}%")
        
        # 5. 展示图表
        st.subheader("📉 技术走势分析")
        fig = plot_chart(df_result, date_col, price_col)
        st.pyplot(fig)
        
        # 6. 数据表格与下载
        with st.expander("📋 查看完整历史数据表"):
            st.dataframe(df_result.round(4), use_container_width=True)
            
        # 7. 下载按钮
        st.subheader("💾 导出结果")
        
        # 准备 Excel 文件到内存
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_result.round(4).to_excel(writer, sheet_name='每日完整数据', index=False)
            df_result.iloc[-1:].to_excel(writer, sheet_name='最新摘要', index=False)
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 下载 Excel 分析报告",
            data=processed_data,
            file_name=f"Fund_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        st.exception(e)
else:
    st.info("👆 请在左侧上传文件开始分析。")
    # 展示示例数据结构
    st.markdown("""
    ### 文件格式要求
    支持 **CSV** 或 **Excel** 文件，必须包含以下两列（列名可模糊匹配）：
    - **日期列**：包含 '日期' 或 'Date'
    - **价格列**：包含 '净值', '收盘', 'Price', 'NAV' 或 '累计'
    
    **示例数据前几行：**
    | 日期 | 累计净值 |
    | :--- | :--- |
    | 2023-01-01 | 1.2345 |
    | 2023-01-02 | 1.2350 |
    """)