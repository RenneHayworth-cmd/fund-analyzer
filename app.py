import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

# --- 页面配置 ---
st.set_page_config(page_title="基金智能评级分析器 (Pro)", page_icon="🚀", layout="wide")

# --- 核心计算函数 ---

def calculate_rsi(series, period=14):
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_rating(rsi, ratio):
    """
    【核心逻辑】综合 RSI 和 动量-波动率比率 进行评级
    特别优化：严防超卖区的"接飞刀"风险
    """
    if pd.isna(rsi) or pd.isna(ratio):
        return "数据不足", "数据尚不充分", "#808080"
    
    # --- 1. 超卖区域判断 (RSI < 30) ---
    if rsi < 30:
        if ratio > 0:
            return "E+ (黄金坑)", "超卖且动能已转正，极佳抄底点", "#9370DB"
        elif ratio > -0.2:
            return "E (企稳观察)", "超卖且跌势放缓，可小仓试错", "#1E90FF"
        elif ratio > -0.5:
            return "E- (阴跌中)", "超卖但仍在持续阴跌，谨慎观望", "#FFA500"
        else:
            return "E-- (接飞刀)", "超卖且剧烈暴跌，切勿伸手！", "#8B0000"
    
    # --- 2. 超买区域判断 (RSI > 70) ---
    if rsi > 70:
        if ratio > 1.0:
            return "C (疯狂逼空)", "超买但走势极强，可能继续疯涨", "#FFA500"
        else:
            return "B (高风险)", "超买且波动剧烈，警惕诱多/回调", "#FF4444"
    
    # --- 3. 正常区域判断 (30 <= RSI <= 70) ---
    if ratio > 1.0:
        return "S (完美主升)", "低波稳健大涨，最佳持有状态", "#00CC00"
    elif ratio > 0.5:
        return "A (稳健上涨)", "趋势健康，安心持有", "#90EE90"
    elif ratio > 0:
        return "D (震荡整理)", "上涨乏力或波动过大，观望", "#808080"
    else:
        return "D- (弱势调整)", "负动量，趋势向下", "#D3D3D3"

def analyze_data(df, date_col, price_col):
    """执行所有指标计算"""
    # 数据预处理
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col])
    
    prices = df[price_col]
    dates = df[date_col]
    
    # 【修复点】显式创建 '当前价格' 列，方便后续统一调用
    df['当前价格'] = prices
    
    # 1. 基础涨跌幅
    df['当前涨跌幅(%)'] = prices.pct_change() * 100
    
    # 2. 周期涨幅
    df['20日涨幅(%)'] = (prices / prices.shift(20) - 1) * 100
    df['60日涨幅(%)'] = (prices / prices.shift(60) - 1) * 100
    
    # 3. 波动率 (未年化)
    daily_ret = prices.pct_change()
    df['20日波动率(%)'] = daily_ret.rolling(window=20).std() * 100
    
    # 4. 动量 - 波动率比率
    momentum = df['20日涨幅(%)'] / 100
    volatility = df['20日波动率(%)'] / 100
    df['动量-波动率比率'] = np.where(volatility != 0, momentum / volatility, 0)
    
    # 5. RSI
    df['RSI(14)'] = calculate_rsi(prices, 14)
    
    # 6. 价格百分位
    df['价格百分位'] = (prices.expanding().rank(pct=True) * 100).round(2)
    
    # 7. YTD
    df['year'] = dates.dt.year
    first_in_year = df.groupby('year').head(1).index
    df['ytd_start'] = np.nan
    df.loc[first_in_year, 'ytd_start'] = prices.loc[first_in_year]
    df['ytd_start'] = df['ytd_start'].ffill()
    df['YTD涨幅(%)'] = (prices / df['ytd_start'] - 1) * 100
    df.drop(columns=['year', 'ytd_start'], inplace=True)
    
    # 8. 202409TD
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
        
    # 9. 应用综合评级
    ratings = df.apply(lambda row: calculate_rating(row['RSI(14)'], row['动量-波动率比率']), axis=1)
    df['综合评级'] = [r[0] for r in ratings]
    df['评级描述'] = [r[1] for r in ratings]
    df['评级颜色'] = [r[2] for r in ratings]
    
    return df

def plot_chart(df, date_col, price_col):
    """绘制三图组合"""
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 图1: 净值 (使用统一的 '当前价格' 列)
    ax[0].plot(df[date_col], df['当前价格'], label='净值', color='#1f77b4', linewidth=1.5)
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
    
    # 图3: 动量 - 波动率比率
    ax[2].plot(df[date_col], df['动量-波动率比率'], label='动量-波动率比率', color='orange', linewidth=1.5)
    ax[2].axhline(1.0, color='green', linestyle=':', linewidth=1, label='优秀阈值 (1.0)')
    ax[2].axhline(0.5, color='lightgreen', linestyle=':', linewidth=1, label='良好阈值 (0.5)')
    ax[2].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax[2].axhline(-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='危险阈值 (-0.5)')
    ax[2].legend(loc='upper right')
    ax[2].grid(True, linestyle='--', alpha=0.6)
    ax[2].set_ylabel('Ratio')
    
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# --- 主界面逻辑 ---

st.title("🚀 基金智能评级分析器 (Pro 版)")
st.markdown("""
基于 **RSI (位置)** + **动量-波动率比率 (质量)** 的双重维度综合评级系统。
特别优化：**严防超卖区“接飞刀”风险**。
""")

with st.sidebar:
    st.header("📂 文件上传")
    uploaded_file = st.file_uploader("选择 CSV/Excel 文件", type=['csv', 'xlsx', 'xls'])
    
    st.markdown("---")
    st.info("""
    ### 📊 评级标准说明
    
    #### 🟢 买入/持有区
    - **S 级**: 完美主升浪
    - **A 级**: 稳健上涨
    - **E+ 级**: 黄金坑
    
    #### ⚠️ 观察/警示区
    - **C 级**: 疯狂逼空
    - **E 级**: 超卖企稳
    - **D 级**: 震荡整理
    
    #### 🔴 卖出/禁止区
    - **B 级**: 高风险
    - **E- 级**: 阴跌中
    - **E-- 级**: 接飞刀 (**严禁抄底**)
    """)

if uploaded_file is not None:
    try:
        with st.spinner('正在读取数据...'):
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        
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
            
        with st.spinner('正在计算综合评级...'):
            df_result = analyze_data(df_raw.copy(), date_col, price_col)
        
        # 展示最新评级卡片
        st.subheader("🏆 当前综合评级")
        latest = df_result.iloc[-1]
        color_hex = latest['评级颜色']
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color_hex}20; border-left: 5px solid {color_hex}; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; color: {color_hex}; font-size: 32px;">🚩 {latest['综合评级']}</h2>
            <p style="font-size: 20px; margin: 10px 0 0 0; color: #333; font-weight: bold;">{latest['评级描述']}</p>
            <p style="font-size: 14px; color: #666; margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px;">
                <strong>RSI:</strong> {latest['RSI(14)']:.2f} &nbsp;|&nbsp; 
                <strong>动量比率:</strong> {latest['动量-波动率比率']:.3f} &nbsp;|&nbsp; 
                <strong>当前价格:</strong> {latest['当前价格']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI(14)", f"{latest['RSI(14)']:.2f}")
        c2.metric("动量-波动率比率", f"{latest['动量-波动率比率']:.3f}")
        c3.metric("20日涨幅", f"{latest['20日涨幅(%)']:.2f}%")
        c4.metric("价格百分位", f"{latest['价格百分位']:.1f}%")
        
        st.subheader("📉 技术走势与评级历史")
        fig = plot_chart(df_result, date_col, price_col)
        st.pyplot(fig)
        
        with st.expander("📋 查看完整历史数据 (含每日评级)"):
            # 【修复点】现在 '当前价格' 列肯定存在了
            display_df = df_result[['日期', '当前价格', '综合评级', '评级描述', 'RSI(14)', '动量-波动率比率', '20日涨幅(%)']].copy()
            display_df['日期'] = display_df['日期'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)
            
        st.subheader("💾 导出结果")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_result.round(4).to_excel(writer, sheet_name='每日完整数据', index=False)
            latest_df = df_result.iloc[-1:].copy()
            latest_df.to_excel(writer, sheet_name='最新摘要', index=False)
            
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 下载 Excel 分析报告",
            data=processed_data,
            file_name=f"Fund_Rating_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        st.exception(e)
else:
    st.info("👆 请在左侧上传基金净值文件开始智能评级。")
