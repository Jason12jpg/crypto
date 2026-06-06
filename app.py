"""
app.py - Crypto AI 趋势预测（小白友好版）
"""
import streamlit as st
import pandas as pd
import os
import analyzer

st.set_page_config(page_title='Crypto AI 助手', page_icon='🤖', layout='wide')

# ============================================================
# CSS
# ============================================================
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
    50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.7); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.7); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.guide-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    line-height: 1.8;
    color: #d1d5db;
}
.guide-box h3 { color: #a78bfa; margin-top: 0; }
.guide-box .step { color: #10b981; font-weight: 700; }

/* ---- Signal Card ---- */
.signal-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: transform 0.3s;
}
.signal-card:hover { transform: translateY(-4px); }

.signal-card.buy {
    background: linear-gradient(145deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    animation: pulseGlow 2.5s infinite, float 3s infinite;
}
.signal-card.wait {
    background: linear-gradient(145deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444;
    animation: pulseRed 2.5s infinite, float 3s infinite;
}
.signal-card.small {
    animation: none;
    border-width: 1px;
    padding: 1.2rem;
}

.signal-action {
    font-size: 2.2rem;
    font-weight: 900;
    margin-bottom: 0.3rem;
}
.signal-action.buy { color: #34d399; }
.signal-action.wait { color: #f87171; }

.signal-coin {
    font-size: 1.6rem;
    font-weight: 900;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
}
.signal-conf {
    font-size: 1.2rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 0.4rem 0;
}
.signal-detail {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.5;
}
.signal-rank { font-size: 2rem; margin-bottom: 0.2rem; display: block; }

/* ---- Metric ---- */
.metric-box {
    background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.metric-box .label { color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.3rem; }
.metric-box .value { color: #e0e0e0; font-size: 2rem; font-weight: 900; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(26,26,46,0.8); border-radius: 12px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #9ca3af; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#7c3aed,#6d28d9); color: white !important; }
</style>
''', unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.title('🤖 Crypto AI 助手')
st.caption('AI 帮你分析，告诉你现在该买什么币')
st.divider()

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs(['🎯 现在该买什么', '📋 历史战绩', '❓ 新手指南'])

# ============================================================
# Tab 1: 现在该买什么
# ============================================================
with tabs[0]:

    st.markdown('''
    <div class="guide-box">
        <h3>📖 三步看懂</h3>
        <span class="step">①</span> 点下面的大按钮 →
        <span class="step">②</span> 看排名第一 🥇 的币 →
        <span class="step">③</span> 绿色 ✅ 可以买 ｜ 红色 ⛔ 先别碰
    </div>
    ''', unsafe_allow_html=True)

    if st.button('🔥 AI 帮我分析（点这里）', use_container_width=True, type='primary'):
        with st.spinner('🤖 8 个 AI 模型正在分析 10 个币种... 请稍等约 30 秒'):
            results = []
            for sym in analyzer.SYMBOLS:
                try:
                    r = analyzer.generate_prediction(sym)
                    results.append(r)
                except Exception:
                    pass

        if results:
            results.sort(key=lambda x: x['confidence'], reverse=True)
            medals = ['🥇', '🥈', '🥉', '④', '⑤']

            # P1 — Hero card
            p1 = results[0]
            is_buy = p1['direction'] == 'UP'
            action_text = '✅ 可以买入' if is_buy else '⛔ 先别碰'
            action_class = 'buy' if is_buy else 'wait'
            coin_name = p1['symbol'].replace('/USDT', '')
            agree = sum(1 for v in p1.get('model_votes', {}).values() if v['direction'] == p1['direction'])
            reason = f'8 个 AI 中有 {agree} 个认为会{"涨" if is_buy else "跌"}'

            st.markdown(f'''
            <div class="signal-card {action_class}">
                <span class="signal-rank">{medals[0]}</span>
                <div class="signal-action {action_class}">{action_text}</div>
                <div class="signal-coin">{coin_name}</div>
                <div class="signal-conf">AI 信心: {p1["confidence"]:.0f}%</div>
                <div class="signal-detail">
                    {'📈 AI 预测这个币下一小时会涨，现在买入可以赚差价' if is_buy else '📉 AI 预测这个币下一小时会跌，建议先观望不要买'}
                    <br>{reason}
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # P2-P5
            if len(results) > 1:
                st.markdown('#### 其他备选')
                cols = st.columns(min(len(results) - 1, 4))
                for i, r in enumerate(results[1:5]):
                    is_b = r['direction'] == 'UP'
                    act = '✅ 买' if is_b else '⛔ 等'
                    cls = 'buy' if is_b else 'wait'
                    cn = r['symbol'].replace('/USDT', '')
                    medal = medals[i + 1] if i + 1 < len(medals) else ''
                    ag = sum(1 for v in r.get('model_votes', {}).values() if v['direction'] == r['direction'])

                    cols[i].markdown(f'''
                    <div class="signal-card {cls} small" style="opacity: {0.85 - i*0.08};">
                        <span class="signal-rank" style="font-size:1.4rem;">{medal}</span>
                        <div class="signal-action {cls}" style="font-size:1.3rem;">{act}</div>
                        <div class="signal-coin" style="font-size:1.1rem;">{cn}</div>
                        <div class="signal-conf" style="font-size:0.9rem;">信心 {r["confidence"]:.0f}%</div>
                        <div class="signal-detail">{ag}/8 AI 同意</div>
                    </div>
                    ''', unsafe_allow_html=True)

            # 其余 P6-P10
            if len(results) > 5:
                with st.expander('📋 查看全部 10 个币'):
                    for r in results[5:]:
                        icon = '🟢' if r['direction'] == 'UP' else '🔴'
                        label = '可以买' if r['direction'] == 'UP' else '先别碰'
                        cn = r['symbol'].replace('/USDT', '')
                        st.write(f"{icon} **{cn}** — {label} (信心 {r['confidence']:.0f}%)")

            st.divider()
            st.markdown('''
            > **💡 小提示**: 只建议操作 🥇 排名第一的币。信心 > 65% 比较靠谱，< 55% 就先观望。
            > 每次用小额试水，千万别 all in！
            ''')
    else:
        # Show latest predictions from log if available
        if os.path.exists(analyzer.LOG_FILE):
            try:
                ldf = pd.read_csv(analyzer.LOG_FILE, dtype=str)
                latest_ts = ldf['timestamp'].max()
                latest = ldf[ldf['timestamp'] == latest_ts].copy()
                if not latest.empty:
                    st.info(f'⏰ 上次分析时间: **{latest_ts}**（点击上面按钮获取最新分析）')
                    for _, row in latest.iterrows():
                        icon = '🟢' if row.get('direction') == 'UP' else '🔴'
                        label = '可以买' if row.get('direction') == 'UP' else '先别碰'
                        cn = row['symbol'].replace('/USDT', '')
                        conf = row.get('confidence', '?')
                        status = row.get('result', '')
                        result_icon = ''
                        if status == 'HIT':
                            result_icon = ' → ✅ 猜对了！'
                        elif status == 'MISS':
                            result_icon = ' → ❌ 猜错了'
                        elif status == 'Pending':
                            result_icon = ' → ⏳ 等待验证'
                        st.write(f"{icon} **{cn}** — {label} (信心 {conf}%){result_icon}")
            except Exception:
                pass

# ============================================================
# Tab 2: 历史战绩
# ============================================================
with tabs[1]:
    st.markdown('### 📋 AI 的历史表现')

    rates = analyzer.get_win_rate()

    # Overall stats
    col1, col2, col3 = st.columns(3)
    with col1:
        rate_str = f"{rates['overall']:.0%}" if rates['total'] > 0 else '—'
        st.markdown(f'''
        <div class="metric-box">
            <div class="label">总猜对率</div>
            <div class="value">{rate_str}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-box">
            <div class="label">猜对次数</div>
            <div class="value">{rates["hits"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-box">
            <div class="label">总预测次数</div>
            <div class="value">{rates["total"]}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.divider()

    # Per-symbol stats
    if rates['total'] > 0:
        st.markdown('#### 各币种猜对率')
        sym_rows = []
        for sym, data in rates.get('symbols', {}).items():
            cn = sym.replace('/USDT', '')
            rate = f"{data['rate']:.0%}" if data['total'] > 0 else '—'
            sym_rows.append({'币种': cn, '猜对率': rate, '猜对': data['hits'], '总次数': data['total']})
        if sym_rows:
            st.dataframe(pd.DataFrame(sym_rows), use_container_width=True, hide_index=True)

    # History log
    st.markdown('#### 详细记录')
    if os.path.exists(analyzer.LOG_FILE):
        try:
            ldf = pd.read_csv(analyzer.LOG_FILE, dtype=str)
            filter_sym = st.selectbox('筛选币种', ['全部'] + [s.replace('/USDT','') for s in analyzer.SYMBOLS])
            if filter_sym != '全部':
                ldf = ldf[ldf['symbol'] == f'{filter_sym}/USDT']

            if not ldf.empty:
                display = ldf[['timestamp', 'symbol', 'direction', 'confidence', 'result']].copy()
                display.columns = ['时间', '币种', '方向', '信心%', '结果']
                display['币种'] = display['币种'].str.replace('/USDT', '', regex=False)
                display['方向'] = display['方向'].map({'UP': '🟢 买', 'DOWN': '🔴 等'})
                display['结果'] = display['结果'].map({'HIT': '✅ 猜对', 'MISS': '❌ 猜错', 'Pending': '⏳ 等待'}).fillna(display['结果'])
                display = display.sort_values('时间', ascending=False).reset_index(drop=True)
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info('暂无记录')
        except Exception:
            st.info('暂无记录')
    else:
        st.info('系统还没有开始预测。后台每 2 小时自动运行，请耐心等待。')

# ============================================================
# Tab 3: 新手指南
# ============================================================
with tabs[2]:
    st.markdown('''
    ### ❓ 常见问题

    ---

    #### 🤔 这是什么？
    这是一个 **AI 自动分析系统**。它用 **8 种不同的 AI 模型** 同时分析 10 个主流加密货币，
    告诉你哪个币最可能在下一小时**涨**或**跌**。

    ---

    #### 📊 怎么看结果？
    - **🟢 绿色 = 可以买**：AI 认为这个币下一小时会涨
    - **🔴 红色 = 先别碰**：AI 认为这个币下一小时会跌
    - **信心 %**：数字越高，AI 越有把握（建议 > 65% 才操作）
    - **🥇 排第一的**：是 AI 最有信心的推荐

    ---

    #### 🛒 怎么买？
    1. 打开你常用的交易所 App（比如 Gate.io、OKX）
    2. 搜索 AI 推荐的币种（比如 SUI）
    3. 用 USDT 买入
    4. 等 1~2 小时后，看涨了就卖出赚差价

    ---

    #### ⏰ 需要自己手动更新数据吗？
    **不需要！** 系统在 GitHub 上每 2 小时自动运行：
    - 自动拉取最新价格数据
    - 自动评估上一轮预测对不对
    - 自动调整 AI 模型权重（猜对加分，猜错扣分）
    - 自动生成下一轮预测

    你只需要打开这个页面，看结果就行！

    ---

    #### ⚠️ 重要提醒
    - 🎰 **这不是稳赚的！** AI 也会猜错，请用小额试水
    - 💰 **不要 all in！** 每次只用你能承受亏损的金额
    - 📈 **长期看趋势**：AI 会自我学习，样本越多越准
    - 🧪 **先观察几天**：看看 AI 的历史战绩再决定要不要跟

    ---

    #### 🤖 这 8 个 AI 模型是什么？

    | 模型 | 功能 |
    |------|------|
    | 趋势跟踪 | 看价格的长期走向 |
    | 动量分析 | 判断是否超买超卖 |
    | 波动率 | 看价格是否到了极端位置 |
    | MACD | 捕捉趋势转折信号 |
    | 成交量 | 分析资金流向 |
    | 随机森林 | 机器学习综合分析 |
    | XGBoost | 高级机器学习模型 |
    | LSTM | 深度学习时间序列预测 |
    ''')
