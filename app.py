"""
app.py - Crypto AI 趋势预测中枢 (Streamlit UI)
4 个标签页: 控制台 / 趋势交叉压榨 / 模型战力 / 实盘记录
"""
import streamlit as st
import pandas as pd
import os
import analyzer

st.set_page_config(page_title='Crypto AI 趋势预测', page_icon='🤖', layout='wide')

# ============================================================
# CSS Injection
# ============================================================
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(124, 58, 237, 0.4); }
    50% { box-shadow: 0 0 35px rgba(124, 58, 237, 0.9); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
@keyframes arrowBounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h2 { color: #a78bfa; margin: 0; }
.hero p { color: #9ca3af; margin: 0.5rem 0 0; }

.pred-card {
    background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.pred-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3);
}

.pred-card.p1 {
    animation: pulseGlow 2.5s infinite ease-in-out, float 3s infinite ease-in-out;
    border: 2px solid #7c3aed;
    background: linear-gradient(145deg, #1a1040, #2d1b69);
}

.pred-card.up { border-left: 4px solid #10b981; }
.pred-card.down { border-left: 4px solid #ef4444; }

.pred-symbol {
    font-size: 1.4rem;
    font-weight: 900;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
}
.pred-dir {
    font-size: 2rem;
    font-weight: 900;
    animation: arrowBounce 1.5s infinite;
}
.pred-dir.up { color: #10b981; }
.pred-dir.down { color: #ef4444; }
.pred-conf {
    font-size: 1.1rem;
    color: #a78bfa;
    font-weight: 700;
    margin: 0.5rem 0;
}
.pred-models {
    font-size: 0.75rem;
    color: #6b7280;
    line-height: 1.4;
}

.medal { font-size: 1.8rem; margin-bottom: 0.3rem; display: block; }

.metric-card {
    background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-card h3 { color: #a78bfa; margin: 0 0 0.3rem; font-size: 0.9rem; }
.metric-card .value { color: #e0e0e0; font-size: 2rem; font-weight: 900; }

.model-bar {
    background: rgba(124, 58, 237, 0.1);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 3px solid #7c3aed;
}
.model-bar .name { color: #a78bfa; font-weight: 700; }
.model-bar .acc { color: #10b981; font-weight: 900; font-size: 1.1rem; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(26, 26, 46, 0.8);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white !important;
}
</style>
''', unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.title('🤖 Crypto AI 趋势预测中枢')
st.caption('融合 8 大 AI 模型的加密货币小时级趋势预测系统')
st.divider()

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs(['🏠 控制台', '⚡ 趋势交叉压榨', '📊 模型战力', '📜 实盘记录'])

# ============================================================
# Tab 1: Dashboard
# ============================================================
with tabs[0]:
    rates = analyzer.get_win_rate()
    model_acc = analyzer.get_model_accuracy()

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        rate_str = f"{rates['overall']:.1%}" if rates['total'] > 0 else '—'
        st.markdown(f'''
        <div class="metric-card">
            <h3>📈 总胜率</h3>
            <div class="value">{rate_str}</div>
        </div>
        ''', unsafe_allow_html=True)
        if rates['total'] > 0:
            st.caption(f'{rates["hits"]} 命中 / {rates["total"]} 总计')

    with col2:
        if model_acc:
            best_model = max(model_acc, key=model_acc.get)
            best_acc = model_acc[best_model]
            st.markdown(f'''
            <div class="metric-card">
                <h3>👑 最强模型</h3>
                <div class="value">{best_model}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.caption(f'准确率 {best_acc:.1%}')
        else:
            st.markdown('''
            <div class="metric-card">
                <h3>👑 最强模型</h3>
                <div class="value">—</div>
            </div>
            ''', unsafe_allow_html=True)

    with col3:
        pending_count = 0
        if os.path.exists(analyzer.LOG_FILE):
            try:
                ldf = pd.read_csv(analyzer.LOG_FILE, dtype=str)
                pending_count = (ldf['result'] == 'Pending').sum()
            except Exception:
                pass
        st.markdown(f'''
        <div class="metric-card">
            <h3>⏳ 待验证预测</h3>
            <div class="value">{pending_count}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Per-symbol summary
    st.markdown('### 📋 各币种最新状态')
    if os.path.exists(analyzer.LOG_FILE):
        try:
            ldf = pd.read_csv(analyzer.LOG_FILE, dtype=str)
            summary_rows = []
            for sym in analyzer.SYMBOLS:
                sym_rows = ldf[ldf['symbol'] == sym].tail(1)
                if not sym_rows.empty:
                    r = sym_rows.iloc[0]
                    sym_rate = rates.get('symbols', {}).get(sym, {})
                    rate_str = f"{sym_rate.get('rate', 0):.0%}" if sym_rate.get('total', 0) > 0 else '—'
                    summary_rows.append({
                        '币种': sym,
                        '最新预测': f"{'🟢' if r.get('direction')=='UP' else '🔴'} {r.get('direction', '—')}",
                        '置信度': f"{r.get('confidence', '—')}%",
                        '状态': r.get('result', '—'),
                        '胜率': rate_str,
                    })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.info('暂无预测数据，请先运行 auto_predict.py 或点击"趋势交叉压榨"标签页。')
        except Exception:
            st.info('暂无预测数据')
    else:
        st.info('暂无预测数据，请先运行 auto_predict.py 或点击"趋势交叉压榨"标签页。')

# ============================================================
# Tab 2: Prediction
# ============================================================
with tabs[1]:
    st.markdown('''
    <div class="hero">
        <h2>⚡ 趋势交叉压榨</h2>
        <p>同时启动 8 大 AI 模型 × 10 个交易对 = 80 次独立分析</p>
    </div>
    ''', unsafe_allow_html=True)

    if st.button('🔥 运行趋势交叉压榨', use_container_width=True, type='primary'):
        with st.spinner('🤖 8 大模型正在分析 10 个交易对... (约 30 秒)'):
            results = []
            for sym in analyzer.SYMBOLS:
                try:
                    r = analyzer.generate_prediction(sym)
                    results.append(r)
                except Exception as e:
                    st.warning(f'{sym} 预测失败: {e}')

        if results:
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            top5 = results[:5]
            medals = ['🥇', '🥈', '🥉', '④', '⑤']

            # P1 — Full width hero card
            p1 = top5[0]
            dir_class = 'up' if p1['direction'] == 'UP' else 'down'
            dir_emoji = '🟢 ▲ UP' if p1['direction'] == 'UP' else '🔴 ▼ DOWN'
            agree = [m for m, v in p1.get('model_votes', {}).items() if v['direction'] == p1['direction']]

            st.markdown(f'''
            <div class="pred-card p1 {dir_class}">
                <span class="medal">🥇</span>
                <div class="pred-symbol">{p1["symbol"]}</div>
                <div class="pred-dir {dir_class}">{dir_emoji}</div>
                <div class="pred-conf">置信度 {p1["confidence"]:.1f}%</div>
                <div class="pred-models">{len(agree)}/8 模型同意 · {", ".join(agree)}</div>
            </div>
            ''', unsafe_allow_html=True)

            # P2-P5
            if len(top5) > 1:
                cols = st.columns(min(len(top5) - 1, 4))
                for i, r in enumerate(top5[1:]):
                    d_class = 'up' if r['direction'] == 'UP' else 'down'
                    d_emoji = '🟢 ▲' if r['direction'] == 'UP' else '🔴 ▼'
                    agree_r = [m for m, v in r.get('model_votes', {}).items() if v['direction'] == r['direction']]
                    medal = medals[i + 1] if i + 1 < len(medals) else ''

                    cols[i].markdown(f'''
                    <div class="pred-card {d_class}" style="opacity: {0.85 - i * 0.1};">
                        <span class="medal">{medal}</span>
                        <div class="pred-symbol" style="font-size: 1.1rem;">{r["symbol"]}</div>
                        <div class="pred-dir {d_class}" style="font-size: 1.4rem;">{d_emoji}</div>
                        <div class="pred-conf">{r["confidence"]:.1f}%</div>
                        <div class="pred-models">{len(agree_r)}/8 同意</div>
                    </div>
                    ''', unsafe_allow_html=True)

            # Remaining (P6-P10) as small list
            if len(results) > 5:
                with st.expander('📋 查看其余交易对'):
                    for r in results[5:]:
                        arrow = '🟢' if r['direction'] == 'UP' else '🔴'
                        st.write(f"{arrow} **{r['symbol']}** — {r['direction']} ({r['confidence']:.1f}%)")

# ============================================================
# Tab 3: Model Performance
# ============================================================
with tabs[2]:
    st.markdown('### 📊 8 大模型战力排行')
    model_acc = analyzer.get_model_accuracy()

    if model_acc:
        sorted_models = sorted(model_acc.items(), key=lambda x: x[1], reverse=True)
        king = sorted_models[0][0] if sorted_models else ''

        for name, acc in sorted_models:
            crown = ' 👑' if name == king else ''
            bar_width = max(acc * 100, 5)
            st.markdown(f'''
            <div class="model-bar">
                <span class="name">{name}{crown}</span>
                <span class="acc">{acc:.1%}</span>
            </div>
            ''', unsafe_allow_html=True)

        st.divider()
        # Per-symbol weights
        st.markdown('### ⚖️ 各币种当前权重')
        weight_rows = []
        for sym in analyzer.SYMBOLS:
            w = analyzer.load_weights(sym)
            row = {'币种': sym}
            row.update({m: f'{v:.2f}' for m, v in w.items()})
            weight_rows.append(row)
        st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)
    else:
        st.info('暂无模型性能数据。系统需要运行至少 1 轮预测并完成复盘后才能显示。')

# ============================================================
# Tab 4: History
# ============================================================
with tabs[3]:
    st.markdown('### 📜 实盘预测记录')

    if os.path.exists(analyzer.LOG_FILE):
        try:
            ldf = pd.read_csv(analyzer.LOG_FILE, dtype=str)

            # Filter
            filter_sym = st.selectbox('筛选币种', ['全部'] + analyzer.SYMBOLS)
            if filter_sym != '全部':
                ldf = ldf[ldf['symbol'] == filter_sym]

            if not ldf.empty:
                display = ldf[['timestamp', 'symbol', 'direction', 'confidence', 'result']].copy()
                display = display.sort_values('timestamp', ascending=False).reset_index(drop=True)

                def color_result(val):
                    if val == 'HIT':
                        return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981;'
                    elif val == 'MISS':
                        return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444;'
                    elif val == 'Pending':
                        return 'background-color: rgba(234, 179, 8, 0.2); color: #eab308;'
                    return ''

                styled = display.style.applymap(color_result, subset=['result'])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.info('暂无记录')
        except Exception as e:
            st.error(f'读取记录失败: {e}')
    else:
        st.info('暂无实盘记录。系统运行后会自动生成。')
