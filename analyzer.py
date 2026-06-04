"""
analyzer.py - Crypto AI 趋势预测核心引擎
8 大模型: trend, momentum, volatility, macd, volume, forest, xgb, lstm
"""
import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'crypto_history.csv')
LOG_FILE = os.path.join(BASE_DIR, 'data', 'prediction_log.csv')
WEIGHT_FILE = os.path.join(BASE_DIR, 'data', 'model_weights.csv')

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
    'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'SUI/USDT',
]
MODEL_NAMES = ['trend', 'momentum', 'volatility', 'macd', 'volume', 'forest', 'xgb', 'lstm']
DEFAULT_WEIGHTS = {m: 0.5 for m in MODEL_NAMES}

# ============================================================
# Data Loading
# ============================================================
def load_data(symbol: str = None) -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['timestamp', 'close']).sort_values('timestamp').reset_index(drop=True)
    if symbol:
        df = df[df['symbol'] == symbol].reset_index(drop=True)
    return df

# ============================================================
# Weight Management
# ============================================================
def load_weights(symbol: str) -> dict:
    if not os.path.exists(WEIGHT_FILE):
        return dict(DEFAULT_WEIGHTS)
    try:
        wdf = pd.read_csv(WEIGHT_FILE)
        row = wdf[wdf['symbol'] == symbol]
        if row.empty:
            return dict(DEFAULT_WEIGHTS)
        r = row.iloc[0]
        return {m: float(r.get(m, 0.5)) for m in MODEL_NAMES}
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def save_weights(symbol: str, weights: dict):
    os.makedirs(os.path.dirname(WEIGHT_FILE), exist_ok=True)
    if os.path.exists(WEIGHT_FILE):
        try:
            wdf = pd.read_csv(WEIGHT_FILE)
        except Exception:
            wdf = pd.DataFrame()
    else:
        wdf = pd.DataFrame()
    row = {'symbol': symbol}
    row.update(weights)
    if not wdf.empty and symbol in wdf['symbol'].values:
        for m in MODEL_NAMES:
            wdf.loc[wdf['symbol'] == symbol, m] = weights.get(m, 0.5)
    else:
        wdf = pd.concat([wdf, pd.DataFrame([row])], ignore_index=True)
    wdf.to_csv(WEIGHT_FILE, index=False)

# ============================================================
# Technical Indicators (pure numpy/pandas)
# ============================================================
def _calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def _calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def _calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = _calc_ema(series, fast)
    ema_slow = _calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _calc_bollinger(series, period=20, std_dev=2):
    mid = _calc_sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def _calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def _calc_obv(close, volume):
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv

def _calc_stochastic(high, low, close, period=14):
    lowest = low.rolling(window=period, min_periods=period).min()
    highest = high.rolling(window=period, min_periods=period).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(window=3, min_periods=3).mean()
    return k, d

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and target column."""
    f = df.copy()
    c = f['close']
    h = f['high']
    l = f['low']
    v = f['volume']

    # Trend
    f['sma_7'] = _calc_sma(c, 7)
    f['sma_25'] = _calc_sma(c, 25)
    f['ema_12'] = _calc_ema(c, 12)
    f['ema_26'] = _calc_ema(c, 26)

    # Momentum
    f['rsi_14'] = _calc_rsi(c, 14)
    f['stoch_k'], f['stoch_d'] = _calc_stochastic(h, l, c, 14)

    # MACD
    f['macd_line'], f['macd_signal'], f['macd_hist'] = _calc_macd(c)

    # Volatility
    f['bb_upper'], f['bb_mid'], f['bb_lower'] = _calc_bollinger(c)
    f['atr_14'] = _calc_atr(h, l, c, 14)

    # Volume
    f['obv'] = _calc_obv(c, v)
    f['vol_sma'] = _calc_sma(v, 20)

    # Price changes
    f['pct_change'] = c.pct_change()
    f['pct_change_3'] = c.pct_change(3)
    f['pct_change_7'] = c.pct_change(7)

    # Target: 1 if next close > current close, else 0
    f['target'] = (c.shift(-1) > c).astype(int)

    # Drop rows with NaN features (first ~26 rows from indicators)
    feature_cols = [
        'sma_7', 'sma_25', 'ema_12', 'ema_26', 'rsi_14', 'stoch_k', 'stoch_d',
        'macd_line', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower',
        'atr_14', 'obv', 'vol_sma', 'pct_change', 'pct_change_3', 'pct_change_7',
    ]
    f = f.dropna(subset=feature_cols).reset_index(drop=True)
    return f

FEATURE_COLS = [
    'sma_7', 'sma_25', 'ema_12', 'ema_26', 'rsi_14', 'stoch_k', 'stoch_d',
    'macd_line', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower',
    'atr_14', 'obv', 'vol_sma', 'pct_change', 'pct_change_3', 'pct_change_7',
]

# ============================================================
# Model 1: Trend (SMA/EMA Crossover)
# ============================================================
def _predict_trend(df: pd.DataFrame) -> dict:
    try:
        last = df.iloc[-1]
        sma_bull = last['sma_7'] > last['sma_25']
        ema_bull = last['ema_12'] > last['ema_26']

        if sma_bull and ema_bull:
            direction = 'UP'
            gap = (last['sma_7'] - last['sma_25']) / last['close'] * 100
            confidence = min(0.5 + abs(gap) * 0.1, 0.95)
        elif not sma_bull and not ema_bull:
            direction = 'DOWN'
            gap = (last['sma_25'] - last['sma_7']) / last['close'] * 100
            confidence = min(0.5 + abs(gap) * 0.1, 0.95)
        else:
            direction = 'UP' if sma_bull else 'DOWN'
            confidence = 0.45
        return {'direction': direction, 'confidence': confidence}
    except Exception:
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 2: Momentum (RSI + Stochastic)
# ============================================================
def _predict_momentum(df: pd.DataFrame) -> dict:
    try:
        last = df.iloc[-1]
        rsi = last['rsi_14']
        k, d = last['stoch_k'], last['stoch_d']

        if rsi < 30:
            direction = 'UP'
            confidence = min(0.6 + (30 - rsi) / 100, 0.95)
        elif rsi > 70:
            direction = 'DOWN'
            confidence = min(0.6 + (rsi - 70) / 100, 0.95)
        else:
            # Stochastic crossover
            prev = df.iloc[-2]
            if k > d and prev['stoch_k'] <= prev['stoch_d']:
                direction = 'UP'
                confidence = 0.55
            elif k < d and prev['stoch_k'] >= prev['stoch_d']:
                direction = 'DOWN'
                confidence = 0.55
            else:
                direction = 'UP' if rsi < 50 else 'DOWN'
                confidence = 0.4 + abs(rsi - 50) / 200
        return {'direction': direction, 'confidence': confidence}
    except Exception:
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 3: Volatility (Bollinger Bands)
# ============================================================
def _predict_volatility(df: pd.DataFrame) -> dict:
    try:
        last = df.iloc[-1]
        close = last['close']
        upper = last['bb_upper']
        lower = last['bb_lower']
        mid = last['bb_mid']
        band_width = upper - lower
        if band_width < 1e-10:
            return {'direction': 'UP', 'confidence': 0.5}

        position = (close - lower) / band_width  # 0=lower, 1=upper

        if position < 0.2:
            direction = 'UP'
            confidence = min(0.6 + (0.2 - position) * 1.5, 0.90)
        elif position > 0.8:
            direction = 'DOWN'
            confidence = min(0.6 + (position - 0.8) * 1.5, 0.90)
        else:
            direction = 'UP' if close < mid else 'DOWN'
            confidence = 0.4 + abs(position - 0.5) * 0.4
        return {'direction': direction, 'confidence': confidence}
    except Exception:
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 4: MACD
# ============================================================
def _predict_macd(df: pd.DataFrame) -> dict:
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        macd = last['macd_line']
        signal = last['macd_signal']
        hist = last['macd_hist']
        prev_hist = prev['macd_hist']

        # Crossover detection
        cross_up = macd > signal and prev['macd_line'] <= prev['macd_signal']
        cross_down = macd < signal and prev['macd_line'] >= prev['macd_signal']

        if cross_up:
            direction = 'UP'
            confidence = 0.65
        elif cross_down:
            direction = 'DOWN'
            confidence = 0.65
        elif hist > 0 and hist > prev_hist:
            direction = 'UP'
            confidence = 0.55
        elif hist < 0 and hist < prev_hist:
            direction = 'DOWN'
            confidence = 0.55
        else:
            direction = 'UP' if hist > 0 else 'DOWN'
            confidence = 0.45

        return {'direction': direction, 'confidence': confidence}
    except Exception:
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 5: Volume (OBV + Volume Analysis)
# ============================================================
def _predict_volume(df: pd.DataFrame) -> dict:
    try:
        last_5 = df.tail(5)
        obv_trend = last_5['obv'].iloc[-1] > last_5['obv'].iloc[0]
        price_trend = last_5['close'].iloc[-1] > last_5['close'].iloc[0]
        vol_spike = df.iloc[-1]['volume'] > df.iloc[-1]['vol_sma'] * 1.5

        if obv_trend and price_trend:
            direction = 'UP'
            confidence = 0.6 if vol_spike else 0.55
        elif not obv_trend and not price_trend:
            direction = 'DOWN'
            confidence = 0.6 if vol_spike else 0.55
        elif obv_trend and not price_trend:
            # Bullish divergence
            direction = 'UP'
            confidence = 0.60
        elif not obv_trend and price_trend:
            # Bearish divergence
            direction = 'DOWN'
            confidence = 0.60
        else:
            direction = 'UP'
            confidence = 0.5

        return {'direction': direction, 'confidence': confidence}
    except Exception:
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 6: Random Forest
# ============================================================
def _predict_forest(df: pd.DataFrame) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier

        # Train on all but last row
        train_df = df[df['target'].notna()].iloc[:-1]
        if len(train_df) < 50:
            return {'direction': 'UP', 'confidence': 0.5}

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['target'].astype(int).values

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        X_last = df[FEATURE_COLS].iloc[[-1]].values
        proba = model.predict_proba(X_last)[0]
        pred_class = model.predict(X_last)[0]

        direction = 'UP' if pred_class == 1 else 'DOWN'
        confidence = float(max(proba))
        return {'direction': direction, 'confidence': confidence}
    except Exception as e:
        log.warning(f'Forest model error: {e}')
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 7: XGBoost
# ============================================================
def _predict_xgb(df: pd.DataFrame) -> dict:
    try:
        from xgboost import XGBClassifier

        train_df = df[df['target'].notna()].iloc[:-1]
        if len(train_df) < 50:
            return {'direction': 'UP', 'confidence': 0.5}

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['target'].astype(int).values

        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)

        X_last = df[FEATURE_COLS].iloc[[-1]].values
        proba = model.predict_proba(X_last)[0]
        pred_class = model.predict(X_last)[0]

        direction = 'UP' if pred_class == 1 else 'DOWN'
        confidence = float(max(proba))
        return {'direction': direction, 'confidence': confidence}
    except Exception as e:
        log.warning(f'XGB model error: {e}')
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Model 8: Lightweight LSTM (pure numpy)
# ============================================================
def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def _tanh(x):
    return np.tanh(x)

def _predict_lstm(df: pd.DataFrame) -> dict:
    try:
        close = df['close'].values
        if len(close) < 60:
            return {'direction': 'UP', 'confidence': 0.5}

        # Normalize to 0-1
        seq = close[-60:]
        mn, mx = seq.min(), seq.max()
        if mx - mn < 1e-10:
            return {'direction': 'UP', 'confidence': 0.5}
        seq_norm = (seq - mn) / (mx - mn)

        # Build training sequences: windows of 12 -> predict next
        window = 12
        X_all, Y_all = [], []
        for i in range(len(seq_norm) - window):
            X_all.append(seq_norm[i:i + window])
            Y_all.append(seq_norm[i + window])
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)

        if len(X_all) < 10:
            return {'direction': 'UP', 'confidence': 0.5}

        # Simple single-layer LSTM
        input_size = 1
        hidden_size = 16
        np.random.seed(42)
        scale = 0.1

        # LSTM weights
        Wf = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bf = np.zeros(hidden_size)
        Wi = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bi = np.zeros(hidden_size)
        Wc = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bc = np.zeros(hidden_size)
        Wo = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bo = np.zeros(hidden_size)
        # Output layer
        Wy = np.random.randn(1, hidden_size) * scale
        by = np.zeros(1)

        lr = 0.005
        epochs = 30

        def lstm_forward(x_seq):
            """Forward pass through LSTM. x_seq shape: (seq_len,)"""
            T = len(x_seq)
            h = np.zeros(hidden_size)
            c = np.zeros(hidden_size)
            for t in range(T):
                xt = np.array([x_seq[t]])
                concat = np.concatenate([h, xt])
                ft = _sigmoid(Wf @ concat + bf)
                it = _sigmoid(Wi @ concat + bi)
                ct_hat = _tanh(Wc @ concat + bc)
                c = ft * c + it * ct_hat
                ot = _sigmoid(Wo @ concat + bo)
                h = ot * _tanh(c)
            y_pred = _sigmoid(Wy @ h + by)
            return y_pred[0], h

        # Quick training loop
        for epoch in range(epochs):
            total_loss = 0
            for j in range(len(X_all)):
                y_pred, h_final = lstm_forward(X_all[j])
                target = Y_all[j]
                error = y_pred - target
                total_loss += error ** 2

                # Simple gradient update on output layer only
                grad_Wy = error * h_final.reshape(1, -1)
                grad_by = np.array([error])
                Wy -= lr * grad_Wy
                by -= lr * grad_by

        # Predict next value
        last_window = seq_norm[-window:]
        pred_val, _ = lstm_forward(last_window)
        current_val = seq_norm[-1]

        direction = 'UP' if pred_val > current_val else 'DOWN'
        confidence = min(0.5 + abs(pred_val - current_val) * 3, 0.85)
        return {'direction': direction, 'confidence': confidence}
    except Exception as e:
        log.warning(f'LSTM model error: {e}')
        return {'direction': 'UP', 'confidence': 0.5}

# ============================================================
# Main Prediction Entry Point
# ============================================================
ALL_MODELS = {
    'trend': _predict_trend,
    'momentum': _predict_momentum,
    'volatility': _predict_volatility,
    'macd': _predict_macd,
    'volume': _predict_volume,
    'forest': _predict_forest,
    'xgb': _predict_xgb,
    'lstm': _predict_lstm,
}

def generate_prediction(symbol: str) -> dict:
    """Generate weighted prediction for a symbol using all 8 models."""
    df = load_data(symbol)
    if len(df) < 100:
        log.warning(f'[{symbol}] 数据不足 ({len(df)} 行)，至少需要 100 行')
        return {'symbol': symbol, 'direction': 'UP', 'confidence': 50.0,
                'model_votes': {}, 'weights': DEFAULT_WEIGHTS}

    feat_df = _build_features(df)
    if len(feat_df) < 50:
        log.warning(f'[{symbol}] 特征数据不足 ({len(feat_df)} 行)')
        return {'symbol': symbol, 'direction': 'UP', 'confidence': 50.0,
                'model_votes': {}, 'weights': DEFAULT_WEIGHTS}

    weights = load_weights(symbol)
    model_votes = {}

    # Collect all model predictions
    up_score = 0.0
    down_score = 0.0
    for name, func in ALL_MODELS.items():
        result = func(feat_df)
        model_votes[name] = result
        w = weights.get(name, 0.5)
        weighted_conf = w * result['confidence']
        if result['direction'] == 'UP':
            up_score += weighted_conf
        else:
            down_score += weighted_conf

    total_score = up_score + down_score
    if total_score < 1e-10:
        total_score = 1.0

    # Direction = whichever side has more weighted conviction
    final_direction = 'UP' if up_score >= down_score else 'DOWN'

    # Confidence = how dominant the winning side is (50% = dead split, 100% = unanimous)
    winning_score = max(up_score, down_score)
    dominance = winning_score / total_score  # 0.5 ~ 1.0

    # Scale dominance to a user-friendly 50-95% range
    # dominance=0.5 → 50%, dominance=1.0 → 95%
    final_confidence = 50.0 + (dominance - 0.5) * 90.0
    final_confidence = min(max(final_confidence, 50.0), 95.0)

    # Bonus: count how many models agree
    agree_count = sum(1 for v in model_votes.values() if v['direction'] == final_direction)
    # 8/8 agree → +5%, 7/8 → +3%, 6/8 → +1%
    if agree_count >= 8:
        final_confidence = min(final_confidence + 5, 95.0)
    elif agree_count >= 7:
        final_confidence = min(final_confidence + 3, 95.0)
    elif agree_count >= 6:
        final_confidence = min(final_confidence + 1, 95.0)

    return {
        'symbol': symbol,
        'direction': final_direction,
        'confidence': round(final_confidence, 1),
        'model_votes': model_votes,
        'weights': weights,
    }

# ============================================================
# Prediction Log
# ============================================================
def save_prediction_log(symbol: str, prediction: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    now_str = datetime.now().strftime('%Y-%m-%d %H:00:00')

    row = {
        'timestamp': now_str,
        'symbol': symbol,
        'direction': prediction['direction'],
        'confidence': prediction['confidence'],
        'result': 'Pending',
    }
    for m in MODEL_NAMES:
        vote = prediction.get('model_votes', {}).get(m, {})
        row[f'pred_{m}'] = vote.get('direction', '')

    if os.path.exists(LOG_FILE):
        try:
            ldf = pd.read_csv(LOG_FILE, dtype=str)
        except Exception:
            ldf = pd.DataFrame()
    else:
        ldf = pd.DataFrame()

    ldf = pd.concat([ldf, pd.DataFrame([row])], ignore_index=True)
    ldf = ldf.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
    ldf.to_csv(LOG_FILE, index=False)

# ============================================================
# Evaluate & Evolve
# ============================================================
def evaluate_and_evolve(symbol: str) -> dict:
    """Check Pending predictions against actual results, update weights."""
    stats = {'evaluated': 0, 'hits': 0, 'misses': 0}

    if not os.path.exists(LOG_FILE) or not os.path.exists(DATA_FILE):
        return stats

    try:
        ldf = pd.read_csv(LOG_FILE, dtype=str)
    except Exception:
        return stats

    pending = ldf[(ldf['symbol'] == symbol) & (ldf['result'] == 'Pending')]
    if pending.empty:
        return stats

    price_df = load_data(symbol)
    if price_df.empty:
        return stats

    price_df['ts_hour'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
    weights = load_weights(symbol)
    updated = False

    for idx, row in pending.iterrows():
        pred_ts = row['timestamp']
        pred_dir = row['direction']

        # Find the candle at pred_ts to get actual result
        match = price_df[price_df['ts_hour'] == pred_ts]
        if match.empty:
            continue

        candle = match.iloc[-1]
        actual_dir = 'UP' if candle['close'] > candle['open'] else 'DOWN'

        if pred_dir == actual_dir:
            ldf.at[idx, 'result'] = 'HIT'
            stats['hits'] += 1
            # Boost weights for correct models
            for m in MODEL_NAMES:
                pred_m = row.get(f'pred_{m}', '')
                if pred_m == actual_dir:
                    weights[m] = min(weights[m] + 0.05, 1.0)
        else:
            ldf.at[idx, 'result'] = 'MISS'
            stats['misses'] += 1
            for m in MODEL_NAMES:
                pred_m = row.get(f'pred_{m}', '')
                if pred_m != actual_dir and pred_m != '':
                    weights[m] = max(weights[m] - 0.03, 0.1)

        stats['evaluated'] += 1
        updated = True

    if updated:
        ldf.to_csv(LOG_FILE, index=False)
        save_weights(symbol, weights)
        log.info(f'[{symbol}] 复盘: {stats["hits"]} HIT / {stats["misses"]} MISS (共 {stats["evaluated"]})')

    return stats

# ============================================================
# Analytics for UI
# ============================================================
def get_model_accuracy(symbol: str = None) -> dict:
    """Per-model accuracy from prediction log."""
    if not os.path.exists(LOG_FILE):
        return {}
    try:
        ldf = pd.read_csv(LOG_FILE, dtype=str)
        evaluated = ldf[ldf['result'].isin(['HIT', 'MISS'])]
        if symbol:
            evaluated = evaluated[evaluated['symbol'] == symbol]
        if evaluated.empty:
            return {}

        acc = {}
        for m in MODEL_NAMES:
            col = f'pred_{m}'
            if col not in evaluated.columns:
                continue
            # Model was correct if its prediction matches the actual result direction
            correct = 0
            total = 0
            for _, row in evaluated.iterrows():
                pred_m = row.get(col, '')
                if pred_m in ('UP', 'DOWN'):
                    actual = 'UP' if row['result'] == 'HIT' else 'DOWN'
                    # Re-derive: if main prediction == actual → HIT
                    # Model correct if pred_m == row['direction'] and result=='HIT'
                    # OR pred_m != row['direction'] and result=='MISS'
                    main_dir = row.get('direction', '')
                    if row['result'] == 'HIT':
                        actual = main_dir
                    else:
                        actual = 'DOWN' if main_dir == 'UP' else 'UP'
                    if pred_m == actual:
                        correct += 1
                    total += 1
            if total > 0:
                acc[m] = correct / total
        return acc
    except Exception:
        return {}


def get_win_rate() -> dict:
    """Overall and per-symbol win rates."""
    result = {'overall': 0.0, 'total': 0, 'hits': 0, 'symbols': {}}
    if not os.path.exists(LOG_FILE):
        return result
    try:
        ldf = pd.read_csv(LOG_FILE, dtype=str)
        evaluated = ldf[ldf['result'].isin(['HIT', 'MISS'])]
        if evaluated.empty:
            return result

        total = len(evaluated)
        hits = (evaluated['result'] == 'HIT').sum()
        result['overall'] = hits / total if total > 0 else 0.0
        result['total'] = int(total)
        result['hits'] = int(hits)

        for sym in SYMBOLS:
            sym_data = evaluated[evaluated['symbol'] == sym]
            sym_total = len(sym_data)
            sym_hits = (sym_data['result'] == 'HIT').sum()
            result['symbols'][sym] = {
                'rate': sym_hits / sym_total if sym_total > 0 else 0.0,
                'total': int(sym_total),
                'hits': int(sym_hits),
            }
        return result
    except Exception:
        return result
