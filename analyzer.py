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

def _calc_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = _calc_atr(high, low, close, period)
    plus_di = 100 * _calc_ema(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * _calc_ema(minus_dm, period) / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = _calc_ema(dx, period)
    return adx, plus_di, minus_di

def _calc_williams_r(high, low, close, period=14):
    """Williams %R — overbought/oversold."""
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()
    wr = -100 * (highest - close) / (highest - lowest + 1e-10)
    return wr

def _calc_cci(high, low, close, period=20):
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    return cci

def _calc_roc(series, period=12):
    """Rate of Change."""
    return series.pct_change(period) * 100

def _build_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute all technical indicators, multi-timeframe context, and BTC correlation."""
    f = df.copy()
    c = f['close']
    h = f['high']
    l = f['low']
    v = f['volume']
    o = f['open']

    # ---- Trend (short + long) ----
    f['sma_7'] = _calc_sma(c, 7)
    f['sma_25'] = _calc_sma(c, 25)
    f['sma_50'] = _calc_sma(c, 50)
    f['sma_100'] = _calc_sma(c, 100)   # ~4-day trend
    f['sma_168'] = _calc_sma(c, 168)   # 7-day trend
    f['ema_12'] = _calc_ema(c, 12)
    f['ema_26'] = _calc_ema(c, 26)
    f['ema_50'] = _calc_ema(c, 50)
    f['adx'], f['plus_di'], f['minus_di'] = _calc_adx(h, l, c, 14)

    # Trend position: where is price relative to long-term MAs
    f['above_sma100'] = (c > f['sma_100']).astype(float)
    f['above_sma168'] = (c > f['sma_168']).astype(float)
    f['trend_strength'] = (f['sma_7'] - f['sma_25']) / (c + 1e-10) * 100

    # ---- Momentum (multi-period) ----
    f['rsi_14'] = _calc_rsi(c, 14)
    f['rsi_7'] = _calc_rsi(c, 7)
    f['rsi_28'] = _calc_rsi(c, 28)    # longer-term RSI
    f['stoch_k'], f['stoch_d'] = _calc_stochastic(h, l, c, 14)
    f['williams_r'] = _calc_williams_r(h, l, c, 14)
    f['cci'] = _calc_cci(h, l, c, 20)
    f['roc_12'] = _calc_roc(c, 12)
    f['roc_6'] = _calc_roc(c, 6)

    # ---- MACD ----
    f['macd_line'], f['macd_signal'], f['macd_hist'] = _calc_macd(c)

    # ---- Volatility ----
    f['bb_upper'], f['bb_mid'], f['bb_lower'] = _calc_bollinger(c)
    f['atr_14'] = _calc_atr(h, l, c, 14)
    f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / (f['bb_mid'] + 1e-10)
    f['bb_position'] = (c - f['bb_lower']) / (f['bb_upper'] - f['bb_lower'] + 1e-10)

    # ---- Volume ----
    f['obv'] = _calc_obv(c, v)
    f['vol_sma'] = _calc_sma(v, 20)
    f['vol_ratio'] = v / (f['vol_sma'] + 1e-10)

    # ---- Price Action ----
    f['pct_change'] = c.pct_change()
    f['pct_change_3'] = c.pct_change(3)
    f['pct_change_7'] = c.pct_change(7)
    f['pct_change_24'] = c.pct_change(24)
    f['body_size'] = (c - o).abs() / (c + 1e-10)
    f['upper_shadow'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (c + 1e-10)
    f['lower_shadow'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (c + 1e-10)
    f['hl_range'] = (h - l) / (c + 1e-10)

    # ---- Lag features ----
    for lag in [1, 2, 3, 6, 12]:
        f[f'ret_lag_{lag}'] = c.pct_change(lag)

    # ---- Rolling stats ----
    f['std_12'] = c.pct_change().rolling(12).std()
    f['std_24'] = c.pct_change().rolling(24).std()
    f['mean_12'] = c.pct_change().rolling(12).mean()

    # ---- BTC Correlation (for altcoins) ----
    if btc_df is not None and not btc_df.empty:
        btc_c = btc_df.set_index('timestamp')['close'].reindex(f['timestamp']).values
        btc_series = pd.Series(btc_c, index=f.index)
        f['btc_ret_6h'] = btc_series.pct_change(6)
        f['btc_ret_24h'] = btc_series.pct_change(24)
        f['btc_trend'] = (_calc_sma(btc_series, 7) > _calc_sma(btc_series, 25)).astype(float)
    else:
        f['btc_ret_6h'] = 0.0
        f['btc_ret_24h'] = 0.0
        f['btc_trend'] = 0.5

    # Target: 1 if next close > current close, else 0
    f['target'] = (c.shift(-1) > c).astype(int)

    # Drop rows with NaN features
    f = f.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return f

FEATURE_COLS = [
    # Trend (11)
    'sma_7', 'sma_25', 'sma_50', 'sma_100', 'sma_168',
    'ema_12', 'ema_26', 'ema_50', 'adx', 'plus_di', 'minus_di',
    'above_sma100', 'above_sma168', 'trend_strength',
    # Momentum (8)
    'rsi_14', 'rsi_7', 'rsi_28', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc_12', 'roc_6',
    # MACD (3)
    'macd_line', 'macd_signal', 'macd_hist',
    # Volatility (5)
    'bb_upper', 'bb_mid', 'bb_lower', 'atr_14', 'bb_width', 'bb_position',
    # Volume (3)
    'obv', 'vol_sma', 'vol_ratio',
    # Price Action (8)
    'pct_change', 'pct_change_3', 'pct_change_7', 'pct_change_24',
    'body_size', 'upper_shadow', 'lower_shadow', 'hl_range',
    # Lags (5)
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_6', 'ret_lag_12',
    # Rolling stats (3)
    'std_12', 'std_24', 'mean_12',
    # BTC correlation (3)
    'btc_ret_6h', 'btc_ret_24h', 'btc_trend',
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

        # Use recent 400 candles to avoid concept drift
        train_df = df[df['target'].notna()].tail(400).iloc[:-1]
        if len(train_df) < 50:
            return {'direction': 'UP', 'confidence': 0.5}

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['target'].astype(int).values

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
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

        # Use recent 400 candles to avoid concept drift
        train_df = df[df['target'].notna()].tail(400).iloc[:-1]
        if len(train_df) < 50:
            return {'direction': 'UP', 'confidence': 0.5}

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['target'].astype(int).values

        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
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
        if len(df) < 120:
            return {'direction': 'UP', 'confidence': 0.5}

        # Multi-feature input: close, rsi, macd_hist, vol_ratio
        features = ['close', 'rsi_14', 'macd_hist', 'vol_ratio']
        raw = df[features].tail(120).values  # (120, 4)

        # Normalize each feature to 0-1
        mins = raw.min(axis=0)
        maxs = raw.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        data = (raw - mins) / ranges

        # Build training sequences: windows of 12 -> predict next candle direction
        window = 12
        input_size = len(features)
        X_all, Y_all = [], []
        for i in range(len(data) - window):
            X_all.append(data[i:i + window])  # (24, 4)
            # Target: 1 if next close > current close (in normalized space)
            Y_all.append(1.0 if data[i + window, 0] > data[i + window - 1, 0] else 0.0)
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)

        if len(X_all) < 20:
            return {'direction': 'UP', 'confidence': 0.5}

        # LSTM params
        hidden_size = 16
        np.random.seed(42)
        scale = 0.08

        Wf = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bf = np.zeros(hidden_size)
        Wi = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bi = np.zeros(hidden_size)
        Wc = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bc = np.zeros(hidden_size)
        Wo = np.random.randn(hidden_size, hidden_size + input_size) * scale
        bo = np.zeros(hidden_size)
        Wy = np.random.randn(1, hidden_size) * scale
        by = np.zeros(1)

        lr = 0.005
        epochs = 15

        def lstm_forward(x_seq):
            """Forward pass. x_seq shape: (seq_len, input_size)"""
            T = x_seq.shape[0]
            h = np.zeros(hidden_size)
            c_state = np.zeros(hidden_size)
            for t in range(T):
                xt = x_seq[t]  # (input_size,)
                concat = np.concatenate([h, xt])
                ft = _sigmoid(Wf @ concat + bf)
                it = _sigmoid(Wi @ concat + bi)
                ct_hat = _tanh(Wc @ concat + bc)
                c_state = ft * c_state + it * ct_hat
                ot = _sigmoid(Wo @ concat + bo)
                h = ot * _tanh(c_state)
            y_pred = _sigmoid(Wy @ h + by)
            return y_pred[0], h

        # Train
        for epoch in range(epochs):
            for j in range(len(X_all)):
                y_pred, h_final = lstm_forward(X_all[j])
                error = y_pred - Y_all[j]
                Wy -= lr * error * h_final.reshape(1, -1)
                by -= lr * np.array([error])

        # Predict
        last_window = data[-window:]
        pred_prob, _ = lstm_forward(last_window)

        direction = 'UP' if pred_prob > 0.5 else 'DOWN'
        confidence = min(0.5 + abs(pred_prob - 0.5) * 1.8, 0.90)
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
    if len(df) < 200:
        log.warning(f'[{symbol}] 数据不足 ({len(df)} 行)，至少需要 200 行')
        return {'symbol': symbol, 'direction': 'HOLD', 'confidence': 0.0,
                'model_votes': {}, 'weights': DEFAULT_WEIGHTS}

    # Load BTC data for correlation (for altcoins)
    btc_df = None
    if symbol != 'BTC/USDT':
        btc_df = load_data('BTC/USDT')

    feat_df = _build_features(df, btc_df)
    if len(feat_df) < 50:
        log.warning(f'[{symbol}] 特征数据不足 ({len(feat_df)} 行)')
        return {'symbol': symbol, 'direction': 'HOLD', 'confidence': 0.0,
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

    # Confidence = how dominant the winning side is
    winning_score = max(up_score, down_score)
    dominance = winning_score / total_score  # 0.5 ~ 1.0
    final_confidence = 50.0 + (dominance - 0.5) * 90.0
    final_confidence = min(max(final_confidence, 50.0), 95.0)

    # Bonus: model agreement
    agree_count = sum(1 for v in model_votes.values() if v['direction'] == final_direction)
    if agree_count >= 8:
        final_confidence = min(final_confidence + 5, 95.0)
    elif agree_count >= 7:
        final_confidence = min(final_confidence + 3, 95.0)
    elif agree_count >= 6:
        final_confidence = min(final_confidence + 1, 95.0)

    # HOLD mechanism: if confidence is too low, don't make a call
    if final_confidence < 55.0:
        final_direction = 'HOLD'

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
