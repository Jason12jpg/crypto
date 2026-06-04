"""
scraper.py - 加密货币 OHLCV 数据采集器
使用 ccxt 从 Gate.io 公开 API 拉取 1H K 线，无需 API Key。
"""
import os
import time
import logging
import pandas as pd
import ccxt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'crypto_history.csv')

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
    'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'SUI/USDT',
]
TIMEFRAME = '1h'
INITIAL_CANDLES = 720  # 30 days of hourly data


def fetch_ohlcv(exchange, symbol: str, since_ms: int = None, limit: int = 1000) -> list:
    """Fetch OHLCV candles for a single symbol."""
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1  # next ms after last candle
        if len(candles) < limit:
            break
        time.sleep(0.3)
    return all_candles


def main():
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

    exchange = ccxt.gate({'enableRateLimit': True})

    # Load existing data to determine since_ms per symbol
    existing = pd.DataFrame()
    if os.path.exists(DATA_FILE):
        try:
            existing = pd.read_csv(DATA_FILE, dtype=str)
            existing['timestamp'] = pd.to_datetime(existing['timestamp'], errors='coerce')
            existing = existing.dropna(subset=['timestamp'])
            log.info(f'📂 已有数据: {len(existing)} 条记录')
        except Exception:
            existing = pd.DataFrame()

    new_rows = []

    for idx, symbol in enumerate(SYMBOLS):
        log.info(f'🔍 [{idx+1}/{len(SYMBOLS)}] 拉取 {symbol} ...')

        # Determine start time
        since_ms = None
        if not existing.empty and symbol in existing['symbol'].values:
            sym_data = existing[existing['symbol'] == symbol]
            last_ts = sym_data['timestamp'].max()
            since_ms = int(last_ts.timestamp() * 1000) + 1
            log.info(f'  ⏩ 增量模式: 从 {last_ts} 之后开始')
        else:
            # First run: fetch INITIAL_CANDLES hours back
            now_ms = exchange.milliseconds()
            since_ms = now_ms - INITIAL_CANDLES * 3600 * 1000
            log.info(f'  📥 首次拉取: 最近 {INITIAL_CANDLES} 小时')

        try:
            candles = fetch_ohlcv(exchange, symbol, since_ms)
            log.info(f'  ✅ 获取 {len(candles)} 根 K 线')

            for c in candles:
                ts = pd.Timestamp(c[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                new_rows.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'open': str(c[1]),
                    'high': str(c[2]),
                    'low': str(c[3]),
                    'close': str(c[4]),
                    'volume': str(c[5]),
                })
        except Exception as e:
            log.error(f'  ❌ {symbol} 拉取失败: {e}')

        if idx < len(SYMBOLS) - 1:
            time.sleep(0.5)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing.empty:
            # Convert existing timestamp back to string for concat
            existing['timestamp'] = existing['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        combined = combined.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        combined.to_csv(DATA_FILE, index=False)
        log.info(f'💾 数据已保存: {len(combined)} 条总记录 → {DATA_FILE}')
    else:
        log.info('ℹ️ 没有新数据')


if __name__ == '__main__':
    main()
