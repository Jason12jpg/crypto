"""
auto_predict.py - 自动复盘 & 预测闭环引擎
GitHub Actions 每 2 小时调用此脚本:
1. 复盘上一轮 Pending 预测
2. 更新模型权重（自我进化）
3. 生成新一轮预测并写入日志
"""
import logging
import analyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def run():
    log.info('=' * 60)
    log.info('CRYPTO AUTO-PREDICT & SELF-EVOLUTION ENGINE')
    log.info('=' * 60)

    all_results = []

    for symbol in analyzer.SYMBOLS:
        log.info(f'\n--- [{symbol}] ---')

        # Step 1: Evaluate past Pending predictions & evolve weights
        try:
            stats = analyzer.evaluate_and_evolve(symbol)
            if stats['evaluated'] > 0:
                log.info(f'  复盘: {stats["hits"]} HIT / {stats["misses"]} MISS')
        except Exception as e:
            log.error(f'  复盘失败: {e}')

        # Step 2: Generate new prediction
        try:
            result = analyzer.generate_prediction(symbol)
            log.info(f'  预测: {result["direction"]} (置信度 {result["confidence"]:.1f}%)')

            # Log model votes
            for m, vote in result.get('model_votes', {}).items():
                log.info(f'    {m}: {vote["direction"]} ({vote["confidence"]:.2f})')

            # Step 3: Save to prediction log as Pending
            analyzer.save_prediction_log(symbol, result)
            all_results.append(result)
        except Exception as e:
            log.error(f'  预测失败: {e}')

    # Summary
    log.info('\n' + '=' * 60)
    log.info('SUMMARY')
    log.info('=' * 60)

    # Sort by confidence
    all_results.sort(key=lambda x: x['confidence'], reverse=True)
    medals = ['🥇', '🥈', '🥉', '④', '⑤']
    for i, r in enumerate(all_results[:5]):
        medal = medals[i] if i < len(medals) else f'  '
        arrow = '🟢 UP' if r['direction'] == 'UP' else '🔴 DOWN'
        log.info(f'{medal} {r["symbol"]}: {arrow} ({r["confidence"]:.1f}%)')

    # Win rate
    rates = analyzer.get_win_rate()
    if rates['total'] > 0:
        log.info(f'\n总胜率: {rates["overall"]:.1%} ({rates["hits"]}/{rates["total"]})')
    else:
        log.info('\n暂无历史复盘数据')

    log.info('Auto-predict complete.')


if __name__ == '__main__':
    run()
