"""Read-only comparison of frozen experiment outputs; no search or LLM calls."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mcts-llm-alpha' / 'src'))
import numpy as np
from mcts_llm_alpha.evaluation.qlib_evaluator import QlibMarket
from mcts_llm_alpha.evaluation.protocol import daily_rank_ic

runs = {'old': 'verify/run2', 'new': 'paper_csi300/run1'}
rows = {}
for name, folder in runs.items():
    with (ROOT / 'mcts-llm-alpha/results' / folder / 'final_alphas_top10.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows[name] = list(csv.DictReader(stream))

market = QlibMarket(
    Path.home() / '.qlib/qlib_data/cn_data_2024', 'csi300',
    {'all': ('2011-01-01', '2024-11-30')}, horizon=10,
    fields=['open', 'high', 'low', 'close', 'volume', 'vwap'],
)
label = market.label('all')
output = {'dataset': 'cn_data_2024', 'horizon': 10, 'factors': []}
for source, factors in rows.items():
    for rank, row in enumerate(factors, 1):
        factor = market.factor(row['formula'], 'all')
        daily = daily_rank_ic(factor, label)
        yearly = {str(year): float(group.mean()) for year, group in daily.groupby(daily.index.year)}
        periods = {}
        for name, start, end in [
            ('2011_2018', '2011-01-01', '2018-12-31'),
            ('2019_2020_09', '2019-01-01', '2020-09-24'),
            ('2019_2020', '2019-01-01', '2020-12-31'),
            ('2021_2024', '2021-01-01', '2024-11-30'),
        ]:
            # Match the evaluator's period-end label purge.
            dates = label.loc[start:end].index
            retained = dates[:-10]
            periods[name] = float(daily.reindex(retained).mean())
        record = {'source': source, 'rank': rank, 'formula': row['formula'],
                  'yearly': yearly, 'periods': periods}
        output['factors'].append(record)
        print(json.dumps({'source': source, 'rank': rank, 'periods': periods}), flush=True)

output['summary'] = {}
for source in runs:
    group = [r for r in output['factors'] if r['source'] == source]
    output['summary'][source] = {
        'periods': {key: float(np.mean([r['periods'][key] for r in group])) for key in group[0]['periods']},
        # Descriptive calendar-year means; annual labels may cross year ends.
        'yearly': {key: float(np.mean([r['yearly'][key] for r in group])) for key in group[0]['yearly']},
    }
path = ROOT / 'tmp/frozen_factor_comparison.json'
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print('SUMMARY=' + json.dumps(output['summary']), flush=True)
print('OUTPUT=' + str(path), flush=True)
