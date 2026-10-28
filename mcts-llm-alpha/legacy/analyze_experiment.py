"""Report signed daily Rank IC for each selected node; no additional model calls."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory',type=Path)
    args=p.parse_args()
    folder=args.run_directory.resolve()
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    tree=json.loads((folder/'search_tree.json').read_text(encoding='utf-8'))
    cfg=manifest['configuration']
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=cfg['data']['qlib_provider_uri'],region='cn',kernels=1,
              expression_cache=None,dataset_cache=None)
    start,end=cfg['data']['start_date'],cfg['data']['end_date']
    instruments=D.list_instruments(D.instruments(cfg['data']['universe']),start_time=start,end_time=end,as_list=True)
    returns=D.features(instruments,['Ref($close,-1)/$close-1'],start_time=start,end_time=end).iloc[:,0].rename('return')
    rows=[]
    series={}
    for i,node in enumerate(tree):
        formula=node.get('concrete_formula') or node['formula']
        factor=D.features(instruments,[formula],start_time=start,end_time=end).iloc[:,0].rename('factor')
        data=pd.concat([factor,returns],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        daily=data.groupby(level='datetime').apply(lambda g:g['factor'].corr(g['return'],method='spearman')
            if len(g)>=3 and g['factor'].nunique()>1 and g['return'].nunique()>1 else np.nan).dropna()
        split=pd.Timestamp(cfg['evaluation']['split_date'])
        ins=daily[daily.index<split];out=daily[daily.index>=split]
        row={'node':i,'attempt_id':node['attempt_id'],'dimension':node['action_dim'],
            'formula':formula,'daily_rank_ic':float(daily.mean()),
            'icir':float(daily.mean()/daily.std(ddof=0)) if daily.std(ddof=0)>0 else None,
            'is_rank_ic':float(ins.mean()),'oos_diagnostic_rank_ic':float(out.mean()),
            'positive_ic_day_fraction':float((daily>0).mean()),'ic_days':len(daily),
            'observations':len(data),'is_days':len(ins),'oos_days':len(out),
            'turnover':node['raw_scores'].get('Turnover'), 'reward':node['reward']}
        rows.append(row);series[f'node_{i}']=daily
    pd.DataFrame(rows).to_csv(folder/'node_metrics.csv',index=False)
    pd.DataFrame(series).to_csv(folder/'daily_rank_ic.csv')
    summary={'metric_definition':'Mean daily cross-sectional Spearman IC against next-day close return',
        'split_warning':'The search observes metrics over the entire interval; the OOS split is a diagnostic, not an untouched holdout.',
        'universe_warning':'Current evaluator uses the union of CSI300 constituents over this interval, rather than a per-date membership mask.',
        'nodes':rows}
    (folder/'metric_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
