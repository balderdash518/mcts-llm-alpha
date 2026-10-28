"""Run a full mining experiment: multi-tree search on the training period, then the
top-k alphas by training RankIR are measured on the untouched test period."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _verify_market(config, D):
    import pandas as pd
    report = {}
    full = D.calendar(freq='day')
    calendar = D.calendar(start_time=config.data.train_period[0], end_time=config.data.test_period[1])
    if not len(calendar):
        raise ValueError('No market dates in the configured periods')
    first, last = str(calendar[0])[:10], str(calendar[-1])[:10]
    slack = pd.Timedelta(days=7)   # period bounds may fall on holidays or weekends
    for name, (start, end) in (('train', config.data.train_period), ('test', config.data.test_period)):
        if full[0] > pd.Timestamp(start) + slack or full[-1] < pd.Timestamp(end) - slack:
            raise ValueError(f'{name} period {start}..{end} is not covered by the data '
                             f'({str(full[0])[:10]}..{str(full[-1])[:10]})')
    instruments = D.list_instruments(D.instruments(config.data.universe),
                                     start_time=config.data.train_period[0],
                                     end_time=config.data.test_period[1], as_list=True)
    if not instruments:
        raise ValueError(f'Universe {config.data.universe} is empty in the configured periods')
    root = Path(config.data.provider_uri()).expanduser()
    for field in config.data.fields:
        if not any((root / 'features' / i.lower() / f'{field}.day.bin').exists() for i in instruments[:20]):
            raise ValueError(f'Field ${field} is missing from the local dataset; remove it from data.fields')
    report.update(trading_days=len(calendar), instruments=len(instruments), first_date=first, last_date=last)
    return report


def _node_rows(trees):
    rows = []
    for root in trees:
        for node in root.walk():
            rows.append({'tree': node.tree, 'label': node.label, 'attempt_id': node.attempt_id,
                         'parent': node.parent.label if node.parent else None,
                         'dimension': node.dimension, 'formula': node.formula,
                         'symbolic': node.symbolic, 'parameters': node.parameters,
                         'metrics': node.metrics, 'scores': node.scores, 'score': node.score,
                         'visits': node.visits, 'q': node.q, 'summary': node.summary,
                         'suggestions': node.suggestions, 'overfit_reason': node.overfit_reason,
                         'admission': node.admission})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config, output_dir, run_id, resume=None):
    from .llm import LLMClient
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient.from_config(config.llm)
    client.tracer.session_id = run_id
    if getattr(client, 'transport', None) is not None:
        client.transport.call_log_path = output_dir / 'llm_calls.jsonl'
    report = {'run_id': run_id, 'status': 'running', 'model': config.llm.model,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'configuration': config.model_dump(mode='json'), 'output_dir': str(output_dir)}

    def write_report():
        report['trace_url'] = client.tracer.last_trace_url
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    try:
        with client.tracer.observe('run-alpha-experiment', as_type='agent', input={
                'run_id': run_id, 'universe': config.data.universe,
                'train_period': config.data.train_period, 'test_period': config.data.test_period,
                'horizon': config.evaluation.horizon,
                'total_search_count': config.mcts.total_search_count}) as run_span:
            write_report()
            print(f'RUN_ID={run_id}', flush=True)
            with client.tracer.observe('initialize-market-data', as_type='tool', input={
                    'provider_uri': config.data.qlib_provider_uri}) as data_span:
                import qlib
                from qlib.data import D
                from .evaluation.qlib_evaluator import QlibMarket, custom_operators
                qlib.init(provider_uri=config.data.provider_uri(), region=config.data.region,
                          custom_ops=custom_operators(), kernels=1,
                          expression_cache=None, dataset_cache=None)
                coverage = _verify_market(config, D)
                report['market_coverage'] = coverage
                data_span.update(output=coverage)
                print('MARKET_COVERAGE=' + json.dumps(coverage), flush=True)

            from .evaluation import EffectiveCriteria
            from .mcts import MCTSSearch, SearchSettings, ExperimentLedger
            market = QlibMarket(config.data.provider_uri(), config.data.universe,
                                {'train': config.data.train_period, 'test': config.data.test_period},
                                config.evaluation.horizon, config.data.fields,
                                config.evaluation.min_days, config.evaluation.min_coverage,
                                initialize=False)
            ledger = ExperimentLedger(config.output.experiment_log_file, {
                'provider_uri': config.data.provider_uri(), 'data_revision': config.data.data_revision,
                'universe': config.data.universe, 'fields': config.data.fields,
                'train_period': config.data.train_period, 'horizon': config.evaluation.horizon,
                'label': 'close_to_close_forward_return', 'metrics_version': 'paper-f1-v1'})
            e = config.evaluation
            criteria = EffectiveCriteria(e.rank_ic_min, e.rank_ir_min, e.relative_rank_max,
                                         e.turnover_max, e.max_correlation, e.relative_rank_min_size)
            m = config.mcts
            settings = SearchSettings(
                total_search_count=m.total_search_count, tree_initial_budget=m.tree_initial_budget,
                budget_increment=m.budget_increment, exploration_constant=m.exploration_constant,
                dimension_temperature=m.dimension_temperature, max_tree_iterations=m.max_tree_iterations,
                max_corrections=m.max_corrections, max_root_attempts=m.max_root_attempts,
                few_shot_k=e.few_shot_k, correlation_filter_ratio=e.correlation_filter_ratio,
                seed_formula=m.initial_seed_formula, random_seed=m.random_seed,
                checkpoint_path=str(output_dir / 'checkpoint.pkl'))
            search = MCTSSearch(client, market, ledger, settings=settings, criteria=criteria,
                                fields=config.data.fields)
            search.fsa.top_k, search.fsa.min_support = config.fsa.top_k, config.fsa.min_support
            if resume:
                search.load_checkpoint(resume)
            repository = search.run()

            (output_dir / 'search_trees.json').write_text(json.dumps(
                _node_rows(search.trees), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            _write_csv(output_dir / 'alpha_repository.csv', [
                {'tree': a.tree, 'label': a.label, 'formula': a.formula,
                 **{f'train_{k}': v for k, v in a.metrics.items()}} for a in repository])
            final = {}
            with client.tracer.observe('evaluate-test-period', as_type='evaluator') as test_span:
                for k in e.final_alpha_counts:
                    rows = []
                    for alpha in search.final_alphas(k):
                        row = {'tree': alpha.tree, 'label': alpha.label, 'formula': alpha.formula,
                               **{f'train_{n}': v for n, v in alpha.metrics.items()}}
                        try:
                            test = market.measure(alpha.formula, 'test').metrics
                            row.update({f'test_{n}': v for n, v in test.items() if n != 'max_corr'})
                        except Exception as error:
                            row['test_error'] = str(error)
                        rows.append(row)
                    _write_csv(output_dir / f'final_alphas_top{k}.csv', rows)
                    final[k] = len(rows)
                test_span.update(output=final)
            statuses = {}
            for record in ledger.records.values():
                statuses[record.get('status')] = statuses.get(record.get('status'), 0) + 1
            report.update(status='completed', search_count=search.search_count,
                          trees=search.tree_index, repository_size=len(repository),
                          final_alpha_sets=final, avoid_patterns=search.avoid,
                          ledger_statuses=statuses)
            run_span.update(output={k: report[k] for k in (
                'status', 'search_count', 'trees', 'repository_size', 'final_alpha_sets')})
            return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_report()
        client.close()
