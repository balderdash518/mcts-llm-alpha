"""A startup failure must itself leave a trace and an explicit failed manifest."""
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from mcts_llm_alpha.config import Config
from mcts_llm_alpha.experiment import run_experiment
from mcts_llm_alpha.llm import LLMClient


def test_market_initialization_failure_is_inside_trace_and_flushes(monkeypatch,tmp_path):
    observations=[]
    class Tracer:
        last_trace_url='https://example.test/trace'
        session_id=None
        @contextmanager
        def observe(self,name,**fields):
            observations.append(name)
            yield SimpleNamespace(update=lambda **kw:None)
    client=SimpleNamespace(tracer=Tracer(),close=lambda:observations.append('flush-and-close'))
    monkeypatch.setattr(LLMClient,'from_config',lambda config:client)
    def fail(**kwargs):
        raise RuntimeError('MARKET_INIT_SENTINEL')
    monkeypatch.setitem(sys.modules,'qlib',SimpleNamespace(init=fail))
    import mcts_llm_alpha.evaluation.qlib_evaluator as qe
    monkeypatch.setattr(qe,'custom_operators',lambda:[])
    monkeypatch.setitem(sys.modules,'qlib.data',SimpleNamespace(D=object()))
    (tmp_path/'calendars').mkdir()
    (tmp_path/'calendars'/'day.txt').write_text('2020-01-02')
    with pytest.raises(RuntimeError,match='MARKET_INIT_SENTINEL'):
        run_experiment(Config(data={'qlib_provider_uri':str(tmp_path)}),tmp_path,'test-run')
    assert observations==['run-alpha-experiment','initialize-market-data','flush-and-close']
    manifest=json.loads((tmp_path/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='failed' and manifest['trace_url']==client.tracer.last_trace_url
    assert manifest['error']=='MARKET_INIT_SENTINEL'


def test_missing_data_folder_names_the_problem(tmp_path):
    from mcts_llm_alpha.config import DataConfig
    with pytest.raises(FileNotFoundError, match='calendars/day.txt'):
        DataConfig(qlib_provider_uri=str(tmp_path / 'absent')).provider_uri()
