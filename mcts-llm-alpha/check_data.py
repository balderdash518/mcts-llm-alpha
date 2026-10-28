#!/usr/bin/env python
"""Check that a config's Qlib data covers its periods, universe and fields (no model calls)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from mcts_llm_alpha.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    uri = Path(config.data.provider_uri())
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(uri), region=config.data.region, kernels=1, expression_cache=None,
              dataset_cache=None)
    calendar = D.calendar(freq="day")
    print(f"数据目录: {uri}")
    print(f"交易日历: {str(calendar[0])[:10]} 至 {str(calendar[-1])[:10]}（需覆盖 "
          f"{config.data.train_period[0]} 至 {config.data.test_period[1]}）")
    print("指数成分文件:", sorted(p.stem for p in (uri / "instruments").glob("*.txt")))
    start, end = config.data.train_period[0], config.data.test_period[1]
    instruments = D.list_instruments(D.instruments(config.data.universe), start_time=start,
                                     end_time=end, as_list=True)
    print(f"{config.data.universe} 在区间内的股票数: {len(instruments)}")
    for period in (config.data.train_period, config.data.test_period):
        frame = D.features(D.instruments(config.data.universe), [f"${f}" for f in config.data.fields],
                           start_time=period[0], end_time=period[1])
        coverage = {f: f"{frame.iloc[:, i].notna().mean():.1%}" for i, f in enumerate(config.data.fields)}
        print(f"{period[0]} 至 {period[1]} 字段非空比例: {coverage}")


if __name__ == "__main__":
    main()
