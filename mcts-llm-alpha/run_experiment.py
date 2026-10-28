#!/usr/bin/env python
"""Paper-protocol alpha mining run with a full Langfuse execution trace."""
import argparse
import contextlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'src'))
from mcts_llm_alpha.config import load_config
from mcts_llm_alpha.experiment import run_experiment


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        if not self.file.closed:   # late messages (e.g. Qlib's exit hook) go to the terminal only
            self.file.write(text)
            self.file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.file.closed:
            self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='YAML config (default: the paper setup)')
    parser.add_argument('--search-count', type=int, help='Override mcts.total_search_count')
    parser.add_argument('--seed-formula', help='Seed formula for the first tree')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume', help='checkpoint.pkl of an interrupted run')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.search_count is not None:
        config.mcts.total_search_count = args.search_count
    if args.seed_formula:
        config.mcts.initial_seed_formula = args.seed_formula
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    run_id = 'mcts-' + stamp
    output = Path(args.output_dir or ROOT / config.output.results_dir / run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.log').open('a' if args.resume else 'w', encoding='utf-8') as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), \
                contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
            logging.basicConfig(level=logging.INFO)
            for noisy in ('httpx', 'httpx2', 'openai', 'urllib3'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            report = run_experiment(config, output, run_id, resume=args.resume)
            print(f"COMPLETED: search_count={report['search_count']}, trees={report['trees']}, "
                  f"repository={report['repository_size']}")
            print('RESULTS=' + str(output))
            print('LANGFUSE_TRACE=' + str(report['trace_url']))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
