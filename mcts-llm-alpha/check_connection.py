#!/usr/bin/env python
"""Verify the configured thinking model and Langfuse without market data or MCTS."""
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'src'))
from mcts_llm_alpha.config import load_config
from mcts_llm_alpha.llm import LLMClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    args = parser.parse_args()
    client = LLMClient.from_config(load_config(args.config).llm)
    try:
        if client.tracer.client and not client.tracer.client.auth_check():
            raise RuntimeError('Langfuse authentication failed')
        with client.tracer.observe('neuroii-connection-check', as_type='agent',
                                  input={'task': 'JSON arithmetic connection test'}) as span:
            answer = client.get_completion_sync(
                'Compute 137 * 29. Return only JSON with status (ok) and result (integer).',
                temperature=.6, name='connection-check')
            result = json.loads(answer)
            if result.get('result') != 3973:
                raise RuntimeError('Connection check returned an unexpected answer')
            span.update(output=result)
        print(json.dumps({'model': client.model, 'result': result,
            'call': client.transport.last_call, 'langfuse_trace': client.tracer.last_trace_url},
            ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
