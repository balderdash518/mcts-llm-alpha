"""Read the current run back through Langfuse's supported observations API."""
import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    folder=args.run_directory.resolve()
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    start=datetime.fromisoformat(manifest['started_at'])-timedelta(minutes=1)
    end=datetime.now(timezone.utc)+timedelta(minutes=1)
    env_path=Path(__file__).resolve().parent/'.env'
    result=subprocess.run(['npx.cmd','--yes','@langfuse/cli@1.2.4','--env',str(env_path),
        'api','observations','list','--session-id',manifest['run_id'],
        '--from-start-time',start.isoformat(),'--to-start-time',end.isoformat(),
        '--fields','core,basic,io,model,usage,metadata,trace_context',
        '--all','--max-items','1000','--json'],capture_output=True,text=True,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Langfuse CLI failed with exit code {result.returncode}')
    envelope=json.loads(result.stdout)
    if envelope['status'] != 200:
        raise RuntimeError(f"Langfuse API returned {envelope['status']}")
    rows=envelope['body']['data']
    (folder/'langfuse_observations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    generations=[r for r in rows if r['type']=='GENERATION']
    output={'observations':len(rows),'names':dict(Counter(r['name'] for r in rows)),
        'types':dict(Counter(r['type'] for r in rows)),
        'model_calls':len(generations),'total_tokens':sum(r.get('usageDetails',{}).get('total',0) for r in generations),
        'generation_models':sorted({r.get('model') for r in generations}),
        'error_observations':[{'name':r['name'],'status':r.get('statusMessage')} for r in rows if r.get('level')=='ERROR'],
        'trace_ids':sorted({r['traceId'] for r in rows}),'session_id':manifest['run_id'],
        'trace_url':manifest['trace_url']}
    assert rows,'No observations were ingested for the experiment'
    assert all(r.get('sessionId')==manifest['run_id'] for r in rows)
    assert any(r['name']=='run-alpha-experiment' for r in rows)
    if manifest['status']=='completed':
        assert any(r['name']=='mcts-iteration' for r in rows),'MCTS iteration trace is absent'
        assert any(r['name']=='decide-repository-admission' for r in rows)
        assert generations and all(r.get('model')==manifest['model'] for r in generations)
        assert all(r.get('input') and r.get('output') and r.get('usageDetails') for r in generations)
    (folder/'langfuse_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
