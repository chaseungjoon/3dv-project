"""Validate the GPU check result identity before accepting its exit status."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);a=p.parse_args()
m=json.loads((a.input/'manifest.json').read_text());job=m['jobs'][0]
r=json.loads((a.input/job['job']/'result.json').read_text())
if r['protocol_hash']!=m['protocol_hash'] or r['checkpoint_sha256']!=m['policies'][0]['sha256']:
    raise ValueError('GPU check result identity mismatch')
if r['status']!='PASS':raise ValueError('GPU checks did not pass')
