"""Run with python3 -m unittest discover -s experiments/tools/progress_reward -p test_cpu.py."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runner
spec=importlib.util.spec_from_file_location('progress_math',runner.ROOT/'skillmimic/utils/object_progress.py')
mathmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mathmod)


class RewardTests(unittest.TestCase):
    def test_approach_retreat_static_and_clip(self):
        out=mathmod.normalized_progress(np.array([1.,1.,1.,1.]),np.array([.9,1.1,1.,0.]),1.,.2)
        np.testing.assert_allclose(out,[.1,-.1,0,.2])
    def test_unclipped_round_trip(self):
        d=np.array([1.,.9,.8,.9,1.])
        self.assertAlmostEqual(mathmod.normalized_progress(d[:-1],d[1:],1.,.5).sum(),0)
    def test_zero_weight(self):
        base=np.array([.1,.3,.8]);extra=mathmod.normalized_progress(np.ones(3),np.zeros(3),1.,.05)
        np.testing.assert_array_equal(base+0*extra,base)
    def test_bad_length(self):
        with self.assertRaises(ValueError):mathmod.normalized_progress(np.ones(1),np.zeros(1),0,.05)


class RunnerTests(unittest.TestCase):
    def test_paths_and_arguments(self):
        from types import SimpleNamespace
        a=SimpleNamespace(num_envs=2048,epochs=20,seed=0)
        cmd=runner.command(a,Path('/tmp/a path/run'))
        self.assertIn('/tmp/a path/run'.split('/')[-1],cmd)
        self.assertEqual(cmd[cmd.index('--max_epochs')+1],'20')
    def test_failure_propagates_and_stops_compare(self):
        calls=[]
        def fake(a):calls.append((a.seed,a.variant));return 7
        with patch.object(runner,'train',fake):
            self.assertEqual(runner.main(['compare','--seeds','0,1']),7)
        self.assertEqual(calls,[(0,'baseline')])
    def test_compare_order(self):
        calls=[]
        def fake(a):calls.append((a.seed,a.variant));return 0
        with patch.object(runner,'train',fake):runner.main(['compare','--seeds','0,1','--dry-run'])
        self.assertEqual(calls,[(0,'baseline'),(0,'progress'),(1,'baseline'),(1,'progress')])
    def test_real_fake_child_exit_code(self):
        with tempfile.TemporaryDirectory() as t:
            code=runner.execute_child([sys.executable,'-c','print("fake-training");raise SystemExit(7)'],Path(t)/'log',os.environ.copy())
            self.assertEqual(code,7)
            self.assertIn('fake-training',(Path(t)/'log').read_text())
    def test_main_requires_explicit_execution(self):
        with self.assertRaises(ValueError):runner.main(['train','--stage','main'])
    def test_invalid_label_and_weight(self):
        for args in [['--label','../bad'],['--weight','nan'],['--num-envs','3']]:
            with self.assertRaises(ValueError):runner.main(['train','--dry-run']+args)
    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as t,patch.object(runner,'RUNS',Path(t)/'runs'):
            self.assertEqual(runner.main(['train','--dry-run']),0)
            self.assertFalse((Path(t)/'runs').exists())
    def test_collection_preserves_and_excludes_checkpoint(self):
        import tarfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);r=root/'run';r.mkdir();(r/'a.pth').write_text('ckpt');(r/'log').write_text('log')
            runner.main(['collect','--run',str(r),'--output',str(root/'out.tgz')])
            with tarfile.open(root/'out.tgz') as f:self.assertEqual(f.getnames(),['run/log'])
            self.assertTrue((r/'a.pth').exists())
            with self.assertRaises(FileExistsError):runner.main(['collect','--run',str(r),'--output',str(root/'out.tgz')])
    def test_resume_fingerprint_and_existing_run(self):
        # Exercise guard paths without GPU or launching training.
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as t,patch.object(runner,'RUNS',Path(t)),patch.object(runner,'preflight'),patch.object(runner,'gpu_lock',nullcontext),patch.object(runner,'manifest_files',return_value={}):
            run=Path(t)/'book_pilot_smoke_baseline_w0_s0_n2048_e20_scratch';run.mkdir()
            with self.assertRaisesRegex(ValueError,'already exists'):runner.main(['train'])
            (run/'experiment.json').write_text(json.dumps({'fingerprint':'different'}))
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):runner.main(['train','--resume'])


if __name__=='__main__':unittest.main()
