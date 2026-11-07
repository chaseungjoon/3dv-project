import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from core import *
from analyze import paired_rows, repeat_check, main as analyze_main
spec=importlib.util.spec_from_file_location('avoidance_runner',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def synthetic():
    p=load_protocol(runner.DEFAULT);p.update(expected_clip_frames=6,start_frame=2,num_envs=2)
    T,N=4,2
    z=dict(reward=np.ones((T,N)),done=np.array([[0,0],[1,0],[0,0],[0,1]],bool),
           train_stop=np.array([[0,0],[0,1],[0,0],[0,0]],bool),fall=np.array([[0,0],[1,0],[0,0],[0,0]],bool),
           obj=np.zeros((T,N,3)),ref_obj=np.zeros((T,N,3)),wrist_d=np.full((T,N),.1),
           factors=np.ones((T,N,len(FACTOR_NAMES))),action=np.zeros((T,N,2)))
    init={'target':np.zeros((N,13)),'goal':np.array([1.,0,0]),'reference_lift':np.array(.1)}
    return p,z,init


class MetricsTests(unittest.TestCase):
    def test_terminal_included_post_terminal_excluded(self):
        p,z,init=synthetic();v,t=transition_masks(z['done'],z['train_stop'])
        np.testing.assert_array_equal(v[:,0],[1,1,0,0]);np.testing.assert_array_equal(t[:,1],[1,1,0,0])
        r=summarize(z,init,p)
        self.assertAlmostEqual(r[0]['return_eval'],1+p['gamma'])
        self.assertAlmostEqual(r[1]['return_train_window'],1+p['gamma'])
        self.assertEqual(r[0]['task_success'],0);self.assertEqual(r[1]['task_success'],1)
    def test_joint_success_not_mixed_across_episodes(self):
        p,z,init=synthetic();z['obj'][0,0,2]=.08;z['ref_obj'][0,0,2]=.08
        rows=summarize(z,init,p)
        self.assertEqual(rows[0]['lift_success'],1);self.assertEqual(rows[0]['stable_lift'],0)
        self.assertEqual(rows[1]['lift_success'],0);self.assertEqual(rows[1]['stable_lift'],0)
    def test_goal_progress_initial_not_first_poststep(self):
        p,z,init=synthetic();z['obj'][:,:,0]=.25
        self.assertAlmostEqual(summarize(z,init,p)[0]['goal_progress_m'],.25)
    def test_initial_history_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError,'history'):
            assert_initial_equal({'history':np.zeros((2,3))},{'history':np.ones((2,3))})
    def test_nonfinite_initial_fails(self):
        with self.assertRaises(ValueError):assert_initial_equal({'x':np.array([np.nan])},{'x':np.array([np.nan])})
    def test_pair_uses_ids_not_order(self):
        base=[dict(env_id=e,**{m:float(e) for m in __import__('analyze').METRICS}) for e in [0,1]]
        other=[dict(env_id=e,**{m:float(e+2) for m in __import__('analyze').METRICS}) for e in [1,0]]
        self.assertEqual(paired_rows(base,other)[0]['return_eval'],2)
    def test_diagnostic_ast_matches_source_product(self):
        class TorchNumpy:
            mean=staticmethod(np.mean);zeros_like=staticmethod(np.zeros_like);exp=staticmethod(np.exp)
            @staticmethod
            def stack(xs,dim):return np.stack(xs,axis=dim)
            @staticmethod
            def cat(xs,dim):return np.concatenate(xs,axis=dim)
        # Actual tensor function needs Tensor.view/transpose semantics. Inspect exact arithmetic AST instead.
        source=runner.ROOT/'skillmimic/env/tasks/skillmimic_parahome.py'
        fn=build_factor_function(source,{'torch':TorchNumpy})
        self.assertTrue(callable(fn));self.assertEqual(fn.__name__,'_diagnostic_factors')
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_humanoid_reward')
        assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reward' for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value),'rb * ro * rig')
    def test_repeat_divergence_inconclusive(self):
        p,z,init=synthetic()
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a';b=Path(td)/'b';a.mkdir();b.mkdir()
            np.savez(a/'trajectory.npz',**z);z['reward'][0,0]+=1;np.savez(b/'trajectory.npz',**z)
            self.assertEqual(repeat_check(a,b,p)['status'],'INCONCLUSIVE')


class RunnerTests(unittest.TestCase):
    def test_protocol_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p=load_protocol(runner.DEFAULT);p['eval_seeds']=[1,1]
            path=Path(td)/'p.json';path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_protocol(path)
    def test_missing_epochs_fails(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'run_meta.json').write_text(json.dumps(dict(clip='place_book',method='fix')))
            with self.assertRaisesRegex(ValueError,'Missing requested'):
                runner.plan_evaluation(SimpleNamespace(run=str(r)),load_protocol(runner.DEFAULT))
    def test_branch_dry_run_from_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='avoidance space ') as td:
            r=Path(td);(r/'nn').mkdir();ck=r/'nn/policy_00001000.pth';ck.write_text('fake')
            (r/'run_meta.json').write_text(json.dumps(dict(clip='place_book',method='fix',num_envs=2048)))
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                code=runner.main(['branch','--source',str(ck),'--label','test','--extra-epochs','20','--seeds','0','--dry-run'])
            self.assertEqual(code,0);self.assertIn('branch_entry.py',out.getvalue())
            self.assertIn('--max_epochs 1019',out.getvalue());self.assertEqual(out.getvalue().count('BRANCH '),2)
            self.assertEqual(sorted(x.name for x in r.iterdir()),['nn','run_meta.json'])
    def test_branch_requires_execute(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'nn').mkdir();ck=r/'nn/policy_00001000.pth';ck.write_text('fake')
            (r/'run_meta.json').write_text(json.dumps(dict(clip='place_book',method='fix',num_envs=2048)))
            with self.assertRaisesRegex(ValueError,'--execute'):
                runner.main(['branch','--source',str(ck),'--label','test','--extra-epochs','20'])
    def test_eval_fresh_process_reference_commands(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);cks=[]
            for e in [1000,3000]:
                ck=r/f'p_{e:08}.pth';ck.write_text('fake');cks.append(str(ck))
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(runner.main(['evaluate','--checkpoints',*cks,'--label','test','--dry-run']),0)
            self.assertEqual(out.getvalue().count('evaluate_worker.py'),7)
            self.assertIn('--reference-initial',out.getvalue())
    def test_failure_stops_fresh_evaluation_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);ck=r/'p_00001000.pth';ck.write_text('fake')
            calls=[]
            def fake(cmd,log,env):calls.append(cmd);return 7
            with patch.object(runner,'OUT',r/'out'),patch.object(runner.P,'RUNS',r/'lock'),\
                 patch.object(runner.P,'preflight'),patch.object(runner.P,'manifest_files',return_value={}),\
                 patch.object(runner.P,'capture',return_value='fake'),patch.object(runner.P,'execute_child',fake),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(['evaluate','--checkpoints',str(ck),'--label','fail']),7)
            self.assertEqual(len(calls),1)
            manifest=json.loads((r/'out/eval_fail/manifest.json').read_text())
            self.assertEqual(manifest['status'],'FAILED');self.assertEqual(manifest['exit_code'],7)

    def test_actual_resampled_reference_window(self):
        p,z,init=synthetic();init['clip_steps']=np.array([2,4]);init['start_frames']=np.array([4,2])
        z['fall'][:]=False
        rows=summarize(z,init,p)
        self.assertEqual(rows[0]['task_success'],1)
        self.assertIsNone(rows[0]['first_tracking_failure_step'])
        self.assertEqual(rows[0]['actual_start_frame'],4)

    def test_invalid_label(self):
        with self.assertRaises(ValueError):runner.guards('../bad')


class AnalysisTests(unittest.TestCase):
    def test_end_to_end_synthetic_pair(self):
        p,z,init=synthetic();p['eval_seeds']=[101];policies=[dict(label='e1000'),dict(label='e3000')]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'protocol.json').write_text(json.dumps(p));jobs=[]
            for label,repeat in [('e1000',False),('e3000',False),('e1000',True)]:
                name=label+'_s101'+('_repeat' if repeat else '');folder=root/name;folder.mkdir()
                np.savez(folder/'initial.npz',**init);np.savez(folder/'trajectory.npz',**z)
                result=dict(status='COMPLETED',protocol_hash=protocol_hash(p),diagnostic_reward_match='PASS',episodes=summarize(z,init,p))
                (folder/'result.json').write_text(json.dumps(result));jobs.append(dict(job=name,policy=label,seed=101,repeat=repeat))
            (root/'manifest.json').write_text(json.dumps(dict(protocol_hash=protocol_hash(p),policies=policies,jobs=jobs)))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(analyze_main(['--input',str(root)]),0)
            report=json.loads((root/'analysis/report.json').read_text())
            self.assertEqual(report['paired_comparison'],'AVAILABLE');self.assertEqual(report['status'],'PASS')
            with self.assertRaises(ValueError):analyze_main(['--input',str(root)])


if __name__=='__main__':unittest.main()
