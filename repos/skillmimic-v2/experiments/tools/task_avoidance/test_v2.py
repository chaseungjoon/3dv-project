"""CPU fixtures for v2 failures and experimental controls; not robot results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_cpu import runner, synthetic
from core import load_protocol, protocol_hash, summarize, unique_state_ids
from analyze import main as analyze_main, repeat_check, paired_rows
from analyze_capability import main as capability_main
from checkpoints import check_span


def fixture():
    _,z,initial=synthetic()
    p=load_protocol(runner.DEFAULT)
    p.update(expected_clip_frames=6,start_frame=2,num_envs=2,eval_seeds=[101],
             reach_window_steps=2,reach_hold_steps=1,capability_prefix_steps=[0,1])
    z['done'][:]=False;z['done'][-1]=True;z['train_stop'][:]=False;z['fall'][:]=False
    z['wrist_pos']=np.zeros((4,2,2,3))
    z['wrist_pos'][:,:,:,0]=np.array([.19,.17,.14,.12])[:,None,None]
    initial.update(eligible=np.ones(2,bool),segment_start_step=np.asarray(0),origin_target=initial['target'].copy(),
                   wrist_d=np.full(2,.19),clip_steps=np.full(2,4),local_actor_state=np.zeros((2,2,13)),
                   dof_pos=np.zeros((2,3)),dof_vel=np.zeros((2,3)),history=np.zeros((2,3)),
                   policy_input=np.zeros((2,3)),start_frames=np.full(2,2),progress=np.zeros(2))
    return p,z,initial


def save_job(root,name,p,z,init,label,seed=101,step=0,no_switch=False):
    folder=root/name;folder.mkdir()
    np.savez(folder/'initial.npz',**init);np.savez(folder/'trajectory.npz',**z)
    result=dict(status='COMPLETED',protocol_hash=protocol_hash(p),diagnostic_reward_match='PASS',
                checkpoint_sha256=label+'sha',prefix_sha256='donorsha',seed=seed,prefix_steps=step,
                control_no_switch=no_switch,episodes=summarize(z,init,p))
    (folder/'result.json').write_text(json.dumps(result,allow_nan=False))


def evaluation_fixture(root,labels=('e1000','e3000')):
    p,z,init=fixture();jobs=[]
    for label in labels:
        for repeat in [False,True]:
            name=label+'_s101'+('_repeat' if repeat else '')
            save_job(root,name,p,z,init,label)
            jobs.append(dict(job=name,policy=label,seed=101,repeat=repeat))
    (root/'protocol.json').write_text(json.dumps(p))
    (root/'manifest.json').write_text(json.dumps(dict(protocol_hash=protocol_hash(p),jobs=jobs,
        policies=[dict(label=k,sha256=k+'sha') for k in labels])))
    return p,z,init


def capability_fixture(root):
    p,z,init=fixture();jobs=[]
    for step in p['capability_prefix_steps']:
        tail={k:v[step:].copy() for k,v in z.items()}
        state={k:v.copy() for k,v in init.items()}
        state['segment_start_step']=np.asarray(step);state['clip_steps']=np.full(2,4-step)
        for case in ['direct','reload','candidate','candidate_repeat']:
            label='donor' if case in ['direct','reload'] else 'candidate'
            name=f'prefix{step}_{case}_s101'
            save_job(root,name,p,tail,state,label,step=step,no_switch=case=='direct')
            jobs.append(dict(job=name,policy=label,seed=101,case=case,prefix_steps=step,repeat=case=='candidate_repeat'))
    (root/'protocol.json').write_text(json.dumps(p))
    (root/'manifest.json').write_text(json.dumps(dict(kind='capability',protocol_hash=protocol_hash(p),jobs=jobs,
        policies=[dict(label=k,sha256=k+'sha') for k in ['donor','candidate']])))


class MetricsV2(unittest.TestCase):
    def test_reach_without_any_object_lift(self):
        p,z,initial=fixture();row=summarize(z,initial,p)[0]
        self.assertEqual(row['reach_attempt_proxy'],1)
        self.assertEqual(row['partial_lift'],0)
        self.assertEqual(row['reach_without_partial_lift'],1)
        self.assertNotIn('lift_attempt',row)

    def test_moving_object_toward_stationary_wrist_is_not_reach(self):
        p,z,initial=fixture();z['wrist_pos'][:,:,:,0]=.19
        z['obj'][:,:,0]=np.arange(4)[:,None]*.03
        self.assertEqual(summarize(z,initial,p)[0]['reach_attempt_proxy'],0)

    def test_post_terminal_reach_excluded(self):
        p,z,initial=fixture();z['done'][0]=True
        self.assertEqual(summarize(z,initial,p)[0]['reach_attempt_proxy'],0)

    def test_switch_eligibility_and_remaining_horizon(self):
        p,z,initial=fixture();initial['eligible'][0]=False
        initial['segment_start_step']=np.asarray(1);initial['clip_steps']=np.full(2,3)
        rows=summarize({k:v[1:] for k,v in z.items()},initial,p)
        self.assertEqual(rows[0]['eligible'],0);self.assertNotIn('task_success',rows[0])
        self.assertEqual(rows[1]['observed_steps'],3)

    def test_nonfinite_geometry_and_initial_rejected(self):
        for key in ['obj','ref_obj','wrist_pos','factors','action']:
            p,z,initial=fixture();z[key].flat[0]=np.nan
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Nonfinite'):summarize(z,initial,p)
        p,z,initial=fixture();initial['target'][0,0]=np.inf
        with self.assertRaises(ValueError):summarize(z,initial,p)

    def test_shape_mismatch_rejected(self):
        p,z,initial=fixture();z['wrist_pos']=np.zeros((4,2,3))
        with self.assertRaisesRegex(ValueError,'wrist_pos'):summarize(z,initial,p)

    def test_duplicate_state_fingerprints(self):
        _,_,initial=fixture();ids=unique_state_ids(initial,2);self.assertEqual(ids[0],ids[1])
        initial['history'][1,0]=1;ids=unique_state_ids(initial,2);self.assertNotEqual(ids[0],ids[1])

    def test_duplicate_episode_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):paired_rows([{'env_id':0},{'env_id':0}],[{'env_id':0}])


class AnalysisV2(unittest.TestCase):
    def test_v2_pair_and_duplicate_counts(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);evaluation_fixture(root)
            self.assertEqual(analyze_main(['--input',td]),0)
            report=json.loads((root/'analysis/report.json').read_text())
            self.assertEqual(report['initial_state_counts'][0]['unique_states'],1)
            self.assertEqual(report['hypothesis_status'],'NOT TESTED')

    def test_missing_pair_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);evaluation_fixture(root,('c0','c1'))
            self.assertEqual(analyze_main(['--input',td]),3)
            self.assertFalse((root/'analysis/paired_episode_differences.csv').exists())

    def test_explicit_pair_is_used(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);evaluation_fixture(root,('c0','c1'))
            self.assertEqual(analyze_main(['--input',td,'--left','c0','--right','c1']),0)

    def test_wrong_checkpoint_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);evaluation_fixture(root);path=root/'e3000_s101/result.json'
            data=json.loads(path.read_text());data['checkpoint_sha256']='wrong';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'identity'):analyze_main(['--input',td])

    def test_wrist_repeat_divergence_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,z,_=evaluation_fixture(root)
            z['wrist_pos'][0,0,0,0]+=.1;np.savez(root/'e3000_s101_repeat/trajectory.npz',**z)
            self.assertEqual(repeat_check(root/'e3000_s101',root/'e3000_s101_repeat',p)['status'],'INCONCLUSIVE')

    def test_capability_controls_and_tail_analysis(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);capability_fixture(root)
            self.assertEqual(capability_main(['--input',td]),0)
            self.assertTrue((root/'analysis/paired_differences.csv').is_file())
            report=json.loads((root/'analysis/report.json').read_text())
            self.assertEqual(len(report['controls']),4);self.assertEqual(report['hypothesis_status'],'NOT TESTED')

    def test_failed_handoff_control_blocks_paired_results(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);capability_fixture(root);path=root/'prefix1_reload_s101/trajectory.npz'
            with np.load(path) as archive:z=dict(archive)
            z['reward'][0,0]+=.1;np.savez(path,**z)
            self.assertEqual(capability_main(['--input',td]),3)
            self.assertFalse((root/'analysis/paired_differences.csv').exists())
            self.assertTrue((root/'analysis/episodes.csv').exists())

    def test_all_ineligible_repeat_inconclusive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,z,initial=fixture();initial['eligible'][:]=False
            for name in ['a','b']:save_job(root,name,p,z,initial,'candidate')
            self.assertEqual(repeat_check(root/'a',root/'b',p)['status'],'INCONCLUSIVE')


class OrchestrationV2(unittest.TestCase):
    def bundle_failure(self,side_effect,exception):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'result';p=load_protocol(runner.DEFAULT)
            with patch.object(runner.P,'RUNS',root/'lock'),patch.object(runner.P,'preflight'),\
                 patch.object(runner.P,'manifest_files',return_value={}),patch.object(runner.P,'capture',return_value='mock'),\
                 patch.object(runner.P,'execute_child',side_effect=side_effect),self.assertRaises(exception):
                runner.execute_bundle(out,p,[],[dict(job='one',command=['mock']),dict(job='two',command=['mock'])],'analyze.py')
            return json.loads((out/'manifest.json').read_text())

    def test_second_worker_launch_failure_never_completed(self):
        result=self.bundle_failure([0,OSError('launch failed')],OSError)
        self.assertEqual(result['status'],'FAILED');self.assertEqual(result['exit_code'],1)
        self.assertEqual(result['jobs'][1]['status'],'FAILED')

    def test_analysis_launch_failure_never_completed(self):
        self.assertEqual(self.bundle_failure([0,0,OSError('analyzer missing')],OSError)['status'],'FAILED')

    def test_interrupt_never_completed(self):
        self.assertEqual(self.bundle_failure([0,KeyboardInterrupt()],KeyboardInterrupt)['exit_code'],130)

    def test_strict_json_rejects_nan(self):
        with tempfile.TemporaryDirectory() as td,self.assertRaises(ValueError):
            runner.P.write_json(Path(td)/'result.json',{'value':float('nan')})

    def test_frame_accounting_exact(self):
        start=dict(epoch=1000,frame=65536000);end=dict(epoch=1020,frame=65536000+20*2048*32)
        self.assertEqual(check_span(start,end,1000,1020)['extra_frames'],1310720)
        end['frame']+=2048*32
        with self.assertRaisesRegex(ValueError,'delta'):check_span(start,end,1000,1020)

    def test_wrong_checkpoint_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError,'epoch'):check_span(dict(epoch=1000,frame=0),dict(epoch=1021,frame=1),1000,1020)

    def completed_target_resume(self,audit_exit=0):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);source=root/'source';(source/'nn').mkdir(parents=True)
            ck=source/'nn/p_00001000.pth';ck.write_text('checkpoint fixture')
            provenance=dict(method='fix',clip='place_book',num_envs=2048)
            (source/'run_meta.json').write_text(json.dumps(provenance))
            out=root/'runs';p=load_protocol(runner.DEFAULT);calls=[]
            with patch.object(runner,'OUT',out),patch.object(runner.P,'RUNS',root/'lock'),\
                 patch.object(runner.P,'preflight'),patch.object(runner.P,'manifest_files',return_value={}):
                for variant in ['baseline','progress']:
                    run=out/f'branch_test_smoke_{variant}_s0_from1000_plus20';(run/'nn').mkdir(parents=True)
                    (run/'nn/p_00001020.pth').write_text('final fixture')
                    signature=dict(source_sha256=runner.P.sha(ck),source_epoch=1000,protocol=p,variant=variant,
                        weight=0 if variant=='baseline' else p['progress_weight'],branch_seed=0,extra_epochs=20,files={},python=runner.P.python_path())
                    (run/'experiment.json').write_text(json.dumps(dict(status='RUNNING',fingerprint=protocol_hash(signature))))
                def execute(cmd,log,env):
                    calls.append(cmd)
                    Path(cmd[cmd.index('--output')+1]).write_text(json.dumps(dict(status='PASS',final_epoch=1020)))
                    return audit_exit
                with patch.object(runner.P,'execute_child',execute):
                    self.assertEqual(runner.main(['branch','--source',str(ck),'--label','test','--extra-epochs','20',
                        '--seeds','0','--execute','--resume']),audit_exit)
            self.assertEqual(len(calls),1 if audit_exit else 2)
            self.assertTrue(all(Path(cmd[1]).name=='checkpoints.py' for cmd in calls))
            run=out/'branch_test_smoke_baseline_s0_from1000_plus20'
            self.assertEqual(json.loads((run/'experiment.json').read_text())['status'],'FAILED' if audit_exit else 'COMPLETED')

    def test_completed_target_resume_audits_without_training(self):
        self.completed_target_resume()

    def test_failed_completion_audit_cannot_complete_or_train(self):
        self.completed_target_resume(audit_exit=7)

    def test_capability_and_gpu_verification_dry_runs(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()) as stream:
            root=Path(td);ck=root/'p_00001000.pth';ck.write_text('fake')
            self.assertEqual(runner.main(['capability','--source',str(ck),'--candidate',str(ck),'--label','cap','--dry-run']),0)
            text=stream.getvalue();self.assertEqual(text.count('evaluate_worker.py'),30)
            self.assertEqual(text.count('--control-no-switch'),9)
            self.assertEqual(runner.main(['verify','--checkpoint',str(ck),'--label','checks','--dry-run']),0)
            self.assertEqual([p.name for p in root.iterdir()],[ck.name])


if __name__=='__main__':unittest.main()
