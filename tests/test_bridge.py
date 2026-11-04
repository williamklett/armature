import base64
import functools
import http.client
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from bridge.core import Store, Sandbox, experiment
from bridge.server import App, Handler
from bridge.jobs import Jobs
from bridge.grader import grade, validate_frames
from bridge.robot import PhysicalArm, JOINTS
from scripts.lerobot_train import command, metric_from_log

FRAME={'data_url':'data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\ntest').decode(),'captured_at':1}


def exp(**kwargs):
    return experiment({'recipe':'toy_q','name':'test','task':'reach','steps':10,**kwargs})


def wait_run(store, rid):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        run=store.get('runs',rid)
        if run['status'] not in ('running','starting','stopping'):
            return run
        time.sleep(.03)
    raise AssertionError('Run did not finish')


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.temp.name)/'test.sqlite')
    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()
    def test_persistence_and_config_validation(self):
        e=self.store.put('experiments',exp())
        reopened=Store(Path(self.temp.name)/'test.sqlite')
        self.assertEqual(reopened.get('experiments',e['id']),e)
        reopened.db.close()
        for data in ({'seed':True},{'steps':float('nan')},{'batch_size':0},{'recipe':'command'}):
            with self.assertRaises(ValueError): exp(**data)
    def test_sandbox_replay_stopped_and_bounds(self):
        e=self.store.put('experiments',exp())
        robot=Sandbox(self.store)
        with self.assertRaises(ValueError): robot.action({'action':'move'})
        o=robot.action({'action':'start','experiment_id':e['id']})
        robot.action({'action':'move','revision':o['revision'],'dx':.05})
        with self.assertRaises(ValueError): robot.action({'action':'move','revision':o['revision'],'dx':.05})
        o=robot.observe()
        with self.assertRaises(ValueError): robot.action({'action':'move','revision':o['revision'],'dx':1})
        robot.action({'action':'stop'})
        with self.assertRaises(ValueError): robot.action({'action':'move','revision':robot.observe()['revision']})
        self.assertTrue(self.store.all('sessions')[0]['state']['stopped'])
    def test_actual_training_lifecycle(self):
        jobs=Jobs(self.store,self.temp.name,{})
        try:
            run=jobs.launch(exp())
            done=wait_run(self.store,run['id'])
            self.assertEqual(done['status'],'completed')
            self.assertEqual(done['exit_code'],0)
            self.assertEqual(done['metrics'][-1]['step'],10)
            self.assertTrue((Path(self.temp.name)/'runs'/run['id']/'stdout.log').exists())
            cancelled=jobs.launch(exp(steps=500))
            with self.assertRaises(ValueError): jobs.launch(exp())
            jobs.stop(cancelled['id'])
            self.assertEqual(wait_run(self.store,cancelled['id'])['status'],'cancelled')
        finally:
            jobs.close()
    def test_shutdown_cancels_child(self):
        jobs=Jobs(self.store,self.temp.name,{})
        run=jobs.launch(exp(steps=500))
        jobs.close()
        self.assertIsNotNone(jobs.processes[run['id']].poll())
        self.assertEqual(self.store.get('runs',run['id'])['status'],'cancelled')
    def test_argv_values_are_not_shell_commands(self):
        jobs=Jobs(self.store,self.temp.name,{'runners':{'smolvla_sft':{'command':[os.sys.executable,'-c','import sys; print(sys.argv[1])','{dataset}']}}})
        payload='lab/data; echo injected'
        try:
            done=wait_run(self.store,jobs.launch(exp(recipe='smolvla_sft',dataset=payload))['id'])
            self.assertEqual(done['logs'],[payload])
        finally: jobs.close()
    def test_sft_argv_and_metric_extraction(self):
        e=exp(recipe='pi05_sft',dataset='lab/cubes',learning_rate=2.5e-5)
        cmd=command(e,'/tmp/out path','cuda')
        self.assertIn('--dataset.repo_id=lab/cubes',cmd)
        self.assertIn('--policy.optimizer_lr=2.5e-05',cmd)
        self.assertIn('--trackio.enable=false',cmd)
        self.assertIn('--output_dir=/tmp/out path',cmd)
        self.assertEqual(metric_from_log('INFO step:1K smpl:8K loss:0.024'),{'step':1000,'loss':.024})


class GraderTests(unittest.TestCase):
    def test_frame_validation(self):
        self.assertEqual(len(validate_frames([FRAME])[0]['sha256']),64)
        for frames in ([],[None],[{'data_url':'data:image/svg+xml;base64,abc'}],[{'data_url':'data:image/png;base64,@@@@'}]):
            with self.assertRaises(ValueError): validate_frames(frames)
    @patch.dict(os.environ,{'OPENAI_API_KEY':'unit-test-key','ARMATURE_GRADER_MODEL':'test-model'})
    def test_structured_grade_and_unclear(self):
        captured=[]
        def opener(req,timeout):
            captured.append(json.loads(req.data))
            return io.BytesIO(json.dumps({'status':'completed','id':'response-test','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'success':None,'score':.9,'confidence':.2,'reason':'Occluded','evidence':['Object hidden']})}]}]}).encode())
        g=grade({'frames':[FRAME]},exp(),opener)
        self.assertIsNone(g['score'])
        self.assertEqual(g['model'],'test-model')
        self.assertFalse(captured[0]['store'])
        self.assertTrue(captured[0]['text']['format']['strict'])
        self.assertEqual(captured[0]['input'][0]['content'][1]['type'],'input_image')
    @patch.dict(os.environ,{'OPENAI_API_KEY':'unit-test-key','ARMATURE_GRADER_MODEL':'test-model'})
    def test_refusal_is_not_reward(self):
        for body in ({'status':'incomplete'},{'status':'completed','output':[]}):
            with self.assertRaises(ValueError): grade({'frames':[FRAME]},exp(),lambda *a,**k:io.BytesIO(json.dumps(body).encode()))


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.app=App(cls.temp.name,{'allowed_origins':['https://lab.example']})
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,app=cls.app))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
        cls.app.jobs.close();cls.app.arm.close();cls.app.store.db.close();cls.temp.cleanup()
    def request(self,path,method='GET',body=None,headers=None,auth=True):
        h={'Content-Type':'application/json',**(headers or {})}
        if auth: h['Authorization']='Bearer '+self.app.token
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        c.request(method,path,json.dumps(body) if body is not None else None,h)
        r=c.getresponse();result=(r.status,dict(r.getheaders()),r.read());c.close();return result
    def test_auth_and_origin(self):
        self.assertEqual(self.request('/api/library',auth=False)[0],401)
        self.assertEqual(self.request('/api/library',headers={'Origin':'https://evil.example'})[0],403)
        result=self.request('/api/library',headers={'Origin':'https://lab.example'})
        self.assertEqual(result[0],200)
        self.assertEqual(result[1]['Access-Control-Allow-Origin'],'https://lab.example')
        self.assertEqual(self.request('/api/library',headers={'Host':'evil.example'})[0],403)
    def test_pairing_same_origin_only(self):
        self.assertEqual(self.request('/api/pair',auth=False)[0],403)
        result=self.request('/api/pair',headers={'Sec-Fetch-Site':'same-origin'},auth=False)
        self.assertEqual(result[0],200)
        self.assertNotIn('Access-Control-Allow-Origin',result[1])
        self.assertEqual(self.request('/api/pair',headers={'Sec-Fetch-Site':'cross-site','Origin':'https://lab.example'})[0],403)
    def test_static_and_body_validation(self):
        self.assertEqual(self.request('/')[0],200)
        self.assertEqual(self.request('/../bridge/server.py')[0],404)
        self.assertEqual(self.request('/api/experiments','POST',[])[0],400)
        self.assertEqual(self.request('/api/experiments','POST',{},headers={'Content-Type':'text/plain'})[0],415)
    def test_experiment_episode_review_flow(self):
        result=self.request('/api/experiments','POST',{'name':'test','task':'reach','recipe':'toy_q'})
        self.assertEqual(result[0],200)
        eid=json.loads(result[2])['id']
        result=self.request('/api/episodes','POST',{'experiment_id':eid,'frames':[FRAME],'notes':'test'})
        self.assertEqual(result[0],200)
        episode=json.loads(result[2]);self.assertIsNone(episode['human_success'])
        reviewed=self.request('/api/episodes/'+episode['id']+'/review','POST',{'human_success':False})
        self.assertFalse(json.loads(reviewed[2])['human_success'])
        self.assertEqual(self.request('/api/arm/action','POST',{'action':'connect'})[0],400)


class FakeArm:
    def __init__(self):
        self.is_connected=False;self.is_calibrated=True;self.positions={j+'.pos':50 if j=='gripper' else 0 for j in JOINTS};self.commands=[]
    def connect(self,calibrate):
        assert calibrate
        self.is_connected=True
    def disconnect(self): self.is_connected=False
    def get_observation(self): return dict(self.positions)
    def send_action(self,target): self.commands.append(target);self.positions.update(target)


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=Store(Path(self.temp.name)/'arm.sqlite')
        self.e=self.store.put('experiments',exp())
        self.fake=FakeArm()
        self.config={'enabled':True,'port':'test','id':'test','calibration_dir':'test','limits':{j:[0,100] if j=='gripper' else [-10,10] for j in JOINTS}}
        self.arm=PhysicalArm(self.store,self.config,lambda _:self.fake,watchdog=False)
    def tearDown(self): self.arm.close();self.store.db.close();self.temp.cleanup()
    def test_opt_in_and_calibration_gate(self):
        self.assertFalse(self.fake.is_connected)
        self.fake.is_calibrated=False
        with self.assertRaises(ValueError): self.arm.action({'action':'connect'})
        self.assertFalse(self.fake.is_connected)
    def test_joint_limits_replay_and_stop(self):
        self.arm.action({'action':'connect'})
        o=self.arm.action({'action':'start','experiment_id':self.e['id']})
        self.arm.last_action=time.monotonic()-1
        moved=self.arm.action({'action':'jog','joint':'shoulder_pan','delta':1,'revision':o['revision']})
        self.assertEqual(moved['positions']['shoulder_pan'],1)
        with self.assertRaises(ValueError): self.arm.action({'action':'jog','joint':'shoulder_pan','delta':1,'revision':o['revision']})
        self.arm.last_action=time.monotonic()-1
        with self.assertRaises(ValueError): self.arm.action({'action':'jog','joint':'shoulder_pan','delta':20,'revision':moved['revision']})
        self.arm.action({'action':'stop'})
        with self.assertRaises(ValueError): self.arm.action({'action':'jog','joint':'shoulder_pan','delta':1,'revision':self.arm.revision})
        self.assertTrue(self.store.all('sessions')[0]['state']['stopped'])
    def test_no_jog_without_configuration(self):
        self.arm.config['enabled']=False
        with self.assertRaises(ValueError): self.arm.action({'action':'connect'})
        self.assertFalse(self.fake.is_connected)


if __name__=='__main__': unittest.main()
