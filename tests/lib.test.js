import test from 'node:test';
import assert from 'node:assert/strict';
import {Camera,AgentView,Bridge,makeExperiment,slurmScript,validateBackup} from '../web/lib.js';
const config={name:'test',task:'Reach',owner:'',recipe:'smolvla_sft',dataset:'lab/data',checkpoint:'lerobot/smolvla_base',steps:10,batch_size:8,seed:42,learning_rate:0.0001,rubric:'Visible completion',notes:''};
function stream(){const track={stopped:false,stop(){this.stopped=true},addEventListener(){}};return {track,getTracks:()=>[track],getVideoTracks:()=>[track]};}
test('camera is opt-in, video-only, selects a device and releases tracks',async()=>{const calls=[],first=stream(),second=stream();const c=new Camera({getUserMedia:async opts=>{calls.push(opts);return calls.length===1?first:second},enumerateDevices:async()=>[{kind:'videoinput',deviceId:'usb'},{kind:'audioinput'}]});assert.equal(calls.length,0);await c.start();assert.equal(calls[0].audio,false);assert.equal(c.devices.length,1);await c.start('usb');assert.equal(calls[1].video.deviceId.exact,'usb');assert.equal(first.track.stopped,true);c.stop();assert.equal(second.track.stopped,true);assert.equal(c.stream,null);});
test('late permission result is stopped after navigation/cancel',async()=>{let resolve;const s=stream();const c=new Camera({getUserMedia:()=>new Promise(r=>resolve=r)});const pending=c.start();c.stop();resolve(s);await pending;assert.equal(s.track.stopped,true);assert.equal(c.stream,null);});
test('permission failure does not fabricate a stream',async()=>{const c=new Camera({getUserMedia:async()=>{throw {name:'NotAllowedError'}}});await assert.rejects(c.start(),/permission was declined/);assert.equal(c.stream,null);});
test('capture requires a running video and produces bounded frames',()=>{const c=new Camera({});assert.throws(()=>c.capture({videoWidth:1280},{}));c.stream=stream();const canvas={getContext:()=>({drawImage(){}}),toDataURL:()=> 'data:image/jpeg;base64,abc'};const frame=c.capture({videoWidth:1920,videoHeight:1080},canvas);assert.equal(canvas.width,800);assert.equal(canvas.height,450);assert.ok(frame.captured_at>0);});
test('config rejects nonfinite values, bad types, and unknown recipes',()=>{for(const changes of [{learning_rate:NaN},{steps:1.5},{batch_size:true},{recipe:'shell'},{name:''}])assert.throws(()=>makeExperiment({...config,...changes}));});
test('Slurm export applies hyperparameters and quotes user strings',()=>{const exp=makeExperiment({...config,dataset:"lab/a'; touch /tmp/should-not-exist; '"});const script=slurmScript(exp);assert.ok(script.includes('--policy.optimizer_lr=0.0001'));assert.ok(script.includes("'\"'\"'"));assert.ok(script.includes('set -euo pipefail'));assert.throws(()=>slurmScript({...exp,recipe:'pi05_rl'}));});
test('bridge only accepts loopback HTTP or authenticated HTTPS destination',()=>{for(const url of ['http://public.test','https://user:pass@example.test','file:///tmp/a','https://a.test/?token=x'])assert.throws(()=>new Bridge(url,'a'));assert.equal(new Bridge('http://127.0.0.1:8787','a').url,'http://127.0.0.1:8787');});
test('backup rejects broken links and malformed renderer records',()=>{const exp=makeExperiment(config);const data={format:'armature-backup-v1',library:{experiments:[exp],runs:[],episodes:[],sessions:[]}};assert.equal(validateBackup(data).experiments.length,1);data.library.runs=[{id:'b'.repeat(32),created_at:1,experiment_id:exp.id,status:'completed',logs:null,metrics:[]}];assert.throws(()=>validateBackup(data));data.library.runs=[];data.library.sessions=[{id:'c'.repeat(32),experiment_id:'f'.repeat(32),created_at:1,mode:'sandbox',history:[]}];assert.throws(()=>validateBackup(data));});
test('agent view shares only after explicit start and revokes sharing on stop',async()=>{
 const calls=[];let tick;
 const bridge={request:async(path,body)=>{calls.push({path,body});return path==='camera/start'?{lease:'lease-a'}:{};}};
 const view=new AgentView(()=>{},()=>{},callback=>{tick=callback;return 1;},()=>{tick=null;});
 assert.equal(calls.length,0);
 await view.start(bridge,()=>({captured_at:1,data_url:'test'}));
 assert.equal(view.active,true);assert.deepEqual(calls.map(c=>c.path),['camera/start','camera/frame']);
 await tick();assert.equal(calls.filter(c=>c.path==='camera/frame').length,2);
 await view.stop();assert.equal(view.active,false);assert.equal(tick,null);assert.equal(calls.at(-1).path,'camera/stop');
});
test('late camera sharing activation is revoked after navigation',async()=>{
 let resolveStart;const calls=[];
 const bridge={request:async(path,body)=>{calls.push(path);if(path==='camera/start')return new Promise(r=>resolveStart=r);return {};}};
 const view=new AgentView();const pending=view.start(bridge,()=>{throw Error('must not capture');});
 await Promise.resolve();assert.ok(resolveStart);
 await view.stop();resolveStart({lease:'late'});await pending;
 assert.equal(view.active,false);assert.deepEqual(calls,['camera/start','camera/stop']);
});
test('camera frame failure ends sharing and cancels future uploads',async()=>{
 const calls=[];let tick,error;
 const bridge={request:async path=>{calls.push(path);return path==='camera/start'?{lease:'a'}:{};}};
 let count=0;
 const view=new AgentView(()=>{},e=>{error=e},cb=>{tick=cb;},()=>{tick=null;});
 await view.start(bridge,()=>{if(count++)throw Error('frozen frame');return {};});
 await tick();assert.equal(view.active,false);assert.match(error.message,/frozen/);assert.equal(tick,null);assert.equal(calls.at(-1),'camera/stop');
});
