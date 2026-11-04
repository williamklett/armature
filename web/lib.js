export const recipes = {
  smolvla_sft:{name:'SmolVLA · SFT',model:'SmolVLA',kind:'Imitation learning',environment:'Recorded dataset',checkpoint:'lerobot/smolvla_base',description:'Build a lightweight policy from your demonstrations. A practical first model for the SO-ARM101.'},
  pi05_sft:{name:'π0.5 · SFT',model:'π0.5',kind:'Imitation learning',environment:'Recorded dataset',checkpoint:'lerobot/pi05_base',description:'Adapt a larger pretrained VLA on Adroit before moving into reinforcement learning.'},
  smolvla_rl:{name:'SmolVLA · experimental GRPO',model:'SmolVLA',kind:'RL · simulation',environment:'LIBERO',checkpoint:'',description:'Experimental FlowVLA-RL integration. Requires its checkout and a matching SFT checkpoint; physical SO101 RL is not included.'},
  pi05_rl:{name:'π0.5 · PPO',model:'π0.5',kind:'RL · simulation',environment:'LIBERO',checkpoint:'',description:'Use a configured RLinf runner for simulation RL. The real-arm task adapter is a separate integration.'},
  agent_session:{name:'Agent session',model:'Agent',kind:'In-context exploration',environment:'Control room',checkpoint:'',description:'Observe, act, and keep a record of what worked. Session context changes; model weights do not.'},
  toy_q:{name:'Q-learning · test',model:'Tabular Q-learning',kind:'RL · toy test',environment:'1D reach',checkpoint:'',description:'Run a small, real CPU learning job to test logs and charts. No VLA, camera, or robot hardware is used.'},
};
export const rubricDefault='The requested task is visibly complete. The object is stable and released by the gripper. Mark unclear if the frames do not establish completion.';
export const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const uid=()=>crypto.randomUUID().replaceAll('-','');
export function validateExperiment(d){
  if(!recipes[d.recipe])throw Error('Choose a recipe.');
  for(const [key,max] of [['name',120],['task',2000],['rubric',4000]])if(typeof d[key]!=='string'||!d[key].trim()||d[key].length>max)throw Error(`${key} is required (up to ${max} characters).`);
  for(const [key,low,high] of [['seed',0,2147483647],['steps',1,10000000],['batch_size',1,4096],['learning_rate',1e-8,1]])if(typeof d[key]!=='number'||!Number.isFinite(d[key])||d[key]<low||d[key]>high||(key!=='learning_rate'&&!Number.isInteger(d[key])))throw Error(`Check ${key.replaceAll('_',' ')}.`);
  for(const [key,max] of [['owner',80],['dataset',300],['checkpoint',500],['notes',5000]])if(typeof d[key]!=='string'||d[key].length>max||d[key].includes('\0'))throw Error(`Check ${key}.`);
  return d;
}
export function makeExperiment(d){return {...validateExperiment(d),id:uid(),created_at:Date.now()/1000};}
export function shellQuote(s){return "'"+String(s).replaceAll("'","'\"'\"'")+"'";}
export function slurmScript(exp){
  if(!['smolvla_sft','pi05_sft'].includes(exp.recipe))throw Error('Slurm export currently supports the two demonstration-training recipes. Configure RL through the runner integration.');
  if(!exp.dataset)throw Error('Add a dataset when creating or cloning this experiment.');
  const args=[`--policy.path=${exp.checkpoint||recipes[exp.recipe].checkpoint}`,`--dataset.repo_id=${exp.dataset}`,`--batch_size=${exp.batch_size}`,`--steps=${exp.steps}`,`--seed=${exp.seed}`,`--policy.optimizer_lr=${exp.learning_rate}`,'--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false'];
  return `#!/usr/bin/env bash\n#SBATCH --job-name=armature-${exp.recipe}\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n#SBATCH --time=04:00:00\n#SBATCH --output=armature-%j.log\n\nset -euo pipefail\n# Set ARMATURE_VENV to the tested environment before sbatch.\n: "\${ARMATURE_VENV:?Set ARMATURE_VENV to your LeRobot virtual environment}"\n: "\${ARMATURE_OUTPUT_ROOT:?Set ARMATURE_OUTPUT_ROOT to your scratch output directory}"\nsource "$ARMATURE_VENV/bin/activate"\n# Camera/action transforms must match your recorded dataset.\nlerobot-train \\\n  ${args.map(shellQuote).join(' \\\n  ')} \\\n  --output_dir="$ARMATURE_OUTPUT_ROOT/${exp.id}-\${SLURM_JOB_ID}"\n`;
}
export function download(name,value,type='application/json'){
  const blob=new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
export class Library{
  async open(){this.db=await new Promise((resolve,reject)=>{const req=indexedDB.open('armature-v1',1);req.onupgradeneeded=()=>{for(const kind of ['experiments','runs','episodes','sessions'])req.result.createObjectStore(kind,{keyPath:'id'});};req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});return this;}
  async all(kind){return new Promise((resolve,reject)=>{const req=this.db.transaction(kind).objectStore(kind).getAll();req.onsuccess=()=>resolve(req.result.sort((a,b)=>b.created_at-a.created_at));req.onerror=()=>reject(req.error);});}
  async put(kind,value){return new Promise((resolve,reject)=>{const tx=this.db.transaction(kind,'readwrite');tx.objectStore(kind).put(value);tx.oncomplete=()=>resolve(value);tx.onerror=()=>reject(Error('Could not save. Browser storage may be full. Export a backup before clearing data.'));});}
  async snapshot(){const result={};for(const k of ['experiments','runs','episodes','sessions'])result[k]=await this.all(k);return result;}
  async restore(library){return new Promise((resolve,reject)=>{const tx=this.db.transaction(['experiments','runs','episodes','sessions'],'readwrite');for(const kind of ['experiments','runs','episodes','sessions'])for(const item of library[kind])tx.objectStore(kind).put(item);tx.oncomplete=resolve;tx.onabort=tx.onerror=()=>reject(Error('Import failed. Existing records were preserved.'));});}
}
export class Bridge{
  constructor(url,token){const u=new URL(url);if(!((u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname))||u.protocol==='https:')||u.username||u.password||u.search||u.hash)throw Error('Use a localhost HTTP URL or an HTTPS bridge URL, without credentials.');this.url=u.origin;this.token=token;}
  async request(path,body){const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),100000);try{const r=await fetch(this.url+'/api/'+path,{method:body===undefined?'GET':'POST',headers:{'Authorization':'Bearer '+this.token,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal});const data=await r.json();if(!r.ok)throw Error(data.error||`Bridge error ${r.status}`);return data;}catch(e){if(e.name==='TypeError'||e.name==='AbortError')throw Error('Bridge could not be reached. Start it on the laptop, allow this site origin, and check browser local-network permission. You can also use the local dashboard.');throw e;}finally{clearTimeout(timer);}}
}
export class Camera{
  constructor(mediaDevices,onChange=()=>{}){this.mediaDevices=mediaDevices;this.onChange=onChange;this.stream=null;this.serial=0;this.devices=[];}
  async start(deviceId){
    if(!this.mediaDevices?.getUserMedia)throw Error('Camera access requires HTTPS or localhost and a browser with camera support.');
    const serial=++this.serial;
    let stream;
    try{stream=await this.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:1280},height:{ideal:720},...(deviceId?{deviceId:{exact:deviceId}}:{})}});}catch(e){const messages={NotAllowedError:'Camera permission was declined. Enable it in your browser’s site settings when you want to use the camera.',NotFoundError:'No camera was found. Connect your UVC camera or use the laptop camera.',NotReadableError:'The camera is busy. Close other camera apps and try again.',OverconstrainedError:'That camera is unavailable. Choose another camera.'};throw Error(messages[e.name]||e.message||'Camera could not start.');}
    if(serial!==this.serial){stream.getTracks().forEach(t=>t.stop());return;}
    if(this.stream)this.stream.getTracks().forEach(t=>t.stop());
    this.stream=stream;
    stream.getVideoTracks().forEach(t=>t.addEventListener('ended',()=>{if(this.stream===stream)this.stop();}));
    try{this.devices=(await this.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');}catch{this.devices=[];}
    this.onChange();
  }
  stop(){this.serial++;const old=this.stream;this.stream=null;if(old)old.getTracks().forEach(t=>t.stop());this.onChange();}
  capture(video,canvas){if(!this.stream||!video.videoWidth)throw Error('Wait for a camera frame before capturing.');canvas.width=Math.min(video.videoWidth,800);canvas.height=Math.round(video.videoHeight*canvas.width/video.videoWidth);canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);return {data_url:canvas.toDataURL('image/jpeg',0.8),captured_at:Date.now()/1000};}
}
export function validateBackup(data){
  if(data?.format!=='armature-backup-v1'||!data.library)throw Error('Choose an Armature backup JSON file.');
  const lib=data.library;
  for(const k of ['experiments','runs','episodes','sessions'])if(!Array.isArray(lib[k])||lib[k].length>10000)throw Error('Invalid backup library.');
  const ids=new Set();for(const exp of lib.experiments){validateExperiment(exp);if(!/^[a-f0-9]{32}$/.test(exp.id)||!Number.isFinite(exp.created_at))throw Error('Invalid experiment identity.');ids.add(exp.id);}
  for(const k of ['runs','episodes','sessions'])for(const item of lib[k]){if(!/^[a-f0-9]{32}$/.test(item.id)||!ids.has(item.experiment_id)||!Number.isFinite(item.created_at))throw Error('Invalid linked record.');}
  for(const run of lib.runs){
    if(!['starting','running','stopping','completed','cancelled','failed','interrupted'].includes(run.status)||!Array.isArray(run.logs)||run.logs.some(s=>typeof s!=='string')||!Array.isArray(run.metrics)||run.metrics.length>10000)throw Error('Invalid run.');
    for(const metric of run.metrics)if(!metric||!Number.isFinite(metric.step)||Object.values(metric).some(v=>!Number.isFinite(v)))throw Error('Invalid metric.');
  }
  for(const ep of lib.episodes){
    if(!Array.isArray(ep.frames)||!ep.frames.length||ep.frames.length>8||!Array.isArray(ep.grades)||![null,true,false].includes(ep.human_success)||typeof ep.notes!=='string')throw Error('Invalid episode.');
    for(const f of ep.frames)if(typeof f.data_url!=='string'||f.data_url.length>1500000||!/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(f.data_url)||!Number.isFinite(f.captured_at))throw Error('Invalid captured frame.');
    for(const g of ep.grades)if(!g||![null,true,false].includes(g.success)||!(g.score===null||(Number.isFinite(g.score)&&g.score>=0&&g.score<=1))||!Number.isFinite(g.confidence)||typeof g.reason!=='string'||typeof g.model!=='string')throw Error('Invalid grade.');
  }
  for(const session of lib.sessions){
    if(!['sandbox','so101'].includes(session.mode)||!Array.isArray(session.history))throw Error('Invalid session.');
    for(const event of session.history)if(!event||!Number.isFinite(event.time)||typeof event.action!=='string'||typeof event.note!=='string')throw Error('Invalid session event.');
  }
  return lib;
}
