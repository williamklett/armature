import {recipes,rubricDefault,esc,uid,makeExperiment,download,Library,Bridge,Camera,slurmScript,validateBackup} from './lib.js';
const $=s=>document.querySelector(s),app=$('#app');
const local=await new Library().open().catch(e=>{app.textContent=e.message;throw e;});
let bridge=null,status=null,library=await local.snapshot(),frames=[],selectedExperiment='',robot=null,arm=null,toastTimer,pollBusy=false;
const camera=new Camera(navigator.mediaDevices,()=>updateCamera());
const date=t=>new Date(t*1000).toLocaleDateString(undefined,{month:'short',day:'numeric'});
const time=t=>new Date(t*1000).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'});
const relative=t=>{const m=Math.floor((Date.now()/1000-t)/60);return m<1?'Just now':m<60?`${m}m ago`:m<1440?`${Math.floor(m/60)}h ago`:date(t);};
const recipe=e=>recipes[e.recipe]||{name:e.recipe,kind:'Unknown',model:'Unknown'};
const expBy=id=>library.experiments.find(e=>e.id===id);
const badge=s=>`<span class="status ${esc(s)}">${esc(s)}</span>`;
const num=(v,d=2)=>typeof v==='number'?v.toFixed(d):'—';
function toast(text){$('#toast').textContent=text;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),6500);}
function heading(title,actions=''){return `<div class="page-heading"><h1>${title}</h1>${actions}</div>`;}
function empty(title,button=''){return `<div class="empty"><strong>${title}</strong>${button?`<div>${button}</div>`:''}</div>`;}
function card(id,icon,tag,style=''){const r=recipes[id];return `<article class="recipe-card"><div class="model-icon ${style}">${icon}</div><h3>${esc(r.name)}</h3><div class="card-tags"><span class="tag">${tag}</span></div><button class="button secondary small" data-action="new" data-recipe="${id}">Select ↗</button></article>`;}
function overview(){
 const active=library.runs.filter(r=>['running','starting','stopping'].includes(r.status)).length;
 const reviewed=library.episodes.filter(e=>e.human_success!==null).length;
 return heading('Overview')+
 `<div class="stats">${[['Experiments',library.experiments.length],['Active runs',active],['Reviewed episodes',reviewed],['Camera',camera.stream?'On':'Off']].map(([label,value])=>`<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div></div>`).join('')}</div>`+
 `<div class="section-head"><h2>Recent experiments</h2><a href="#experiments" class="text-link">View all →</a></div><div class="panel">${experimentTable(library.experiments.slice(0,4))}</div>`+
 `<div class="section-head"><h2>Recipes</h2></div><div class="cards">${card('smolvla_sft','s','450M · SFT')}${card('pi05_sft','π','SFT','blue')}${card('agent_session','⌖','Context only','lavender')}</div>`;
}
function experimentTable(exps){
 if(!exps.length)return empty('No experiments',`<button class="button secondary small" data-action="new">New experiment</button>`);
 return `<div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Method</th><th>Latest run</th><th>Episodes</th><th>Created</th><th></th></tr></thead><tbody>${exps.map(e=>{const run=library.runs.find(r=>r.experiment_id===e.id);return `<tr class="click-row" data-action="open" data-id="${e.id}"><td><a class="run-name" href="#experiment/${e.id}">${esc(e.name)}</a><span class="subtext">${esc(e.owner||recipe(e).model)}</span></td><td>${esc(recipe(e).kind)}</td><td>${badge(run?.status||'draft')}</td><td class="mono">${library.episodes.filter(ep=>ep.experiment_id===e.id).length}</td><td>${relative(e.created_at)}</td><td>↗</td></tr>`;}).join('')}</tbody></table></div>`;
}
function experiments(){return heading('Experiments',`<input class="search" id="experiment-search" placeholder="Search" aria-label="Search experiments">`)+`<div class="panel" id="experiment-table">${experimentTable(library.experiments)}</div>`;}
function chart(run){
 const values=(run?.metrics||[]).filter(m=>typeof m.reward==='number'||typeof m.loss==='number');
 if(!values.length)return empty('No metrics');
 const key=typeof values[0].reward==='number'?'reward':'loss',pts=values.filter(m=>typeof m[key]==='number');
 const lo=Math.min(...pts.map(p=>p[key])),hi=Math.max(...pts.map(p=>p[key])),maxStep=Math.max(1,...pts.map(p=>p.step));
 const line=pts.map(p=>`${35+p.step/maxStep*590},${150-(p[key]-lo)/(hi-lo||1)*120}`).join(' ');
 return `<div class="chart"><div class="chart-label">${key==='reward'?'Episode reward':'Training loss'}</div><svg viewBox="0 0 650 180" role="img" aria-label="${key} over training steps"><path d="M35 30H625M35 90H625M35 150H625" stroke="#e5eadc" stroke-dasharray="4 4"/><polyline points="${line}" fill="none" stroke="#7b9d4d" stroke-width="2.5"/><g font-size="9" fill="#8b987d" font-family="monospace"><text x="0" y="33">${num(hi)}</text><text x="0" y="153">${num(lo)}</text><text x="35" y="172">0</text><text x="570" y="172">${maxStep} steps</text></g></svg></div>`;
}
function detail(id){
 const e=expBy(id);if(!e)return empty('Experiment not found');
 const runs=library.runs.filter(r=>r.experiment_id===id),run=runs[0],eps=library.episodes.filter(ep=>ep.experiment_id===id),r=recipe(e),available=status?.runners?.[e.recipe];
 const config=[['Recipe',r.name],['Dataset',e.dataset||'Not set'],['Checkpoint',e.checkpoint||'Not set'],['Steps',e.steps.toLocaleString()],['Batch size',e.batch_size],['Learning rate',e.learning_rate],['Seed',e.seed],['Owner',e.owner||'Not set']];
 return `<a class="back" href="#experiments">← All experiments</a>`+heading(esc(e.name),`<div class="row wrap"><button class="button secondary small" data-action="clone" data-id="${id}">Clone</button><button class="button secondary small" data-action="export-exp" data-id="${id}">Export</button><button class="button small" data-action="launch" data-id="${id}">${e.recipe==='agent_session'?'Open session ↗':'▷ Start run'}</button></div>`)+
 `<div class="row wrap detail-status">${badge(e.recipe==='agent_session'?'Context only':available?.ready?'Runner ready':bridge?'Runner required':'Bridge required')}<span class="tag">${esc(r.kind)}</span><span class="tag">${esc(r.environment)}</span></div>`+
 `<div class="grid-two"><div class="stack"><section class="panel"><div class="panel-title"><h2>Learning curve</h2>${run?badge(run.status):'<span class="hint">No runs</span>'}</div>${chart(run)}</section><section class="panel"><div class="panel-title"><h2>Run history</h2><span class="hint">${runs.length} runs</span></div>${runs.length?`<div class="table-wrap"><table><thead><tr><th>Run</th><th>Status</th><th>Started</th><th></th></tr></thead><tbody>${runs.map(v=>`<tr><td class="mono">${v.id.slice(0,8)}</td><td>${badge(v.status)}</td><td>${relative(v.created_at)}</td><td><button class="button ghost small" data-action="export-run" data-id="${v.id}">JSON</button>${['running','starting','stopping'].includes(v.status)?`<button class="button danger small" data-action="stop-run" data-id="${v.id}">Stop</button>`:`<button class="button ghost small" data-action="show-run" data-id="${v.id}">Logs</button>`}</td></tr>`).join('')}</tbody></table></div>`:empty('No runs')}</section>${run?`<section class="panel"><div class="panel-title"><h2>Output</h2><span class="mono">${run.remote?`Slurm #${esc(run.remote.job_id)} · ${esc(run.remote.state)}`:run.id.slice(0,8)}</span></div><pre class="log">${esc(run.logs.join('\n')||'Waiting for output…')}</pre></section>`:''}</div>
 <div class="stack"><section class="panel"><div class="panel-title"><h2>Configuration</h2></div><div class="panel-body"><p class="subtitle" style="margin-top:0">${esc(e.task)}</p><hr class="divider"><dl class="config-list">${config.map(([k,v])=>`<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${e.notes?`<hr class="divider"><p class="subtitle">${esc(e.notes)}</p>`:''}${e.recipe.endsWith('_sft')?`<hr class="divider"><button class="button secondary small" data-action="slurm" data-id="${id}">↓ Export Slurm job</button>`:''}</div></section><section class="panel"><div class="panel-title"><h2>Grading rubric</h2></div><div class="panel-body"><p class="subtitle" style="margin:0">${esc(e.rubric)}</p></div></section></div></div><div class="section-head"><h2>Episodes</h2><button class="button secondary small" data-action="capture-for" data-id="${id}">Open camera →</button></div><div class="panel">${episodeTable(eps)}</div>`;
}
function episodeTable(eps){if(!eps.length)return empty('No episodes');return `<div class="table-wrap"><table><thead><tr><th>Frames</th><th>Captured</th><th>Human label</th><th>Model score</th><th></th></tr></thead><tbody>${eps.map(ep=>{const g=ep.grades.at(-1);return `<tr class="click-row" data-action="episode" data-id="${ep.id}"><td><div class="episode-thumbs">${ep.frames.slice(0,3).map(f=>`<img src="${esc(f.data_url)}" alt="Captured frame">`).join('')}</div></td><td>${date(ep.created_at)}<span class="subtext">${time(ep.created_at)}</span></td><td>${ep.human_success===null?'Unreviewed':ep.human_success?'Success':'Failure'}</td><td class="mono">${g?(g.score===null?'Unclear':num(g.score)):'Not graded'}</td><td>Review ↗</td></tr>`;}).join('')}</tbody></table></div>`;}
function experimentOptions(){return `<option value="">Select experiment</option>`+library.experiments.map(e=>`<option value="${e.id}" ${e.id===selectedExperiment?'selected':''}>${esc(e.name)}</option>`).join('');}
function control(){return heading('Control room')+`<div class="control-grid"><section class="panel"><div class="panel-title"><h2>Camera</h2><span id="camera-status" class="pill">Off</span></div><div class="camera-stage"><video id="camera-video" autoplay muted playsinline aria-label="Live camera preview" hidden></video><div class="camera-placeholder" id="camera-placeholder"><svg viewBox="0 0 48 48" fill="none" stroke-width="1.4"><rect x="6" y="14" width="29" height="24" rx="5"/><path d="m35 22 8-5v18l-8-5M15 14l3-5h9l3 5"/><circle cx="21" cy="26" r="6"/></svg></div></div><div class="camera-actions"><button class="button small" data-action="camera-on" id="camera-on">Enable camera</button><select id="camera-device" aria-label="Camera device" hidden></select><button class="button secondary small" data-action="capture" id="capture-btn" disabled>◎ Capture</button><button class="button ghost small" data-action="camera-off" id="camera-off" hidden>Turn off</button></div><div class="frames" id="frames"></div><div class="panel-body" style="padding-top:0"><div class="form-field"><label for="episode-experiment">Experiment</label><select id="episode-experiment">${experimentOptions()}</select></div><div class="spacer"></div><div class="form-field"><label for="episode-notes">Notes</label><textarea id="episode-notes" rows="2"></textarea></div><div class="split" style="margin-top:15px"><span class="hint" id="frame-count">${frames.length}/8 frames</span><button class="button small" data-action="save-episode">Save episode</button></div></div></section><div class="stack"><section class="panel"><div class="panel-title"><h2>2D sandbox</h2><span class="tag">NO HARDWARE</span></div><div class="sandbox" id="sandbox-scene"></div><div class="panel-body"><div class="split"><strong id="session-label">Stopped</strong><div class="row"><button class="button secondary small" data-action="session-start">Start</button><button class="button danger small" data-action="session-stop">Stop</button></div></div><hr class="divider"><div class="split"><div class="jog"><span></span><button data-action="jog" data-dy="-0.06" aria-label="Move up">↑</button><span></span><button data-action="jog" data-dx="-0.06" aria-label="Move left">←</button><button data-action="jog" data-dy="0.06" aria-label="Move down">↓</button><button data-action="jog" data-dx="0.06" aria-label="Move right">→</button></div><button class="button secondary small" data-action="gripper">Toggle gripper</button></div><div class="spacer"></div><div class="form-field"><label for="session-note">Session notes</label><textarea id="session-note" rows="2"></textarea><button class="button secondary small" data-action="session-note">Save note</button></div><div class="session-events" id="session-events"></div></div></section><div class="connection-card" id="arm-panel"></div></div></div>`;}
function models(){return heading('Models',`<span class="tag">Weights not installed</span>`)+`<div class="cards">${Object.entries(recipes).map(([id,r],i)=>card(id,r.model==='π0.5'?'π':r.model==='SmolVLA'?'s':r.model==='Agent'?'⌖':'q',esc(r.kind),i%2?'blue':'')).join('')}</div><div class="section-head"><h2>Documentation</h2></div><div class="panel panel-body row wrap"><a class="text-link" target="_blank" rel="noreferrer" href="https://huggingface.co/docs/lerobot/smolvla">SmolVLA ↗</a><a class="text-link" target="_blank" rel="noreferrer" href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html">π0.5 RL ↗</a><a class="text-link" target="_blank" rel="noreferrer" href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/so101.html">SO101 ↗</a></div>`;}
function setup(){return heading('Connections')+`<div class="grid-two"><div class="stack"><div class="connection-card"><div class="split"><h2>Laptop bridge</h2><span class="pill ${bridge?'connected':''}">${bridge?'Connected':'Disconnected'}</span></div><pre>python3 -m bridge.server</pre><form id="bridge-form" class="stack"><div class="form-field"><label for="bridge-url">Bridge URL</label><input id="bridge-url" type="url" required value="${esc(bridge?.url||localStorage.getItem('armature-bridge-url')||'http://127.0.0.1:8765')}"></div><div class="form-field"><label for="bridge-token">Pairing token</label><input id="bridge-token" type="password" required autocomplete="off"></div><div class="row wrap"><button class="button" type="submit">Connect</button>${bridge?`<button type="button" class="button secondary" data-action="disconnect">Disconnect</button><button type="button" class="button ghost" data-action="copy-token">Copy token</button>`:''}</div></form><details class="setup-details"><summary>Setup</summary><dl class="config-list"><dt>Pairing token</dt><dd><a href="http://127.0.0.1:8765" target="_blank" rel="noreferrer">Local dashboard ↗</a></dd><dt>Allowed origin</dt><dd class="mono">bridge.local.json → allowed_origins</dd></dl></details></div><div class="connection-card"><div class="split"><h2>Backup</h2><span class="tag">Includes frames</span></div><div class="row wrap spaced"><button class="button secondary" data-action="backup">↓ Export library</button><label class="button secondary" for="import-backup">↑ Import to browser</label><input type="file" id="import-backup" accept="application/json,.json" hidden></div></div></div><div class="stack"><div class="connection-card"><div class="split"><h2>Grader</h2>${badge(status?.grader_ready?'Configured':'Not configured')}</div>${status?.grader_model?`<div class="mono spaced">${esc(status.grader_model)}</div>`:''}<details class="setup-details"><summary>Environment variables</summary><pre>OPENAI_API_KEY<br>ARMATURE_GRADER_MODEL</pre><span class="tag">Bridge only</span></details></div><div class="connection-card"><h2>Training</h2><div class="mono spaced">bridge.local.json → runners</div><dl class="config-list">${['smolvla_sft','pi05_sft','smolvla_rl','pi05_rl'].map(id=>`<dt>${esc(recipes[id].name)}</dt><dd>${status?.runners?.[id]?.ready?'Configured':'Not configured'}</dd>`).join('')}</dl></div></div></div>`;}
function render(){
 const [page,id]=(location.hash.slice(1)||'overview').split('/');
 document.querySelectorAll('[data-nav]').forEach(n=>n.classList.toggle('active',n.dataset.nav===(page==='experiment'?'experiments':page)));
 $('#nav-count').textContent=library.experiments.length;$('#storage-badge').className='pill'+(bridge?' connected':'');$('#storage-badge').innerHTML=`<i></i>${bridge?'Laptop connected':'Browser library'}`;

 app.innerHTML=page==='experiment'?detail(id):({overview,experiments,control,models,setup}[page]||overview)();if(page==='control'){updateCamera();drawSandbox();drawArm();}
}
async function refresh(){const current=bridge;const next=current?await current.request('library'):await local.snapshot();if(current!==bridge)return false;const changed=JSON.stringify(library)!==JSON.stringify(next);library=next;return changed;}
function newDialog(recipeId='smolvla_sft',copy){
 const d={name:'',task:'Place the block in the tray.',owner:'',recipe:recipeId,dataset:'',checkpoint:recipes[recipeId].checkpoint,seed:42,steps:recipeId==='toy_q'?200:20000,batch_size:8,learning_rate:({toy_q:.2,pi05_sft:.000025,pi05_rl:.000005,smolvla_rl:.000001})[recipeId]??.0001,rubric:rubricDefault,notes:'',...copy};if(copy)d.name+=' · copy';
 const input=(name,label,type='text')=>`<div class="form-field"><label for="f-${name}">${label}</label><input id="f-${name}" name="${name}" type="${type}" value="${esc(d[name])}" ${type==='number'?'step="any"':''}></div>`;
 $('#experiment-dialog').innerHTML=`<form id="experiment-form"><div class="dialog-head"><div><h2 id="dialog-title">${copy?'Clone experiment':'New experiment'}</h2></div><button class="icon" type="button" data-action="close" aria-label="Close">×</button></div><div class="dialog-body"><div class="form-grid">${input('name','Experiment name')}${input('owner','Owner')}<div class="form-field full"><label for="f-recipe">Recipe</label><select name="recipe" id="f-recipe">${Object.entries(recipes).map(([id,r])=>`<option value="${id}" ${id===d.recipe?'selected':''}>${esc(r.name)} — ${esc(r.kind)}</option>`).join('')}</select></div><div class="form-field full"><label for="f-task">Task</label><textarea name="task" id="f-task" required>${esc(d.task)}</textarea></div>${input('dataset','LeRobot dataset')}${input('checkpoint','Starting checkpoint')}${input('steps','Steps / toy episodes','number')}${input('batch_size','Batch size','number')}${input('seed','Seed','number')}${input('learning_rate','Learning rate','number')}<div class="form-field full"><label for="f-rubric">Grading rubric</label><textarea id="f-rubric" name="rubric" required>${esc(d.rubric)}</textarea></div><div class="form-field full"><label for="f-notes">Notes</label><textarea id="f-notes" name="notes" rows="2">${esc(d.notes)}</textarea></div></div><div class="dialog-error" id="form-error" role="alert"></div></div><div class="dialog-footer"><button class="button secondary" type="button" data-action="close">Cancel</button><button class="button" type="submit">Save experiment →</button></div></form>`;
 $('#experiment-dialog').showModal();$('#f-name').focus();
}

function updateCamera(){
 const video=$('#camera-video');if(!video)return;
 video.srcObject=camera.stream;video.hidden=!camera.stream;$('#camera-placeholder').hidden=!!camera.stream;
 $('#camera-status').textContent=camera.stream?'● Live · local':'Off';
 $('#camera-on').hidden=!!camera.stream;$('#camera-off').hidden=!camera.stream;$('#capture-btn').disabled=!camera.stream;
 const select=$('#camera-device');select.hidden=!camera.stream;
 const active=camera.stream?.getVideoTracks()[0]?.getSettings?.().deviceId;
 select.innerHTML=camera.devices.map((d,i)=>`<option value="${esc(d.deviceId)}" ${d.deviceId===active?'selected':''}>${esc(d.label||`Camera ${i+1}`)}</option>`).join('');
 if(camera.stream)video.play().catch(()=>toast('Click preview to play.'));
 drawFrames();
}
function drawFrames(){if(!$('#frames'))return;$('#frames').innerHTML=frames.map((f,i)=>`<div class="frame"><img src="${esc(f.data_url)}" alt="Captured frame ${i+1}"><button data-action="remove-frame" data-index="${i}" aria-label="Remove frame ${i+1}">×</button></div>`).join('');$('#frame-count').textContent=`${frames.length}/8 frames`;}
function episodeDialog(id){
 const ep=library.episodes.find(e=>e.id===id);if(!ep)return;
 const grade=ep.grades.at(-1),exp=expBy(ep.experiment_id);
 $('#episode-dialog').innerHTML=`<div class="dialog-head"><div><div class="eyebrow">Episode review</div><h2>${esc(exp?.name||'Episode')}</h2></div><button class="icon" data-action="close" aria-label="Close">×</button></div><div class="dialog-body"><div class="review-frames">${ep.frames.map((f,i)=>`<div><img src="${esc(f.data_url)}" alt="Episode frame ${i+1}"><small>Frame ${i+1} · ${time(f.captured_at)}</small></div>`).join('')}</div><p class="subtitle">${esc(ep.notes||'')}</p><hr class="divider"><h3>Human label</h3><p class="subtitle">${ep.human_success===null?'Unreviewed':ep.human_success?'Success':'Failure'}</p><div class="row wrap" style="margin-top:12px"><button class="button secondary small" data-action="review" data-id="${id}" data-value="true">✓ Success</button><button class="button secondary small" data-action="review" data-id="${id}" data-value="false">× Failure</button><button class="button ghost small" data-action="review" data-id="${id}" data-value="null">Clear label</button></div><hr class="divider"><h3>Model grade</h3>${grade?`<div class="info"><strong>${grade.success===null?'Unclear':grade.success?'Task succeeded':'Task not completed'} · Score ${num(grade.score)}</strong><p>${esc(grade.reason)}</p><span class="hint">${esc(grade.model)} · confidence ${num(grade.confidence)} · ${ep.grades.length} grades</span></div>`:`<p class="subtitle">Not graded</p>`}<div class="info warning">${ep.frames.length} frames + task + rubric → OpenAI · API charges apply</div><button class="button" data-action="grade" data-id="${id}" ${status?.grader_ready?'':'disabled'}>Grade with OpenAI</button>${status?.grader_ready?'':`<p class="hint" style="margin-top:10px">Grader not connected</p>`}</div>`;
 if(!$('#episode-dialog').open)$('#episode-dialog').showModal();
}
function initialRobot(){return {x:.35,y:.6,gripper:1,stopped:true,revision:0,history:[],session_id:uid(),target:{x:.72,y:.3}};}
function drawSandbox(){
 if(!$('#sandbox-scene'))return;robot??=initialRobot();const x=robot.x*400,y=robot.y*230;
 $('#sandbox-scene').innerHTML=`<svg viewBox="0 0 400 230" role="img" aria-label="Sandbox gripper at x ${num(robot.x)}, y ${num(robot.y)}"><defs><pattern id="grid" width="25" height="25" patternUnits="userSpaceOnUse"><path d="M25 0H0V25" fill="none" stroke="#dfe7d5" stroke-width=".6"/></pattern></defs><rect width="400" height="230" fill="url(#grid)"/><circle cx="288" cy="69" r="20" fill="#d9e8b6" stroke="#a5bf7e" stroke-dasharray="4 4"/><text x="288" y="104" text-anchor="middle" font-size="9" fill="#8b9e75">TARGET</text><path d="M60 200 125 115 ${x} ${y}" stroke="#8d9d7d" stroke-width="15" stroke-linecap="round" fill="none"/><circle cx="125" cy="115" r="12" fill="#3e5946"/><circle cx="60" cy="200" r="16" fill="#3e5946"/><circle cx="${x}" cy="${y}" r="9" fill="#3e5946"/><path d="m${x-7} ${y+7} ${robot.gripper?-6:0} 14m14-14 ${robot.gripper?6:0} 14" stroke="#3e5946" stroke-width="4" fill="none"/><text x="15" y="22" font-size="9" fill="#7c8d6d" font-family="monospace">SANDBOX / x ${num(robot.x)} · y ${num(robot.y)}</text></svg>`;
 $('#session-label').textContent=robot.stopped?'Stopped':'Active';
 $('#session-events').innerHTML=robot.history.slice(-6).reverse().map(h=>`<div class="session-event"><time>${time(h.time)}</time>${esc(h.note||h.action)}</div>`).join('');
}
function drawArm(){
 const panel=$('#arm-panel');if(!panel)return;
 const state=arm||status?.robot||{configured:false,connected:false};
 panel.innerHTML=`<div class="split"><h3>SO-ARM101</h3>${badge(state.connected?(state.stopped?'Stopped':'Active'):state.configured?'Disconnected':'Not configured')}</div>${state.fault?`<div class="info warning">${esc(state.fault)}</div>`:''}<div class="row wrap spaced">${state.connected?`<button class="button secondary small" data-action="arm-start">Start session</button><button class="button danger small" data-action="arm-stop">Stop</button><button class="button ghost small" data-action="arm-disconnect">Disconnect</button>`:`<button class="button secondary small" data-action="arm-connect" ${state.configured?'':'disabled'}>Connect motors</button>`}</div>${state.connected?`<div class="form-field spaced"><label for="arm-joint">Joint</label><select id="arm-joint">${Object.keys(state.positions||{}).map(j=>`<option value="${esc(j)}">${esc(j.replaceAll('_',' '))} · ${num(state.positions[j])}${j==='gripper'?'%':'°'}</option>`).join('')}</select></div><div class="row spaced"><button class="button secondary small" data-action="arm-jog" data-sign="-1">− 0.5</button><button class="button secondary small" data-action="arm-jog" data-sign="1">+ 0.5</button></div>`:''}<a class="text-link hardware-link" target="_blank" rel="noreferrer" href="https://huggingface.co/docs/lerobot/so101">Hardware setup ↗</a>`;
}
async function robotAction(data){
 if(bridge){const obs=await bridge.request('robot/observe');robot=await bridge.request('robot/action',{...data,revision:obs.revision});}
 else{
  robot??=initialRobot();
  if(data.action==='start'){if(!expBy(data.experiment_id))throw Error('Select an experiment for the session.');robot={...initialRobot(),stopped:false,experiment_id:data.experiment_id};}
  else if(data.action==='stop')robot.stopped=true;
  else if(data.action==='move'){
   if(robot.stopped)throw Error('Start a sandbox session first.');
   const x=robot.x+(data.dx||0),y=robot.y+(data.dy||0);if(x<.08||x>.92||y<.08||y>.92)throw Error('Action exceeds the sandbox workspace.');
   Object.assign(robot,{x,y,gripper:data.gripper??robot.gripper});
  }else if(!robot.experiment_id)throw Error('Start a session first.');
  robot.revision++;robot.history.push({time:Date.now()/1000,action:data.action,note:data.note||'',state:{x:robot.x,y:robot.y,gripper:robot.gripper}});
  if(robot.experiment_id)await local.put('sessions',{id:robot.session_id,created_at:robot.history[0].time,experiment_id:robot.experiment_id,mode:'sandbox',history:robot.history,state:{x:robot.x,y:robot.y,gripper:robot.gripper,stopped:robot.stopped}});
 }
 drawSandbox();
}
async function connect(url,token){
 const next=new Bridge(url,token),nextStatus=await next.request('status');
 bridge=next;status=nextStatus;robot=null;arm=null;
 sessionStorage.setItem('armature-connection',JSON.stringify({url:next.url,token}));localStorage.setItem('armature-bridge-url',next.url);
 await refresh();render();toast('Bridge connected.');
}
document.addEventListener('submit',async event=>{
 const form=event.target;if(!['experiment-form','bridge-form'].includes(form.id))return;event.preventDefault();const button=event.submitter;button.disabled=true;
 try{
  if(form.id==='bridge-form')await connect($('#bridge-url').value,$('#bridge-token').value.trim());
  else{
   const d=Object.fromEntries(new FormData(form));for(const key of ['steps','batch_size','seed','learning_rate'])d[key]=Number(d[key]);
   const checked=makeExperiment(d);const exp=bridge?await bridge.request('experiments',checked):await local.put('experiments',checked);
   await refresh();$('#experiment-dialog').close();location.hash='experiment/'+exp.id;toast('Experiment saved.');
  }
 }catch(error){if(form.id==='experiment-form')$('#form-error').textContent=error.message;else toast(error.message);}finally{button.disabled=false;}
});
document.addEventListener('input',event=>{if(event.target.id==='experiment-search'){const term=event.target.value.toLowerCase();$('#experiment-table').innerHTML=experimentTable(library.experiments.filter(e=>(e.name+' '+e.task+' '+e.owner+' '+recipe(e).model).toLowerCase().includes(term)));}});
document.addEventListener('change',async event=>{
 try{
  if(event.target.id==='f-recipe'){const id=event.target.value;$('#f-checkpoint').value=recipes[id].checkpoint;$('#f-learning_rate').value=({toy_q:.2,pi05_sft:.000025,pi05_rl:.000005,smolvla_rl:.000001})[id]??.0001;if(id==='toy_q')$('#f-steps').value=200;}
  if(event.target.id==='episode-experiment')selectedExperiment=event.target.value;
  if(event.target.id==='camera-device')await camera.start(event.target.value);
  if(event.target.id==='import-backup'){
   const file=event.target.files[0];if(!file)return;if(file.size>50000000)throw Error('Backup exceeds the 50 MB import limit.');
   const incoming=validateBackup(JSON.parse(await file.text()));
   await local.restore(incoming);
   if(!bridge){await refresh();render();}toast('Backup imported.'+plusBridge());
  }
 }catch(error){toast(error.message);}
});
function plusBridge(){return bridge?' Disconnect to view it.':'';}
document.addEventListener('click',async event=>{
 const btn=event.target.closest('[data-action]');if(!btn||btn.disabled)return;
 const action=btn.dataset.action,id=btn.dataset.id;
 if(action==='new'){newDialog(btn.dataset.recipe||'smolvla_sft');return;}
 if(action==='close'){btn.closest('dialog').close();return;}
 if(action==='open'){location.hash='experiment/'+id;return;}
 if(action==='clone'){newDialog(expBy(id).recipe,expBy(id));return;}
 if(action==='episode'){episodeDialog(id);return;}
 btn.disabled=true;
 try{
  if(action==='export-exp')download('armature-experiment-'+id.slice(0,8)+'.json',expBy(id));
  if(action==='slurm')download('armature-'+id.slice(0,8)+'.sbatch',slurmScript(expBy(id)),'text/plain');
  if(action==='export-run')download('armature-run-'+id.slice(0,8)+'.json',library.runs.find(r=>r.id===id));
  if(action==='show-run'){const run=library.runs.find(r=>r.id===id);$('#episode-dialog').innerHTML=`<div class="dialog-head"><h2>Run ${id.slice(0,8)}</h2><button class="icon" data-action="close" aria-label="Close">×</button></div><pre class="log">${esc(run.logs.join('\n')||'No output recorded.')}</pre>`;$('#episode-dialog').showModal();}
  if(action==='launch'){
   const exp=expBy(id);if(exp.recipe==='agent_session'){selectedExperiment=id;location.hash='control';return;}
   if(!bridge)throw Error('Connect the laptop bridge in Connections to launch training.');
   await bridge.request(`experiments/${id}/runs`,{});await refresh();render();toast('Run started.');
  }
  if(action==='stop-run'){await bridge.request(`runs/${id}/stop`,{});await refresh();render();toast('Stopping run.');}
  if(action==='capture-for'){selectedExperiment=id;location.hash='control';}
  if(action==='camera-on')await camera.start();
  if(action==='camera-off')camera.stop();
  if(action==='capture'){if(frames.length>=8)throw Error('Save this episode before capturing more than 8 frames.');frames.push(camera.capture($('#camera-video'),document.createElement('canvas')));drawFrames();}
  if(action==='remove-frame'){frames.splice(Number(btn.dataset.index),1);drawFrames();}
  if(action==='save-episode'){
   selectedExperiment=$('#episode-experiment').value;if(!expBy(selectedExperiment))throw Error('Select an experiment before saving.');if(!frames.length)throw Error('Capture at least one frame first.');
   const data={experiment_id:selectedExperiment,frames:structuredClone(frames),notes:$('#episode-notes').value};
   const ep=bridge?await bridge.request('episodes',data):await local.put('episodes',{...data,id:uid(),created_at:Date.now()/1000,human_success:null,grades:[],source:'camera'});
   frames=[];$('#episode-notes').value='';drawFrames();await refresh();toast('Episode saved.');episodeDialog(ep.id);
  }
  if(action==='review'){const value=JSON.parse(btn.dataset.value);if(bridge)await bridge.request(`episodes/${id}/review`,{human_success:value});else{const ep=library.episodes.find(e=>e.id===id);await local.put('episodes',{...ep,human_success:value});}await refresh();episodeDialog(id);}
  if(action==='grade'){btn.textContent='Grading…';await bridge.request(`episodes/${id}/grade`,{});await refresh();episodeDialog(id);toast('Grade saved.');}
  if(action.startsWith('arm-')){
   if(!bridge)throw Error('Bridge required.');
   const operation=action.slice(4),data={action:operation};
   if(operation==='start')data.experiment_id=$('#episode-experiment').value;
   if(operation==='jog'){data.joint=$('#arm-joint').value;data.delta=Number(btn.dataset.sign)*0.5;const observation=await bridge.request('arm/observe');data.revision=observation.revision;}
   arm=await bridge.request('arm/action',data);drawArm();
  }
  if(action==='session-start'){selectedExperiment=$('#episode-experiment').value;await robotAction({action:'start',experiment_id:selectedExperiment});}
  if(action==='session-stop')await robotAction({action:'stop'});
  if(action==='jog')await robotAction({action:'move',dx:Number(btn.dataset.dx||0),dy:Number(btn.dataset.dy||0)});
  if(action==='gripper')await robotAction({action:'move',gripper:robot?.gripper?0:1});
  if(action==='session-note'){const note=$('#session-note').value.trim();if(!note||note.length>2000)throw Error('Write a note of 1–2000 characters.');await robotAction({action:'note',note});$('#session-note').value='';}
  if(action==='backup'){await refresh();download('armature-backup-'+new Date().toISOString().slice(0,10)+'.json',{format:'armature-backup-v1',exported_at:new Date().toISOString(),library});}
  if(action==='disconnect'){bridge=null;status=null;robot=null;arm=null;sessionStorage.removeItem('armature-connection');await refresh();render();toast('Browser library.');}
  if(action==='copy-token'){await navigator.clipboard.writeText(bridge.token);toast('Token copied.');}
 }catch(error){toast(error.message);if(action==='grade')btn.textContent='Grade with OpenAI';}finally{btn.disabled=false;}
});
window.addEventListener('hashchange',()=>{if(!location.hash.startsWith('#control'))camera.stop();render();if(location.hash.startsWith('#control')&&bridge){bridge.request('arm/observe').then(v=>{arm=v;drawArm();}).catch(e=>toast(e.message));}if(location.hash.startsWith('#control')&&bridge)bridge.request('robot/observe').then(v=>{robot=v;drawSandbox();}).catch(e=>toast(e.message));});
window.addEventListener('pagehide',()=>camera.stop());
document.addEventListener('visibilitychange',()=>{if(document.hidden)camera.stop();});
window.addEventListener('beforeunload',event=>{if(frames.length){event.preventDefault();event.returnValue='';}});
render();
try{
 if(['127.0.0.1','localhost'].includes(location.hostname)){
  const response=await fetch('/api/pair');if(response.ok){const pair=await response.json();await connect(location.origin,pair.token);}
 }else{const saved=sessionStorage.getItem('armature-connection');if(saved){const {url,token}=JSON.parse(saved);await connect(url,token);}}
}catch{sessionStorage.removeItem('armature-connection');}
setInterval(async()=>{
 if(!bridge||pollBusy)return;pollBusy=true;
 try{const changed=await refresh();const page=location.hash.slice(1).split('/')[0];if(changed&&!['control','setup'].includes(page)&&!document.querySelector('dialog[open]')&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){const scroll=window.scrollY;render();window.scrollTo(0,scroll);}}
 catch{ /* Keep the existing library visible; actions report connection errors. */ }
 finally{pollBusy=false;}
},3000);
