const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(__dirname+'/index.html','utf8');
const code = html.split('<script>')[1].split('</script>')[0].replace(/poll\(\);\s*$/, '');
const elements = new Map();
const document = {getElementById(id){if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',querySelectorAll:()=>[]});return elements.get(id)},querySelectorAll:()=>[]};
const ctx=vm.createContext({document,Date,console,setTimeout(){}});
vm.runInContext(code,ctx);
const recipe={title:'<script>alert(1)</script>',servings:2,ingredients:[{name:'Рис',amount:100,unit:'г'}],steps:['Сварить'],notes:[],used_source_ids:['a']};
ctx.render({id:'a',mode:'v1',status:'done',query:'Ужин',started:0,finished:10,result:{recipe,selected_sources:[{id:'a',title:'Книга',text:'<img onerror=bad>'}]}});
let result=elements.get('result').innerHTML;
assert(result.includes('&lt;script&gt;'));
assert(!result.includes('<img onerror'));
assert(result.includes('100 г'));
ctx.render({id:'a',mode:'v2',status:'stopped',query:'Ужин',started:0,finished:10,result:{state:{draft:recipe}}});
assert(elements.get('result').innerHTML.includes('Черновик · пока не принят'));
ctx.render({id:'a',mode:'v2',status:'done',query:'Ужин',started:0,finished:10,result:{recipe,accepted:true,llm_calls:[{number:1,role:'critic',status:'ok',value:{verdict:'pass',summary:'OK'}}]}});
assert(elements.get('result').innerHTML.includes('Принят внутренним критиком'));
assert(elements.get('diagnostics').innerHTML.includes('Критик'));
console.log('UI rendering: escaping, ingredients, draft/accepted states, agent timeline OK');

// A cleared field stays empty across polling; edits before the first response win.
const query=elements.get('query');
query.value='';
ctx.initializeQuery('Сохранённый запрос');
assert.equal(query.value,'Сохранённый запрос');
query.value='';query.oninput();
ctx.initializeQuery('Сохранённый запрос');
assert.equal(query.value,'');
vm.runInContext('queryInitialized=false',ctx);
query.value='Мой новый запрос';query.oninput();
ctx.initializeQuery('Сохранённый запрос');
assert.equal(query.value,'Мой новый запрос');
console.log('Query field: cleared and edited values survive polling');

ctx.render({id:'empty',mode:'v2',status:'stopped',query:'Мясо',started:0,finished:3,result:{status:'stopped'}});
assert(elements.get('result').innerHTML.includes('Рецепт не создан'));
assert(elements.get('result').innerHTML.includes('badge warn'));

const bookJob={id:'book',mode:'v2',status:'stopped',query:'Мясо',started:0,finished:3,
  result:{state:{selected:[{id:'s',title:'Мясо в духовке',text:'Запечь <мясо> без жарки.'}]}},
  conversation:[{id:'q',question:'Можно <без чеснока>?',answer:'Да, можно.',status:'done'}]};
ctx.render(bookJob);
assert(elements.get('result').innerHTML.includes('Запечь &lt;мясо&gt; без жарки.'));
assert(elements.get('result').innerHTML.includes('Книжный рецепт'));
assert(!elements.get('result').innerHTML.includes('Рецепт не создан'));
assert.equal(elements.get('chat').hidden,false);
assert(elements.get('chatMessages').innerHTML.includes('&lt;без чеснока&gt;'));
const question=elements.get('question');question.value='А сколько соли?';question.oninput();
ctx.render(bookJob);
assert.equal(question.value,'А сколько соли?');
ctx.render({...bookJob,id:'other',conversation:[]});
assert.equal(question.value,'');
ctx.render(bookJob);
assert.equal(question.value,'А сколько соли?');
ctx.render({...bookJob,conversation:[{question:'?',status:'running'}]});
assert.equal(elements.get('chatSend').disabled,true);
ctx.render(bookJob);
assert.equal(ctx.elapsedSeconds({status:'running',started:99999999999,elapsed_seconds:7.5}),7.5);
assert.equal(ctx.elapsedSeconds({status:'done',started:100,finished:10}),0);
assert.equal(ctx.elapsedSeconds({status:'done',started:100,finished:10,result:{total_seconds:8}}),8);
ctx.render({...bookJob,elapsed_seconds:9,presentation:{status:'formatted',llm_calls:1,documents:[
  {id:'s',title:'Мясо в духовке',sections:[{heading:'Приготовление',blocks:[{text:'Запечь <мясо> 40 минут.',number:1}]}]}
]}});
assert(elements.get('result').innerHTML.includes('9 сек.'));
assert(elements.get('result').innerHTML.includes('Запечь &lt;мясо&gt; 40 минут.'));
assert(elements.get('result').innerHTML.includes('Исходный рецепт'));
assert(!elements.get('result').innerHTML.includes('Оформить рецепт'));
ctx.render(bookJob);
assert(html.indexOf('id="result"')<html.indexOf('id="chat"'));
assert(html.indexOf('id="chat"')<html.indexOf('id="diagnostics"'));
ctx.render({...bookJob,status:'running',stage:'Оформление рецепта',presentation:{documents:[{title:'Рано показывать'}]},result:{recipe,stats:{llm_calls:7,generated_tokens:100}}});
assert(!elements.get('result').innerHTML.includes('recipe-title'));
assert(elements.get('result').innerHTML.includes('после оформления'));
assert.equal(elements.get('chat').hidden,true);
ctx.render({...bookJob,conversation:[{question:'Другой рецепт?',status:'running',stage:'Агенты подбирают рецепт…',agent_used:true,child_id:'child'}]});
assert(elements.get('chatMessages').innerHTML.includes('Агенты подбирают рецепт'));
assert(elements.get('chatMessages').innerHTML.includes('data-child-job="child"'));
assert.equal(elements.get('chatSend').disabled,true);
ctx.render(bookJob);
document.getElementById('chatAgents').checked=true;
ctx.render({...bookJob,result:{...bookJob.result,settings:{max_llm_calls:30,max_searches:4},
  state:{...bookJob.result.state,search_rounds:2},stats:{llm_calls:6,generated_tokens:300}}});
assert(elements.get('result').innerHTML.includes('Лимит агентов: 30 вызовов · поиск 2/4'));
ctx.render(bookJob);
const waitingJob={...bookJob,id:'waiting',status:'waiting_input',result:null,
  clarification:{id:'ingredients1',status:'waiting',question:'Можно добавить муку и яйцо?',ingredients:['мука','яйцо']}};
ctx.render(waitingJob);
assert.equal(elements.get('clarification').hidden,false);
assert.equal(elements.get('clarificationQuestion').textContent,'Можно добавить муку и яйцо?');
assert(!elements.get('result').innerHTML.includes('Рецепт не создан'));
document.getElementById('clarificationAnswer').value='Мука есть, яйца нет';
ctx.render(waitingJob);
assert.equal(elements.get('clarificationAnswer').value,'Мука есть, яйца нет');
ctx.render({...bookJob,conversation:[{question:'Другой рецепт',answer:'Можно добавить муку?',status:'done',child_id:'waiting',
  needs_clarification:true,child_clarification:waitingJob.clarification}]});
assert(elements.get('chatMessages').innerHTML.includes('data-child-choice="allow"'));
assert(elements.get('chatMessages').innerHTML.includes('Свой ответ на уточнение'));
ctx.render(bookJob);
document.getElementById('chatClarify').checked=true;
const calls=[];
ctx.fetch=async(path,options)=>{
  calls.push({path,options});
  return {ok:true,json:async()=>options?.method==='POST'?{id:'new'}:path==='/api/jobs'?{jobs:[bookJob],busy:true}:bookJob};
};
vm.runInContext('selected="book"',ctx);
(async()=>{
  await elements.get('chatForm').onsubmit({preventDefault(){}});
  assert.equal(calls[0].path,'/api/jobs/book/questions');
  assert.equal(JSON.parse(calls[0].options.body).question,'А сколько соли?');
  assert.equal(JSON.parse(calls[0].options.body).allow_agents,true);
  assert.equal(JSON.parse(calls[0].options.body).allow_clarifications,true);
  assert.equal(question.value,'');
  assert.equal(elements.get('chatSend').disabled,true);
  let reads=0,releaseOld;
  ctx.fetch=async path=>{
    if(path==='/api/jobs')return {ok:true,json:async()=>({jobs:[bookJob],busy:false})};
    if(++reads===1)return new Promise(resolve=>{releaseOld=()=>resolve({ok:true,json:async()=>({...bookJob,status:'running',stage:'Устаревший ответ'})})});
    return {ok:true,json:async()=>bookJob};
  };
  const oldRefresh=ctx.refresh();
  for(let i=0;i<20&&!releaseOld;i++)await Promise.resolve();
  assert(releaseOld);
  await ctx.refresh();
  const completedHtml=elements.get('result').innerHTML;
  releaseOld();await oldRefresh;
  assert.equal(elements.get('result').innerHTML,completedHtml);
  assert(!elements.get('result').innerHTML.includes('Устаревший ответ'));
  vm.runInContext('selected="waiting";busy=false',ctx);ctx.render(waitingJob);
  const clarificationCalls=[];
  ctx.fetch=async(path,options)=>{clarificationCalls.push({path,options});return {ok:true,json:async()=>options?.method==='POST'?{id:'waiting'}:path==='/api/jobs'?{jobs:[{...waitingJob,status:'running'}],busy:true}:{...waitingJob,status:'running'}}};
  await ctx.sendClarification('allow');
  assert.equal(clarificationCalls[0].path,'/api/jobs/waiting/clarification');
  assert.equal(JSON.parse(clarificationCalls[0].options.body).clarification_id,'ingredients1');
  assert.equal(JSON.parse(clarificationCalls[0].options.body).choice,'allow');
  assert.equal(elements.get('clarification').hidden,true);
  const startCalls=[];
  ctx.fetch=async(path,options)=>{startCalls.push({path,options});return {ok:true,json:async()=>options?.method==='POST'?{id:'book'}:path==='/api/jobs'?{jobs:[bookJob],busy:false}:bookJob}};
  document.getElementById('clarify').checked=true;document.getElementById('reuse').checked=true;
  await elements.get('form').onsubmit({preventDefault(){}});
  assert.equal(JSON.parse(startCalls[0].options.body).allow_clarifications,true);
  startCalls.length=0;document.getElementById('clarify').checked=false;
  await elements.get('form').onsubmit({preventDefault(){}});
  assert.equal(JSON.parse(startCalls[0].options.body).allow_clarifications,false);
  console.log('Clarifications: toggles, persistent draft, explicit approval and resume OK');
})().catch(error=>{console.error(error);process.exitCode=1});
