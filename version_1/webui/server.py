#!/usr/bin/env python3
"""Local UI for the original RAG and multi-agent recipe pipelines."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from process_control import launch_options, stop_tree
from recipe_chat import recipe_context, save as save_chat
from format_recipe import documents as format_documents
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'ainura_local'
V2 = ROOT.parent / 'version_2' / 'ainura_local'
RUNS = ROOT / 'browser_runs'
INDEX = V1 / 'recipe_index'
LOCK = threading.Lock()
JOBS = {}
ACTIVE = None


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def snapshot(job):
    data = {k: v for k, v in job.items() if not k.startswith('_')}
    data['result'] = read_json(RUNS / job['id'] / 'answer.json')
    data['presentation'] = read_json(RUNS / job['id'] / 'presentation.json')
    data['formatting'] = ACTIVE == 'format:'+job['id']
    if job['status'] == 'running' and '_started_monotonic' in job:
        elapsed = job.get('_elapsed_before',0) + time.monotonic() - job['_started_monotonic']
    else:
        elapsed = job.get('elapsed_seconds', (data['result'] or {}).get('total_seconds'))
        if elapsed is None:
            elapsed = job.get('finished', job['started']) - job['started']
    data['elapsed_seconds'] = max(0, elapsed) if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) else 0
    data['conversation'] = [{k:v for k,v in turn.items() if k in {
        'id','question','answer','status','error','started','finished','stage','allow_agents','agent_used','child_id','allow_clarifications','needs_clarification'}}
        for turn in (read_json(RUNS / job['id'] / 'chat.json') or [])]
    for turn in data['conversation']:
        if turn.get('child_id') in JOBS:
            child=JOBS[turn['child_id']]
            turn['needs_clarification'] = child['status'] == 'waiting_input'
            if turn['needs_clarification']:
                turn['child_clarification'] = child.get('clarification')
    data['log'] = (RUNS / job['id'] / 'console.log').read_text(errors='replace', encoding="utf-8")[-18000:] if (RUNS / job['id'] / 'console.log').exists() else ''
    return data


def persist(job):
    path = RUNS / job['id'] / 'job.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({k:v for k,v in job.items() if not k.startswith('_')}, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def execute(job, command, stage):
    with LOCK:
        if job.get('_cancel'):
            raise InterruptedError('Запуск остановлен')
        job['stage'] = stage
        persist(job)
        log = (RUNS / job['id'] / 'console.log').open('a', encoding='utf-8')
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                **launch_options())
        job['_process'] = proc
    try:
        code = proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        stop_tree(proc)
        proc.wait()
        raise RuntimeError('Превышен часовой лимит этапа')
    finally:
        log.close()
        with LOCK:
            job.pop('_process', None)
    if job.get('_cancel'):
        raise InterruptedError('Запуск остановлен')
    return code


def perform_generation(job):
    folder = RUNS / job['id']
    job.setdefault('_started_monotonic', time.monotonic())
    try:
        search = folder / 'search.json'
        cached = read_json(V1 / 'search.json')
        if job.get('clarification',{}).get('status') == 'answered':
            if not read_json(search):raise ValueError('Не найден сохранённый поиск для продолжения')
        elif job.get('reuse') and cached and cached.get('query') == job['query']:
            search.write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')
        else:
            code = execute(job, [sys.executable, '-B', str(V1/'retrieval.py'), 'search', '--index', str(INDEX), '--query', job['query'], '--output', str(search), '--local-only'], 'Поиск рецептов')
            if code:
                raise RuntimeError('Ошибка поиска. Подробности в журнале.')
        if job.get('allow_clarifications') and not job.get('clarification_checked'):
            code = execute(job,[sys.executable,'-B',str(Path(__file__).with_name('clarify_recipe.py')),
                                '--search',str(search),'--output',str(folder/'clarification.json')],
                           'Уточнение необходимых продуктов')
            info = read_json(folder/'clarification.json')
            if code or not info or info.get('action') not in ('ask','proceed'):
                raise RuntimeError('Не удалось подготовить уточнение продуктов. Подробности в журнале.')
            job['clarification_checked'] = True
            if info['action'] == 'ask':
                if not info.get('question') or not info.get('ingredients'):
                    raise ValueError('Уточнение не содержит вопроса и продуктов')
                job.update(status='waiting_input',stage='Ожидаю ответ о продуктах',clarification={
                    'id':uuid.uuid4().hex,'status':'waiting','question':info['question'],'ingredients':info['ingredients']})
                return
        initial = read_json(search)
        clarification = job.get('clarification',{})
        effective = job['query']
        if clarification.get('status') == 'answered':
            effective += '\nУточнение помощника: '+clarification['question']+'\nОтвет пользователя: '+clarification['answer']
        effective += ('\nПри составлении рецепта сохрани явные запреты пользователя. '
                      'Если закрытого списка продуктов нет, необходимые дополнительные продукты допустимы: '
                      'перечисли их в ингредиентах и явно отметь, что ещё нужно иметь или приобрести.')
        initial.update(query=effective,original_user_query=job['query'],retrieval_query=initial.get('query'),
                       user_clarification=clarification if clarification else None)
        search = folder/'generation_search.json'
        save_chat(search,initial)
        if job['mode'] == 'v1':
            command = [sys.executable, '-B', str(V1/'answer_recipe.py'), '--search', str(search)]
        else:
            command = [sys.executable, '-B', str(V2/'agent_recipe.py'), '--search', str(search), '--index', str(INDEX)]
        code = execute(job, command + ['--output', str(folder/'answer.json'), '--local-only'], 'Генерация рецепта' if job['mode']=='v1' else 'Работа агентов')
        result = read_json(folder/'answer.json') or {}
        final_status = 'done' if code == 0 else ('stopped' if code == 2 else 'error')
        job['error'] = result.get('error') or result.get('reason') if code else None
        if code and not job['error']:
            job['error'] = 'Процесс завершился с ошибкой. Подробности в журнале.'
        if not job.get('_cancel') and result:
            try:
                recipe_context(result)
            except ValueError:
                pass
            else:
                try:
                    execute(job, format_command(job), 'Оформление рецепта')
                except InterruptedError:
                    raise
                except Exception:
                    pass  # The original recipe remains usable if formatting fails.
        job['status'] = final_status
    except InterruptedError as exc:
        job.update(status='cancelled', error=str(exc))
    except Exception as exc:
        job.update(status='error', error=str(exc))
    finally:
        with LOCK:
            if job['status'] == 'waiting_input':job.pop('finished',None)
            else:job['finished'] = time.time()
            job['elapsed_seconds'] = max(0,job.get('_elapsed_before',0)+time.monotonic()-job['_started_monotonic'])
            persist(job)


def worker(job):
    global ACTIVE
    try:
        perform_generation(job)
    finally:
        with LOCK:ACTIVE = None


def start_job(body):
    global ACTIVE
    query = body.get('query', '')
    mode = body.get('mode')
    allow_clarifications=body.get('allow_clarifications',False)
    if type(allow_clarifications) is not bool:raise ValueError('Уточнение продуктов включается галочкой')
    if not isinstance(query, str) or not query.strip() or len(query)>4000 or mode not in ('v1','v2'):
        raise ValueError('Введите запрос до 4000 символов и выберите версию.')
    if not (INDEX/'manifest.json').exists():
        raise ValueError('Индекс recipe_index не найден.')
    if mode == 'v2' and not (V2/'agent_recipe.py').exists():
        raise ValueError('Не найдена соседняя папка version_2/ainura_local.')
    with LOCK:
        if ACTIVE:
            raise RuntimeError('Уже выполняется запуск. Дождитесь завершения или остановите его.')
        job = dict(id=uuid.uuid4().hex, query=query.strip(), mode=mode, reuse=body.get('reuse') is True,
                   status='running', stage='Подготовка', started=time.time(), _started_monotonic=time.monotonic(),
                   allow_clarifications=allow_clarifications)
        (RUNS/job['id']).mkdir(parents=True)
        JOBS[job['id']] = job
        ACTIVE = job['id']
        persist(job)
        threading.Thread(target=worker, args=(job,), daemon=True).start()
        return job['id']


def format_command(job):
    folder = RUNS/job['id']
    return [sys.executable,'-B',str(Path(__file__).with_name('format_recipe.py')),
            '--result',str(folder/'answer.json'),'--output',str(folder/'presentation.json')]


def respond_clarification(job_id, body):
    global ACTIVE
    with LOCK:
        job=JOBS.get(job_id)
        if not job or job['status'] != 'waiting_input':
            raise ValueError('В этом запуске нет ожидающего уточнения')
        if ACTIVE:
            raise RuntimeError('Дождитесь завершения текущей работы модели')
        info=job.get('clarification',{})
        if body.get('clarification_id') != info.get('id'):
            raise ValueError('Уточнение устарело. Обновите карточку.')
        choice=body.get('choice')
        ingredients=', '.join(info['ingredients'])
        if choice == 'allow':
            answer='Можно добавить следующие продукты: '+ingredients+'.'
        elif choice == 'deny':
            answer='Нельзя добавлять следующие продукты: '+ingredients+'. Предложи подходящий вариант без них.'
        elif choice == 'skip':
            answer='Продолжай без дополнительных уточнений; явно перечисли, какие ещё продукты потребуются.'
        elif choice == 'custom':
            answer=body.get('answer')
            if not isinstance(answer,str) or not answer.strip() or len(answer)>4000:
                raise ValueError('Введите ответ до 4000 символов')
            answer=answer.strip()
        else:
            raise ValueError('Выберите ответ на уточнение')
        info.update(status='answered',answer=answer,choice=choice)
        job.update(status='running',stage='Продолжаю с учётом ответа',
                   _elapsed_before=job.get('elapsed_seconds',0),_started_monotonic=time.monotonic())
        job.pop('_cancel',None);job.pop('finished',None);job.pop('error',None)
        persist(job)
        ACTIVE=job_id
        threading.Thread(target=worker,args=(job,),daemon=True).start()
        return job_id


def format_worker(job):
    global ACTIVE
    try:
        execute(job,format_command(job),'Оформление рецепта')
    except Exception as error:
        save_chat(RUNS/job['id']/'presentation.json',
                  {'status':'error','error':str(error),'documents':[],'llm_calls':0})
    finally:
        with LOCK:
            ACTIVE = None


def start_format(job_id):
    global ACTIVE
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise ValueError('Запуск не найден')
        if ACTIVE or job['status'] == 'running':
            raise RuntimeError('Дождитесь завершения текущего запуска или ответа')
        format_documents(read_json(RUNS/job_id/'answer.json') or {})
        job.pop('_cancel',None)
        ACTIVE = 'format:'+job_id
        threading.Thread(target=format_worker,args=(job,),daemon=True).start()
        return job_id


def question_worker(job, turn_id):
    global ACTIVE
    path = RUNS / job['id'] / 'chat.json'
    try:
        command = [sys.executable, '-B', str(Path(__file__).with_name('recipe_chat.py')),
                            '--result', str(path.with_name('answer.json')),
                            '--chat', str(path), '--turn', turn_id]
        code = execute(job, command, 'Ответ на вопрос')
        turns = read_json(path) or []
        turn = next(t for t in turns if t['id'] == turn_id)
        if turn['status'] == 'agent_requested':
            if not turn.get('allow_agents'):
                raise ValueError('Вызов агентов для этого вопроса не разрешён')
            child = agent_question_job(job, turn, path, turns)
            if child['status'] == 'waiting_input':
                turn.update(status='done',answer=child['clarification']['question'],
                            needs_clarification=True,finished=time.time(),stage='')
                save_chat(path,turns)
                return
            code = execute(job, command+['--agent-result',str(RUNS/child['id']/'answer.json')],
                           'Ответ по результату агентов')
            turns = read_json(path) or []
            turn = next(t for t in turns if t['id'] == turn_id)
        if turn['status'] == 'running':
            turn.update(status='error',error=f'Не удалось получить ответ (код {code})',finished=time.time())
            save_chat(path, turns)
    except Exception as error:
        turns = read_json(path) or []
        for turn in turns:
            if turn['id'] == turn_id:
                turn.update(status='error',error=str(error),finished=time.time())
        save_chat(path,turns)
    finally:
        with LOCK:
            ACTIVE = None


def agent_question_job(parent, turn, chat_path, turns):
    query = turn.get('agent_query')
    if not isinstance(query,str) or not query.strip() or len(query)>4000:
        raise ValueError('Некорректный запрос агентам')
    child = dict(id=uuid.uuid4().hex,query=query.strip(),mode='v2',status='running',reuse=False,
                 stage='Подбор по вопросу',started=time.time(),_started_monotonic=time.monotonic(),
                 parent_id=parent['id'],allow_clarifications=turn.get('allow_clarifications',False))
    folder = RUNS/child['id']; folder.mkdir()
    with LOCK:
        JOBS[child['id']] = child
        persist(child)
    turn.update(status='running',stage='Агенты подбирают рецепт…',agent_used=True,child_id=child['id'])
    save_chat(chat_path,turns)
    perform_generation(child)
    if child['status'] != 'waiting_input' and not (folder/'answer.json').exists():
        save_chat(folder/'answer.json',{'status':child['status'],'query':query,'reason':child.get('error')})
    turn['stage'] = 'Готовлю ответ по результату агентов…'
    save_chat(chat_path,turns)
    return child


def start_question(job_id, body):
    global ACTIVE
    question = body.get('question')
    allow_clarifications=body.get('allow_clarifications',False)
    if type(allow_clarifications) is not bool:raise ValueError('Уточнение продуктов включается галочкой')
    allow_agents = body.get('allow_agents',False)
    if type(allow_agents) is not bool:
        raise ValueError('Разрешение вызова агентов должно быть галочкой')
    if not isinstance(question,str) or not question.strip() or len(question)>4000:
        raise ValueError('Введите вопрос до 4000 символов')
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise ValueError('Запуск не найден')
        if ACTIVE or job['status'] == 'running':
            raise RuntimeError('Дождитесь завершения текущего запуска или ответа')
        path = RUNS/job_id/'chat.json'
        recipe_context(read_json(path.with_name('answer.json')) or {})
        turns = read_json(path) or []
        turn = dict(id=uuid.uuid4().hex,question=question.strip(),status='running',started=time.time(),
                    allow_agents=allow_agents,allow_clarifications=allow_clarifications)
        turns.append(turn)
        save_chat(path,turns)
        job.pop('_cancel', None)
        ACTIVE = 'question:'+job_id
        threading.Thread(target=question_worker,args=(job,turn['id']),daemon=True).start()
        return turn['id']


class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, content_type='application/json; charset=utf-8'):
        raw = json.dumps(data, ensure_ascii=False).encode() if not isinstance(data, bytes) else data
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            return self.send(200, (Path(__file__).parent/'index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/jobs':
            with LOCK:
                jobs = [{k:v for k,v in j.items() if not k.startswith('_')} for j in JOBS.values()]
            cached = read_json(V1/'search.json') or {}
            return self.send(200, {'jobs':sorted(jobs,key=lambda j:j['started'],reverse=True), 'busy':bool(ACTIVE), 'cached_query':cached.get('query'), 'v2':(V2/'agent_recipe.py').exists()})
        if path.startswith('/api/jobs/'):
            job_id = path.rsplit('/',1)[-1]
            with LOCK:
                job = JOBS.get(job_id)
                data = snapshot(job) if job else None
            return self.send(200 if data else 404, data or {'error':'Запуск не найден'})
        self.send(404, {'error':'Не найдено'})

    def do_POST(self):
        # JSON-only, same-origin requests; server binds to loopback only.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host',''):
            return self.send(403, {'error':'Недопустимый источник запроса'})
        if self.headers.get_content_type() != 'application/json':
            return self.send(415, {'error':'Ожидается JSON'})
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 20000:
                raise ValueError('Некорректный размер запроса')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body,dict):
                raise ValueError('Ожидается объект')
            if self.path == '/api/jobs':
                return self.send(202, {'id':start_job(body)})
            parts = self.path.strip('/').split('/')
            if len(parts) == 4 and parts[:2] == ['api','jobs'] and parts[3] == 'questions':
                return self.send(202, {'id':start_question(parts[2],body)})
            if len(parts) == 4 and parts[:2] == ['api','jobs'] and parts[3] == 'format':
                return self.send(202, {'id':start_format(parts[2])})
            if len(parts) == 4 and parts[:2] == ['api','jobs'] and parts[3] == 'clarification':
                return self.send(202, {'id':respond_clarification(parts[2],body)})
            if self.path.endswith('/cancel'):
                job_id = self.path.split('/')[-2]
                with LOCK:
                    job = JOBS.get(job_id)
                    if not job or job['status'] != 'running':
                        raise ValueError('Нет активного запуска')
                    job['_cancel'] = True
                    proc = job.get('_process')
                    if proc and proc.poll() is None:
                        stop_tree(proc)
                return self.send(200, {'ok':True})
            self.send(404, {'error':'Не найдено'})
        except (ValueError, TypeError) as exc:
            self.send(400, {'error':str(exc)})
        except RuntimeError as exc:
            self.send(409, {'error':str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    for path in RUNS.glob('*/job.json'):
        job = read_json(path)
        if job and job.get('id') == path.parent.name:
            if job['status']=='running':
                job.update(status='interrupted', error='Сервер был перезапущен')
            JOBS[job['id']] = job
            chat_path = path.with_name('chat.json')
            turns = read_json(chat_path) or []
            for turn in turns:
                if turn.get('status') in ('running','agent_requested'):
                    turn.update(status='error',error='Ответ прерван перезапуском сервера',finished=time.time())
            if turns:
                save_chat(chat_path,turns)
    server = ThreadingHTTPServer(('127.0.0.1',args.port), Handler)
    print(f'AIнура: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with LOCK:
            for job in JOBS.values():
                proc = job.get('_process')
                if proc and proc.poll() is None:
                    job['_cancel'] = True
                    stop_tree(proc)
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
