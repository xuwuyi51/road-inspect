#!/usr/bin/env node
/**
 * 标注台前端（src/rdinspect/api/static/index.html）自查脚本 —— 零依赖，直接跑：
 *
 *     node scripts/check_annotator_ui.mjs
 *
 * 做法：把 index.html 里的内联 <script> 抽出来，在一个 Node + 手写 DOM 桩的 vm 沙箱里执行，
 * 用桩 fetch 记录所有请求，然后针对「模型辅助标注（预标注）」相关交互做断言：
 *   1) 候选行「忽略」→ DELETE /api/annotations/{id}，并从 state.anns / 列表 DOM 中移除；
 *   2) 候选行「采纳」→ 与 Z 同源的本地流转 model → model_edited，且 PUT 载荷带 source；
 *   3) 「预标注本图」→ POST /api/tasks/{id}/prelabel；501 时显示指定的 ML 依赖文案；
 *   4) 「批量预标注」→ body {limit:20,status:"pending"}；confirm 取消时不发请求；
 *   5) 质量看板：metrics 字段为 null 时显示 —，绝不出现 NaN；
 *   6) 兼容性：S 保存仍走「对齐 → PUT → submit」，且未处理的候选不会混进 PUT 载荷。
 *
 * 退出码 0 = 全部通过；1 = 有断言失败。
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(HERE, '..', 'src', 'rdinspect', 'api', 'static', 'index.html');
const html = readFileSync(HTML_PATH, 'utf8');

const m = html.match(/<script>\n([\s\S]*)\n<\/script>/);
if (!m) throw new Error('未在 index.html 中找到内联 <script>');
const scriptSrc = m[1];

/* ----------------------------- 极简 DOM 桩 ----------------------------- */
const CTX2D = new Proxy({ measureText: () => ({ width: 12 }) }, {
  get(t, k) {
    if (k in t) return t[k];
    return () => {};                 // setTransform/fillRect/... 一律空实现
  },
  set() { return true; }
});

function makeEl(tag) {
  const classes = new Set();
  const el = {
    tagName: String(tag).toUpperCase(),
    children: [], dataset: {}, style: {}, _ls: {},
    textContent: '', value: '', disabled: false, title: '', width: 0, height: 0,
    _innerHTML: '',
    appendChild(c) { c.parentNode = el; el.children.push(c); return c; },
    append(...cs) { cs.forEach((c) => el.appendChild(c)); },
    insertBefore(c, ref) {
      const i = el.children.indexOf(ref);
      if (i < 0) el.children.push(c); else el.children.splice(i, 0, c);
      c.parentNode = el; return c;
    },
    removeChild(c) { const i = el.children.indexOf(c); if (i >= 0) el.children.splice(i, 1); return c; },
    remove() { if (el.parentNode) el.parentNode.removeChild(el); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener(t, fn) { (el._ls[t] = el._ls[t] || []).push(fn); },
    removeEventListener() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600 }; },
    getContext() { return CTX2D; },
    setPointerCapture() {}, releasePointerCapture() {}, focus() {}, blur() {}, scrollIntoView() {},
  };
  el.classList = {
    add: (c) => classes.add(c),
    remove: (c) => classes.delete(c),
    contains: (c) => classes.has(c),
    toggle: (c, force) => {
      const on = force === undefined ? !classes.has(c) : !!force;
      if (on) classes.add(c); else classes.delete(c);
      return on;
    },
  };
  Object.defineProperty(el, 'className', {
    get: () => [...classes].join(' '),
    set: (v) => { classes.clear(); String(v).split(/\s+/).filter(Boolean).forEach((c) => classes.add(c)); },
  });
  Object.defineProperty(el, 'innerHTML', {
    get: () => el._innerHTML,
    set: (v) => { el._innerHTML = String(v); el.children.length = 0; },   // 与真实 DOM 一致：整体替换会清空子节点
  });
  return el;
}

const byId = new Map();
for (const id of new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((x) => x[1]))) byId.set(id, makeEl('div'));
const documentStub = {
  getElementById: (id) => byId.get(id) || (byId.set(id, makeEl('div')), byId.get(id)),
  createElement: (tag) => makeEl(tag),
  addEventListener() {},
  querySelector: () => null,
};
const windowStub = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

/* --------------------------- 计时器桩（可控） --------------------------- */
const pendingTimers = [];
function fakeSetTimeout(fn, ms) {
  if (!ms) return setTimeout(fn, 0);            // 0ms 仍然真跑，保证加载时序自然
  const t = { fn, ms };
  pendingTimers.push(t);
  return t;
}
function fakeClearTimeout(t) { const i = pendingTimers.indexOf(t); if (i >= 0) pendingTimers.splice(i, 1); }

/* ------------------------------ fetch 桩 ------------------------------ */
const calls = [];
const store = {
  prelabelStatus: 200,     // 501 = 未安装 ML 依赖
  prelabelShape: 'report', // report = 服务端 report 直出 {run_id,...}；async = 契约 {run:{id,status:running}}
  batchStatus: 200,
  metricsStatus: 200,
  metrics: { tasks_prelabeled: 3, candidates_total: 12, adopted: 5, ignored: 2, adoption_rate: 0.4167, model_human_iou_mean: 0.5512 },
  taskAnnotations: [],
  taskStatus: 'prelabeled',
  extraCandidates: [],     // 预标注 run 完成后才出现的候选
  deleteMiss: new Set(),   // 服务端已不存在的标注 id（DELETE → 404）
};
globalThis.__confirmAnswer = true;
const confirmCalls = [];

function resp(status, payload) {
  return { ok: status >= 200 && status < 300, status, statusText: String(status),
           text: async () => (payload === undefined ? '' : JSON.stringify(payload)) };
}
function route(url, method, body) {
  if (url === '/api/health') return resp(200, { status: 'ok' });
  if (url === '/api/classes') {
    return resp(200, [{ code: 'pothole', name_zh: '坑槽', name_en: 'pothole', color: '#e6194b', order_index: 1, active: true },
                      { code: 'crack', name_zh: '裂缝', name_en: 'crack', color: '#3cb44b', order_index: 2, active: true }]);
  }
  if (url.startsWith('/api/tasks?') || url === '/api/tasks') {
    return resp(200, { items: [{ id: 1, image_id: 11, status: store.taskStatus, priority: 5, prelabel_state: 'done' }], next_cursor: null });
  }
  if (/^\/api\/tasks\/\d+$/.test(url) && method === 'GET') {
    return resp(200, {
      id: 1, image_id: 11, status: store.taskStatus, prelabel_state: 'done',
      image: { id: 11, width: 640, height: 480 },
      annotations: store.taskAnnotations.concat(store.extraCandidates),
    });
  }
  if (url === '/api/tasks/1/prelabel') {
    if (store.prelabelStatus !== 200) {
      return resp(store.prelabelStatus, { code: 'ml_deps_missing', message: 'torch/ultralytics 未安装' });
    }
    if (store.prelabelShape === 'async') return resp(200, { run: { id: 6, kind: 'prelabel', status: 'running' } });
    return resp(200, { run_id: 5, requested: 1, processed: 1, candidates: 3, masks: 0, tiles: 1, task_id: 1 });
  }
  if (url === '/api/prelabel/batches') {
    if (store.batchStatus !== 200) {
      return resp(store.batchStatus, { code: 'ml_deps_missing', message: 'torch/ultralytics 未安装' });
    }
    return resp(200, { run: { id: 9, kind: 'prelabel', status: 'queued' }, requested: body && body.limit });
  }
  if (url === '/api/prelabel/metrics') {
    if (store.metricsStatus !== 200) return resp(store.metricsStatus, { code: 'boom', message: '指标不可用' });
    return resp(200, store.metrics);
  }
  if (/^\/api\/annotations\/\d+$/.test(url) && method === 'DELETE') {
    const id = Number(url.split('/').pop());
    if (store.deleteMiss.has(id)) return resp(404, { code: 'not_found', message: `标注 ${id} 不存在或已删除` });
    store.taskAnnotations = store.taskAnnotations.filter((a) => a.id !== id);   // 桩后端同步软删效果
    return resp(200, { deleted: true, id });
  }
  if (/^\/api\/tasks\/\d+\/annotations$/.test(url) && method === 'PUT') {
    let nextId = 100;
    const saved = (body.annotations || []).map((a) => ({
      ...a, id: nextId++, task_id: 1, image_id: 11, difficult: !!a.difficult, score: null,
      source: a.source || 'human', model_version_id: null,
    }));
    const pending = store.taskAnnotations.filter((a) => a.source === 'model');
    return resp(200, { annotations: saved.concat(pending), changed: saved.length, added: saved.length, updated: 0, deleted: 0 });
  }
  if (/^\/api\/tasks\/\d+\/submit$/.test(url) && method === 'POST') return resp(200, { id: 1, status: 'submitted' });
  return resp(404, { code: 'not_found', message: 'no route for ' + url });
}
const fetchStub = async (url, init = {}) => {
  const method = (init.method || 'GET').toUpperCase();
  const body = init.body ? JSON.parse(init.body) : null;
  calls.push({ url, method, body });
  return route(url, method, body);
};

/* ------------------------------- 沙箱 ------------------------------- */
const sandbox = {
  document: documentStub, console,
  devicePixelRatio: 1, addEventListener() {}, removeEventListener() {},   // window === globalThis
  fetch: fetchStub,
  localStorage: (() => { const s = new Map(); return {
    getItem: (k) => (s.has(k) ? s.get(k) : null), setItem: (k, v) => s.set(k, String(v)),
    removeItem: (k) => s.delete(k), clear: () => s.clear() }; })(),
  setTimeout: fakeSetTimeout, clearTimeout: fakeClearTimeout,
  confirm: (msg) => { confirmCalls.push(String(msg)); return globalThis.__confirmAnswer; },
  Image: class {
    constructor() { this.naturalWidth = 0; this.naturalHeight = 0; this.onload = null; this.onerror = null; }
    set src(v) { this._src = v; setTimeout(() => { this.naturalWidth = 640; this.naturalHeight = 480; if (this.onload) this.onload(); }, 0); }
    get src() { return this._src; }
  },
  Date, Math, JSON, Number, String, Array, Object, Map, Set, isFinite, parseInt, parseFloat, Promise, Error, Boolean,
};
sandbox.globalThis = sandbox;
const context = vm.createContext(sandbox);
sandbox.window = vm.runInContext('globalThis', context);   // 浏览器语义：window === globalThis

const EPILOGUE = `
globalThis.__T = { state, api, toPayload, openTask, saveAndSubmit, adoptCandidate, adoptAllCandidates,
  ignoreCandidate, prelabelCurrentTask, batchPrelabel, loadMetrics, renderMetrics, refreshCandidates,
  saveDraft, readDraft, clearDraft, ML_MISSING_HINT };`;
vm.runInContext(scriptSrc + EPILOGUE, context, { filename: 'index.inline.js' });
const T = sandbox.__T;

/* ------------------------------ 断言工具 ------------------------------ */
let pass = 0;
const failures = [];
function ok(name, cond, extra) {
  if (cond) { pass++; console.log('  ✓ ' + name); }
  else { failures.push(name + (extra ? ' → ' + extra : '')); console.log('  ✗ ' + name + (extra ? ' → ' + extra : '')); }
}
const eq = (name, actual, expected) => ok(name, actual === expected, `实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const flush = async () => { for (let i = 0; i < 6; i++) await sleep(0); };
const lastCall = (url, method) => [...calls].reverse().find((c) => c.url === url && (!method || c.method === method));
const findCalls = (url, method) => calls.filter((c) => c.url === url && (!method || c.method === method));
const bannerText = () => byId.get('banners').children.map((b) => b.children.map((c) => c.textContent).join(' ')).join(' | ');
/** 递归收集元素（DOM 桩里只有 createElement 出来的节点才是真实子节点） */
function walk(el, out = []) { for (const c of el.children) { out.push(c); walk(c, out); } return out; }
function buttonsOf(row) { return walk(row).filter((e) => e.tagName === 'BUTTON' && e.dataset.act); }
function actBtn(row, act) { return buttonsOf(row).find((b) => b.dataset.act === act); }
const click = async (btn) => { btn.onclick({ stopPropagation() {} }); await flush(); };

const A = (id, source, x1) => ({
  id, class_code: 'pothole', kind: 'bbox', bbox: { x1, y1: 0.1, x2: x1 + 0.2, y2: 0.3 },
  difficult: false, score: source === 'model' ? 0.82 : null, source, model_version_id: source === 'model' ? 3 : null,
});

/* -------------------------------- 用例 -------------------------------- */
await flush();                          // 等 boot() 跑完

console.log('\n[0] 启动');
ok('boot 触发 /api/health', !!lastCall('/api/health'));
ok('boot 触发 /api/classes', !!lastCall('/api/classes'));
ok('boot 加载质量看板（/api/prelabel/metrics）', !!lastCall('/api/prelabel/metrics'));
ok('无外部依赖：无 http/https 外链', !/https?:\/\//.test(html));

console.log('\n[1] 候选面板：渲染 + 采纳 + 忽略');
store.taskAnnotations = [A(7, 'model', 0.10), A(8, 'model', 0.40), A(9, 'human', 0.70)];
await T.openTask({ id: 1, image_id: 11, status: 'prelabeled' });
await flush();
eq('加载后 state.anns 3 条', T.state.anns.length, 3);
let rows = byId.get('annList').children;
eq('列表渲染 3 行', rows.length, 3);
eq('候选行有 2 个动作按钮', buttonsOf(rows[0]).length, 2);
eq('按钮文案为 采纳/忽略', buttonsOf(rows[0]).map((b) => b.textContent).join(','), '采纳,忽略');
eq('人工行没有动作按钮', buttonsOf(rows[2]).length, 0);

await click(actBtn(rows[0], 'adopt'));
eq('采纳后 source=model_edited', T.state.anns.find((a) => a.id === 7).source, 'model_edited');
ok('采纳后标记为脏（草稿逻辑照旧）', T.state.dirty === true);
eq('采纳后该行动作按钮消失（仅剩 1 个候选行）', buttonsOf(byId.get('annList').children[0]).length, 0);
const payload = T.toPayload(T.state.anns);
eq('PUT 载荷排除未处理候选（3 → 2）', payload.length, 2);
eq('已采纳项带 source=model_edited', payload[0].source, 'model_edited');
ok('人工项不带多余 source 字段', payload[1].source === undefined);

rows = byId.get('annList').children;
await click(actBtn(rows[1], 'ignore'));           // 行 1 = 候选 #8
const del = lastCall('/api/annotations/8', 'DELETE');
ok('忽略发出 DELETE /api/annotations/8', !!del, JSON.stringify(calls.filter((c) => c.method === 'DELETE')));
ok('忽略从 state.anns 移除该项', !T.state.anns.some((a) => a.id === 8));
eq('忽略后列表剩 2 行', byId.get('annList').children.length, 2);
ok('忽略后候选计数刷新', byId.get('metaCand').textContent === '候选 0', byId.get('metaCand').textContent);

console.log('\n[2] 预标注本图：501 提示 / 成功刷新候选');
store.taskAnnotations = [A(9, 'human', 0.70)];
T.state.anns = T.state.anns.filter((a) => a.id !== 7);
store.prelabelStatus = 501;
await click(byId.get('btnPrelabel'));
const post = lastCall('/api/tasks/1/prelabel', 'POST');
ok('发出 POST /api/tasks/1/prelabel', !!post);
ok('501 时提示指定的 ML 依赖文案', bannerText().includes("未安装 ML 依赖：用 .venv/bin/pip install -e '.[ml]' 安装后重试"), bannerText());
ok('501 文案与代码常量一致', sandbox.__T.ML_MISSING_HINT === "未安装 ML 依赖：用 .venv/bin/pip install -e '.[ml]' 安装后重试");

store.prelabelStatus = 200;
store.extraCandidates = [A(12, 'model', 0.55)];
const before = T.state.anns.length;
await click(byId.get('btnPrelabel'));
ok('成功时再次 POST /api/tasks/1/prelabel', findCalls('/api/tasks/1/prelabel', 'POST').length === 2);
eq('成功后 GET /api/tasks/1 刷新候选', T.state.anns.length, before + 1);
ok('新候选进入本地状态（source=model）', T.state.anns.some((a) => a.id === 12 && a.source === 'model'));
ok('服务端 report 形态（{run_id,...}）提示 run #5 与候选数',
   bannerText().includes('预标注已完成：run #5（任务 #1，候选 3 个）'), bannerText());
store.prelabelShape = 'async';
await click(byId.get('btnPrelabel'));
ok('异步 run 形态（{run:{id,status}}）提示 run #6', bannerText().includes('已提交预标注任务：run #6'), bannerText());
store.prelabelShape = 'report';

console.log('\n[2b] 忽略：服务端已不存在（404）时按已忽略处理');
store.extraCandidates = [A(99, 'model', 0.60)];      // 陈旧候选（例如服务端已被别处删除）
await T.refreshCandidates(1);
ok('陈旧候选已进入列表', T.state.anns.some((a) => a.id === 99));
store.deleteMiss.add(99);
const listRows = byId.get('annList').children;
await click(actBtn(listRows[listRows.length - 1], 'ignore'));
ok('DELETE 404 时仍从本地移除（陈旧候选可清理）', !T.state.anns.some((a) => a.id === 99));
ok('404 不产生「忽略候选失败」红条', !bannerText().includes('忽略候选失败'), bannerText());

console.log('\n[3] 批量预标注');
globalThis.__confirmAnswer = true;
const batchesBefore = findCalls('/api/prelabel/batches', 'POST').length;
await click(byId.get('btnBatchPrelabel'));
const batch = lastCall('/api/prelabel/batches', 'POST');
ok('发出 POST /api/prelabel/batches', !!batch);
eq('请求体为 {limit:20,status:"pending"}', JSON.stringify(batch && batch.body), JSON.stringify({ limit: 20, status: 'pending' }));
ok('成功提示 run 与张数', bannerText().includes('已提交预标注任务：run #9，共 20 张'), bannerText());
ok('批量按钮先 confirm 确认', confirmCalls.some((c) => c.includes('批量预标注')));
globalThis.__confirmAnswer = false;
await click(byId.get('btnBatchPrelabel'));
eq('confirm 取消时不发请求', findCalls('/api/prelabel/batches', 'POST').length, batchesBefore + 1);
globalThis.__confirmAnswer = true;

console.log('\n[4] 质量看板');
store.metrics = { tasks_prelabeled: null, candidates_total: null, adopted: null, ignored: null, adoption_rate: null, model_human_iou_mean: null };
await T.loadMetrics();
const vals = ['mTasks', 'mCand', 'mAdopt', 'mIgnore', 'mRate', 'mIou'].map((id) => byId.get(id).textContent);
eq('全 null 时 6 项均显示 —', vals.join(','), '—,—,—,—,—,—');
ok('全 null 时不出现 NaN', vals.every((v) => !/NaN/.test(v)), vals.join(','));
store.metrics = { tasks_prelabeled: 3, candidates_total: 12, adopted: 5, ignored: 2, adoption_rate: 0.4167, model_human_iou_mean: 0.5512 };
await T.loadMetrics();
eq('采纳率转百分比（1 位小数）', byId.get('mRate').textContent, '41.7%');
eq('IoU 保留 2 位小数', byId.get('mIou').textContent, '0.55');
eq('计数为整数文本', byId.get('mTasks').textContent, '3');
T.renderMetrics({ adoption_rate: 0.5, model_human_iou_mean: 0 });
eq('0 不是缺失值（IoU=0 → 0.00）', byId.get('mIou').textContent, '0.00');
store.metrics = undefined; store.metricsStatus = 500;
await T.loadMetrics();
ok('指标接口失败走顶部提示条且不崩溃', bannerText().includes('预标注指标加载失败'), bannerText());
store.metricsStatus = 200;
store.metrics = { tasks_prelabeled: 4, candidates_total: 13, adopted: 6, ignored: 2, adoption_rate: 0.4615, model_human_iou_mean: 0.55 };
ok('可折叠面板存在且默认为展开', !byId.get('metrics').classList.contains('collapsed'));
byId.get('metricsHead').onclick();
ok('点击表头可折叠', byId.get('metrics').classList.contains('collapsed'));
byId.get('metricsHead').onclick();

console.log('\n[5] 兼容性：S 保存并提交');
// 服务端真值：候选 #7 仍是 model（用户在本地按了「采纳」但还没保存），#9 是人工框
store.extraCandidates = [];
store.taskAnnotations = [A(7, 'model', 0.10), A(9, 'human', 0.70)];
T.state.anns = [
  { uid: 's7', id: 7, class_code: 'pothole', kind: 'bbox', bbox: { x1: 0.10, y1: 0.10, x2: 0.30, y2: 0.30 },
    source: 'model_edited', score: 0.82, difficult: false, color: '#e6194b' },
  { uid: 's9', id: 9, class_code: 'pothole', kind: 'bbox', bbox: { x1: 0.70, y1: 0.10, x2: 0.90, y2: 0.30 },
    source: 'human', score: null, difficult: false, color: '#e6194b' },
];
await T.refreshCandidates(1);
ok('候选 #7 仍在服务端候选集合（candidateIds）中', T.state.candidateIds.has(7));
calls.length = 0;
await T.saveAndSubmit();
await flush();
const seq = calls.map((c) => c.method + ' ' + c.url);
ok('保存前先对齐已采纳候选（DELETE /api/annotations/7）', seq.indexOf('DELETE /api/annotations/7') >= 0, seq.join(' ， '));
ok('DELETE 早于 PUT', seq.indexOf('DELETE /api/annotations/7') < seq.findIndex((s) => s.startsWith('PUT')));
const put = calls.find((c) => c.method === 'PUT');
ok('仍发出 PUT /api/tasks/1/annotations', !!put);
ok('PUT 载荷不含未处理候选（source=model）', put.body.annotations.every((a) => a.source !== 'model'), JSON.stringify(put.body.annotations));
eq('PUT 载荷条数 = 人工 + 已采纳', put.body.annotations.length, 2);
ok('已采纳项以 model_edited 提交', put.body.annotations.some((a) => a.source === 'model_edited'));
ok('提交仍发出 POST /api/tasks/1/submit', seq.some((s) => s === 'POST /api/tasks/1/submit'), seq.join(' ， '));

console.log('\n[5b] 回归：二次保存不误删已落库的 model_edited');
ok('首次保存后本地仍有 model_edited 行', T.state.anns.some((a) => a.source === 'model_edited'));
calls.length = 0;
await T.saveAndSubmit();
await flush();
eq('二次保存没有多余 DELETE', calls.filter((c) => c.method === 'DELETE').length, 0);
ok('二次保存仍发出 PUT', calls.some((c) => c.method === 'PUT'));
ok('已落库的 model_edited 仍在本地状态（未被误删）', T.state.anns.some((a) => a.source === 'model_edited'));

console.log('\n[6] 兼容性：快捷键映射与草稿逻辑');
ok("Z 仍绑定 adoptAllCandidates", /case 'z': case 'Z': adoptAllCandidates\(\)/.test(scriptSrc));
ok("S 仍绑定 saveAndSubmit", /case 's': case 'S': saveAndSubmit\(\)/.test(scriptSrc));
ok('Delete/D/A/C/Q/E 等键映射未改', /case 'Delete': case 'Backspace': deleteSelected\(\)/.test(scriptSrc)
  && /case 'c': case 'C': copyFromPrevious\(\)/.test(scriptSrc) && /case 'q': case 'Q': zoomAt/.test(scriptSrc));
T.state.anns = [
  { uid: 'z1', id: null, class_code: 'pothole', kind: 'bbox', bbox: { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 }, source: 'model', score: 0.5, difficult: false, color: '#e6194b' },
  { uid: 'z2', id: null, class_code: 'pothole', kind: 'bbox', bbox: { x1: 0.3, y1: 0.1, x2: 0.4, y2: 0.2 }, source: 'model', score: 0.4, difficult: false, color: '#e6194b' },
  { uid: 'z3', id: null, class_code: 'pothole', kind: 'bbox', bbox: { x1: 0.5, y1: 0.1, x2: 0.6, y2: 0.2 }, source: 'human', score: null, difficult: false, color: '#e6194b' },
];
T.state.dirty = false;
T.adoptAllCandidates();
eq('Z 采纳全部：3 条中的 2 条候选转为 model_edited', T.state.anns.filter((a) => a.source === 'model_edited').length, 2);
ok('Z 提示文案与原来一致', bannerText().includes('已采纳 2 个模型候选，可继续调整'), bannerText());
ok('Z 仍标记脏（走同一套草稿流程）', T.state.dirty === true);
T.saveDraft(true);
const draft = T.readDraft(1);
ok('草稿仍写入 localStorage 且保留 source/score', !!draft && draft.annotations.length === 3
  && draft.annotations.filter((a) => a.source === 'model_edited').length === 2, JSON.stringify(draft && draft.annotations.map((a) => a.source)));

/* ------------------------------- 汇总 ------------------------------- */
console.log('\n──────────────────────────────────────────');
console.log(failures.length ? `✗ 失败 ${failures.length} 项 / 通过 ${pass} 项` : `✓ 全部通过（${pass} 项断言）`);
failures.forEach((f) => console.log('  - ' + f));
process.exit(failures.length ? 1 : 0);
