import { MockRequest, MockStatusError } from '@delon/mock';

const datalogs = [];
const actlogs = [];
const total = 50;

for (let i = 0; i < total; i += 1) {
  datalogs.push({
    id: i + 1+'',
    no: `${i + 1}`,
    logType: `数据日志 ${i}`,
    userName: `曲丽丽 ${i}`,
    logTime: new Date(`2017-07-${Math.floor(i / 2) + 1}`),
  });
}
for (let i = 0; i < total; i += 1) {
  actlogs.push({
    id: i + 1+'',
    no: `${i + 1}`,
    ipAddress: `172.16.17.32`,
    userName: `曲丽丽 ${i}`,
    logMethod:'xxxxxxxx'+i,
    logTime: new Date(`2017-07-${Math.floor(i / 2) + 1}`),
  });
}


function getActLogList(params: any){
  debugger;
  let ret = [...actlogs];
  const pi = +params.pi,
    ps = +params.ps,
    start = (pi - 1) * ps;
  if (params.sorter) {
    const s = params.sorter.split('_');
    ret = ret.sort((prev, next) => {
      debugger;
      if (s[1] === 'descend') {
        return next[s[0]] - prev[s[0]];
      }
      return prev[s[0]] - next[s[0]];
    });
  }

  if (params.userName && params.userName.length > 0) {
    ret = ret.filter(data => data.userName.indexOf(params.userName) > -1);
  }
  return { total: ret.length, list: ret.slice(start, ps * pi) };
}
function getDataLogList(params: any){
  debugger;
  let ret = [...datalogs];
  const pi = +params.pi,
    ps = +params.ps,
    start = (pi - 1) * ps;
  if (params.sorter) {
    const s = params.sorter.split('_');
    ret = ret.sort((prev, next) => {
      debugger;
      if (s[1] === 'descend') {
        return next[s[0]] - prev[s[0]];
      }
      return prev[s[0]] - next[s[0]];
    });
  }

  if (params.userName && params.userName.length > 0) {
    ret = ret.filter(data => data.userName.indexOf(params.userName) > -1);
  }
  return { total: ret.length, list: ret.slice(start, ps * pi) };
}
function removeActLogById(nos: string): boolean {
  debugger;
  nos.split(',').forEach(id => {
    const idx = actlogs.findIndex(w => w.id == id);
    if (idx !== -1) actlogs.splice(idx, 1);
  });
  return true;
}
function removeDataLogById(nos: string): boolean {
  debugger;
  nos.split(',').forEach(id => {
    const idx = datalogs.findIndex(w => w.id == id);
    if (idx !== -1) datalogs.splice(idx, 1);
  });
  return true;
}

export const logs = {
  '/actLogList': (req: MockRequest) => getActLogList(req.queryString),
  '/dataLogList': (req: MockRequest) => getDataLogList(req.queryString),

  '/qs': (req: MockRequest) => req.queryString.pi,
  '/users/:id': (req: MockRequest) => req.params, // /users/100, output: { id: 100 }

  'DELETE /removeDataLogById': (req: MockRequest) => removeDataLogById(req.queryString.ids),
  'DELETE /removeActLogById': (req: MockRequest) => removeActLogById(req.queryString.ids),
  // 发送 Status 错误
  '/404': () => {
    throw new MockStatusError(404);
  },
};