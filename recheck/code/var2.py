import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc import *
def j2(r):
    bx=[b for b in boxed_all(r["risposta"]) if b[1].strip()]
    if not bx: return False
    v=tonum(bx[-1][1],lenient=True); return v is not None and v==G[r["i"]]
K={}
for n in RUNS:
    rs=[r for r in load(n) if r["i"] in prim]
    K[n]={r["i"]:j2(r) for r in rs}
    k=sum(K[n].values());print(n,k,[round(x*100,1) for x in wilson(k,100)],"diff-da-base-stretto:",[i for i in K[n] if K[n][i]!=judge([x for x in rs if x["i"]==i][0])[0]])
a,b=K[RUNS[0]],K[RUNS[1]]
ka,kb=sum(a.values()),sum(b.values());la,ha=wilson(ka,100);lb,hb=wilson(kb,100)
print("diff",kb-ka,"disgiunti",lb>ha)
# permutation-free sign test pensa: solo mod vs solo base
print("pensa solo mod",sum(b[i] and not a[i] for i in prim),"solo base",sum(a[i] and not b[i] for i in prim))
# truncated-correct combos
