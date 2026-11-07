import sys,random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc import *
# parse failures and strict!=indulgent
for n in RUNS:
    rs=load(n)
    for r in rs:
        bx=boxed_all(r["risposta"])
        if bx and tonum(bx[-1][1]) is None: print("NONPARSE",n,r["i"],repr(bx[-1][1]),"oro",oro[r["i"]])
        st,ind,hb=judge(r)
        if st!=ind: print("STR!=IND",n,r["i"],"oro",oro[r["i"]],"coda:",repr(r["risposta"][-200:]))
        # boxed exists but only in thinking among truncated & correct?
        if r["tagliata"] and st: print("TAGLIATA_GIUSTA",n,r["i"])
random.seed(20261107)
rs=load("modificato__pensa__c1")
J={r["i"]:judge(r) for r in rs}
by={r["i"]:r for r in rs}
gi=random.sample([i for i in prim if J[i][0]],5)
sb=random.sample([i for i in prim if not J[i][0]],5)
for lab,L in (("GIUSTA",gi),("SBAGLIATA",sb)):
    for i in L:
        r=by[i];t=r["risposta"]
        print("=====",lab,i,"oro",oro[i],"ntok",r["n_token"],"tagl",r["tagliata"],"boxed",[b[1] for b in boxed_all(t)][-3:])
        print(t[-450:].replace("\n"," | "))
