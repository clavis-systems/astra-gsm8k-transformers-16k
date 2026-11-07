import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc import *
r=[x for x in load("modificato__pensa__c1") if x["i"]==52][0]
t=r["risposta"];print(r["n_token"],r["tagliata"],t.count("</think>"),t.find("</think>"),len(t))
print(t[t.find("</think>")-200:t.find("</think>")+300])
# diretto base senza box che sono giuste con ultimo numero? 
for n in RUNS[2:]:
    for x in load(n):
        if x["i"] in prim and not boxed_all(x["risposta"]): print(n,x["i"],x["tagliata"],oro[x["i"]],repr(x["risposta"][-80:]))
