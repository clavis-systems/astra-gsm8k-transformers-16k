import json, re, math, sys, random
from fractions import Fraction
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data") + "/"
BS = chr(92)
dom = [json.loads(l) for l in open(D + "domande.jsonl", encoding="utf-8")]
prim = sorted(d["i"] for d in dom)[:100]
oro = {d["i"]: d["oro"] for d in dom}
RUNS = ['base__pensa__c1', 'modificato__pensa__c1', 'base__diretto__c1', 'modificato__diretto__c1']


def load(n):
    rs = [json.loads(l) for l in open(D + "risposte/" + n + ".jsonl", encoding="utf-8")]
    for r in rs:
        r["_run"] = n
    return rs


def boxed_all(t):
    out = []
    k = 0
    key = BS + "boxed"
    while True:
        p = t.find(key, k)
        if p < 0:
            break
        q = p + len(key)
        while q < len(t) and t[q] == ' ':
            q += 1
        if q < len(t) and t[q] == '{':
            dpt = 0
            j = q
            while j < len(t):
                if t[j] == '{':
                    dpt += 1
                elif t[j] == '}':
                    dpt -= 1
                    if dpt == 0:
                        break
                j += 1
            if dpt == 0:
                out.append((p, t[q + 1:j]))
                k = j + 1
            else:
                k = q + 1
        else:
            k = q
    return out


TXT = re.compile(re.escape(BS) + r"(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}")


def clean(s):
    s = TXT.sub(r"\1", s)
    for x in [BS + "$", "$", BS + "%", "%", BS + "!", BS + ",", BS + " ", ",", BS + "left", BS + "right"]:
        s = s.replace(x, "")
    return s.strip()


NUM = r"-?\d+(?:\.\d+)?"
FR = re.compile(r"\s*(-?)" + re.escape(BS) + r"d?frac\{(" + NUM + r")\}\{(" + NUM + r")\}\s*")


def tonum(s, lenient=False):
    s = clean(s)
    m = re.fullmatch(r"\s*(" + NUM + r")\s*", s)
    if m:
        return Fraction(m.group(1))
    m = FR.fullmatch(s)
    if m and Fraction(m.group(3)) != 0:
        return (-1 if m.group(1) else 1) * Fraction(m.group(2)) / Fraction(m.group(3))
    if lenient:
        ns = re.findall(NUM, s)
        if ns:
            return Fraction(ns[-1])
    return None


G = {i: Fraction(oro[i].replace(",", "")) for i in oro}


def judge(r, variant="base"):
    t = r["risposta"]
    g = G[r["i"]]
    bx = boxed_all(t)
    if variant == "dopothink":
        if "</think>" in t:
            cut = t.rfind("</think>")
            bx = [b for b in bx if b[0] > cut]
        elif "pensa" in r["_run"]:
            bx = []
    hasbox = len(bx) > 0
    st = False
    if hasbox:
        v = tonum(bx[-1][1], lenient=(variant == "lenparse"))
        st = v is not None and v == g
    ind = st
    if not hasbox and "</think>" in t:
        tail = t.split("</think>")[-1]
        ns = re.findall(NUM, clean(tail))
        if ns and Fraction(ns[-1]) == g:
            ind = True
    return st, ind, hasbox


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


if __name__ == "__main__":
    res = {}
    for variant in ["base", "lenparse", "dopothink"]:
        print("== variante", variant)
        for n in RUNS:
            rs = load(n)
            by = {r["i"]: r for r in rs}
            ids = [r["i"] for r in rs]
            comp = (sorted(ids) == prim and len(set(ids)) == 100)
            tet = set(r["tetto"] for r in rs)
            incoh = [r["i"] for r in rs if r["tagliata"] != (r["n_token"] >= r["tetto"])]
            J = {i: judge(by[i], variant) for i in prim}
            ks = sum(J[i][0] for i in prim)
            ki = sum(J[i][1] for i in prim)
            tag = sum(by[i]["tagliata"] for i in prim)
            nb = sum(not J[i][2] for i in prim)
            tok = sum(by[i]["n_token"] for i in prim) / 100
            lo, hi = wilson(ks, 100)
            print(f"{n}: complete={comp} n={len(rs)} tetti={tet} incoerenti={incoh} stretto={ks} W[{lo*100:.1f},{hi*100:.1f}] indulgente={ki} tagliate={tag} tokmedi={tok:.0f} senzabox={nb} cambiano={sum(J[i][0] != J[i][1] for i in prim)}")
            res[(variant, n)] = J
        for pr in ["pensa", "diretto"]:
            a = res[(variant, f"base__{pr}__c1")]
            b = res[(variant, f"modificato__{pr}__c1")]
            both = sum(a[i][0] and b[i][0] for i in prim)
            onlyb = sum((not a[i][0]) and b[i][0] for i in prim)
            onlya = sum(a[i][0] and not b[i][0] for i in prim)
            non = sum((not a[i][0]) and not b[i][0] for i in prim)
            print(pr, "entrambi", both, "solo modificato", onlyb, "solo base", onlya, "nessuno", non)
