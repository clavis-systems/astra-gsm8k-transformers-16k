"""Rimisura indipendente della modifica di pesi "Astra" su Qwen/Qwen3.5-0.8B. Regole, previsioni e deviazioni dichiarate
in PREMESSA.md (congelata prima di ogni corsa).

Sottocomandi (dalla cartella di questo file):
  prepara  --base DIR --modifica DIR --codice DIR --out DIR [--gsm8k PARQUET] [--n 200]
           verifica l'impronta dei pesi di base, ricostruisce i due tensori modificati con il codice ufficiale dei log
           (apply_edit.py, che controlla le impronte dei tensori), confronta i due file byte per byte con un lettore
           nostro, scrive verifica_pesi.json e le domande scelte (domande.jsonl).
  genera   --out DIR --base DIR --modello base|modificato --protocollo pensa|diretto --campione 1|2 [--lotto 8]
           [--max-nuovi N]   (riprende dalle domande già fatte; lucchetto GPU)
  analizza --out DIR         (punteggi, intervalli, confronto appaiato, verdetto secondo la premessa)
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time

QUI = os.path.dirname(os.path.abspath(__file__))
LUCCHETTO = os.path.normpath(os.path.join(QUI, "..", "..", "risultati", ".gpu_lock"))
GSM8K_PREDEFINITO = os.path.normpath(os.path.join(QUI, *[".."] * 6, "esterni", "gsm8k", "main",
                                                  "test-00000-of-00001.parquet"))

PESI_BASE = "model.safetensors-00001-of-00001.safetensors"
REVISIONE_BASE = "2fc06364715b967f1860aea9cf38778875588b17"
SHA_BASE = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"   # LFS oid del file dei pesi a quella revisione
ISTRUZIONE = "Please reason step by step, and put your final answer within \\boxed{}."   # mathqa.build_prompt dei log
SEME_DOMANDE = 20261104
N_DOMANDE = 200          # scelte con il seme; le prime PRIMARIE (in ordine di indice) decidono il verdetto
PRIMARIE = 100
CAMPIONAMENTO = {"temperatura": 0.6, "top_p": 0.95, "top_k": 20, "presenza": 1.5}   # protocollo think35 dei log
PROTOCOLLI = {"pensa": {"pensiero": True, "max_nuovi": 32768, "campiona": True},     # tetto come nei log
              "diretto": {"pensiero": False, "max_nuovi": 2048, "campiona": False}}
SEMI = {1: 1000, 2: 2000}
FRAZIONE_MEMORIA = 0.9   # deviazione 2: oltre la memoria della scheda Windows usa la RAM e tutto rallenta di 5-6 volte
TOKEN_FINE = ("<|im_end|>", "<|endoftext|>")
BYTE_PER_VOCE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


# ----------------------------------------------------------------------------------------------------- funzioni pure

def sha256_file(percorso, blocco=1 << 24):
    h = hashlib.sha256()
    with open(percorso, "rb") as f:
        for pezzo in iter(lambda: f.read(blocco), b""):
            h.update(pezzo)
    return h.hexdigest()


def intestazione_safetensors(percorso):
    """(intestazione, inizio dei dati) di un file safetensors, letto senza librerie esterne."""
    with open(percorso, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def confronta_pesi(base, modificato):
    """Confronta due file safetensors tensore per tensore, byte per byte (lettore indipendente da quello dei log).
    Ritorna nomi presenti in uno solo, tensori diversi (con voci diverse e impronte) e il totale delle voci."""
    hb, ob = intestazione_safetensors(base)
    hm, om = intestazione_safetensors(modificato)
    nb = {k for k in hb if k != "__metadata__"}
    nm = {k for k in hm if k != "__metadata__"}
    esito = {"solo_base": sorted(nb - nm), "solo_modificato": sorted(nm - nb), "diversi": {},
             "totale_voci": 0, "voci_modello_di_testo": 0, "tensori_confrontati": 0}
    with open(base, "rb") as fb, open(modificato, "rb") as fm:
        for k in sorted(nb & nm):
            a, b = hb[k]["data_offsets"]
            c, d = hm[k]["data_offsets"]
            fb.seek(ob + a)
            fm.seek(om + c)
            xb, xm = fb.read(b - a), fm.read(d - c)
            voci = (b - a) // BYTE_PER_VOCE[hb[k]["dtype"]]
            esito["totale_voci"] += voci
            if k.startswith("model.language_model.") or k.startswith("lm_head."):
                esito["voci_modello_di_testo"] += voci
            esito["tensori_confrontati"] += 1
            if xb != xm or hb[k]["shape"] != hm[k]["shape"] or hb[k]["dtype"] != hm[k]["dtype"]:
                voce = {"forma_base": hb[k]["shape"], "forma_modificato": hm[k]["shape"], "tipo": hb[k]["dtype"],
                        "sha_base": hashlib.sha256(xb).hexdigest(), "sha_modificato": hashlib.sha256(xm).hexdigest(),
                        "voci_diverse": None}
                if hb[k]["shape"] == hm[k]["shape"] and hb[k]["dtype"] == hm[k]["dtype"] == "BF16":
                    import numpy as np
                    voce["voci_diverse"] = int((np.frombuffer(xb, dtype=np.uint16) != np.frombuffer(xm, dtype=np.uint16)).sum())
                esito["diversi"][k] = voce
    return esito


def leggi_gsm8k(parquet):
    import pyarrow.parquet as pq
    righe = pq.read_table(parquet).to_pylist()
    return [{"domanda": r["question"], "oro": r["answer"].split("####")[-1].strip().replace(",", "")} for r in righe]


def scegli_domande(righe, n, seme):
    """n indici scelti con un seme fisso, in ordine crescente; 'i' è l'indice nel file di prova di GSM8K."""
    indici = sorted(random.Random(seme).sample(range(len(righe)), n))
    return [{"i": i, "domanda": righe[i]["domanda"], "oro": righe[i]["oro"]} for i in indici]


def conta_generati(ids, fine, pad):
    """Quanti token generati contano: fino al primo token di fine compreso; senza fine, tolte le imbottiture in coda."""
    for k, t in enumerate(ids):
        if t in fine:
            return k + 1
    n = len(ids)
    while n > 0 and ids[n - 1] == pad:
        n -= 1
    return n


def estrai_boxed(testo):
    """Contenuto dell'ultimo \\boxed{...} (parentesi bilanciate), o None."""
    inizio = testo.rfind("\\boxed")
    if inizio < 0:
        return None
    k = inizio + len("\\boxed")
    while k < len(testo) and testo[k] == " ":
        k += 1
    if k >= len(testo) or testo[k] != "{":
        m = re.match(r"\s*([^$\n]*)", testo[inizio + len("\\boxed"):])
        return m.group(1).strip() if m and m.group(1).strip() else None
    livello, j = 0, k
    while j < len(testo):
        if testo[j] == "{":
            livello += 1
        elif testo[j] == "}":
            livello -= 1
            if livello == 0:
                return testo[k + 1:j]
        j += 1
    return None


def numero(s):
    """Il numero in una risposta scritta: via \\text{...}, $, %, virgole e spazi; frazioni \\frac{a}{b}; None se non c'è."""
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}", " ", s)
    s = s.replace("\\$", " ").replace("$", " ").replace("\\%", " ").replace("%", " ").replace(",", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("{", " ").replace("}", " ").strip()
    m = re.search(r"\\d?frac\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    s = re.sub(r"\\[a-zA-Z]+", " ", s).strip()
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if m:
        return float(s)
    numeri = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(numeri[-1]) if numeri else None


def ultimo_numero(testo):
    parte = testo.split("</think>")[-1]
    numeri = re.findall(r"-?\d+(?:\.\d+)?", parte.replace(",", ""))
    return float(numeri[-1]) if numeri else None


def giudica(risposta, oro):
    """(stretto, lento): stretto = ultimo \\boxed uguale all'oro; lento = stretto, oppure (senza \\boxed) ultimo numero."""
    oro_n = numero(oro)
    box = estrai_boxed(risposta)
    stretto = box is not None and numero(box) is not None and abs(numero(box) - oro_n) < 1e-6
    if box is not None:
        return stretto, stretto
    u = ultimo_numero(risposta)
    return False, u is not None and abs(u - oro_n) < 1e-6


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    semi = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (round(centro - semi, 4), round(centro + semi, 4))


class Campionatore:
    """Campionamento come vLLM, nell'ordine di vLLM: penalità di presenza (sui soli token generati), temperatura,
    top-k, top-p. Si passa a generate() con temperature=1.0, top_k=0, top_p=1.0, così HF non aggiunge altro."""

    def __init__(self, lunghezza_prompt, temperatura, top_p, top_k, presenza):
        self.lunghezza_prompt = lunghezza_prompt
        self.temperatura, self.top_p, self.top_k, self.presenza = temperatura, top_p, top_k, presenza

    def __call__(self, input_ids, punteggi):
        import torch
        punteggi = punteggi.to(torch.float32)
        generati = input_ids[:, self.lunghezza_prompt:]
        if self.presenza and generati.shape[1] > 0:
            maschera = torch.zeros_like(punteggi, dtype=torch.bool)
            maschera.scatter_(1, generati, True)
            punteggi = punteggi - self.presenza * maschera.to(punteggi.dtype)
        punteggi = punteggi / self.temperatura
        if self.top_k and self.top_k > 0:
            soglia = torch.topk(punteggi, min(self.top_k, punteggi.shape[-1])).values[:, -1:]
            punteggi = punteggi.masked_fill(punteggi < soglia, float("-inf"))
        if self.top_p < 1.0:
            ordinati, indici = torch.sort(punteggi, descending=True)
            prob = torch.softmax(ordinati, dim=-1)
            togli = prob.cumsum(dim=-1) - prob > self.top_p
            ordinati = ordinati.masked_fill(togli, float("-inf"))
            punteggi = torch.full_like(punteggi, float("-inf")).scatter(1, indici, ordinati)
        return punteggi


class Cronometro:
    """Non tocca i punteggi né il generatore casuale: ogni `ogni` passi stampa i passi fatti, i secondi e la memoria
    della GPU (serve alla regola del tempo della premessa, senza aspettare la fine di un lotto)."""

    def __init__(self, ogni=1024):
        self.ogni, self.passi, self.t0 = ogni, 0, time.time()

    def __call__(self, input_ids, punteggi):
        self.passi += 1
        if self.passi % self.ogni == 0:
            import torch
            print(f"    passo {self.passi}: {time.time() - self.t0:.0f} s, GPU riservata "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB, picco usato "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return punteggi


# ------------------------------------------------------------------------------------------------------- lucchetto

def prendi_lucchetto(attivita):
    if os.path.exists(LUCCHETTO):
        raise SystemExit(f"lucchetto GPU già presente: {LUCCHETTO}\n{open(LUCCHETTO, encoding='utf-8').read()}")
    with open(LUCCHETTO, "w", encoding="utf-8") as f:
        f.write(f"agente: Claude (sessione stato-progetti-priorita)\nattivita: {attivita}\n"
                f"ora: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def lascia_lucchetto():
    if os.path.exists(LUCCHETTO):
        os.remove(LUCCHETTO)


# ----------------------------------------------------------------------------------------------------- sottocomandi

def cmd_prepara(a):
    os.makedirs(a.out, exist_ok=True)
    base_file = os.path.join(a.base, PESI_BASE)
    sha = sha256_file(base_file)
    print(f"pesi di base: {PESI_BASE} sha256 {sha[:16]}… {'OK' if sha == SHA_BASE else 'DIVERSA DALL ATTESA'}", flush=True)
    if sha != SHA_BASE:
        raise SystemExit("il file dei pesi di base non è quello della revisione fissata")
    mod_dir = os.path.join(a.out, "modificato")
    os.makedirs(mod_dir, exist_ok=True)
    for nome in os.listdir(a.base):
        if nome != PESI_BASE and not nome.endswith(".log") and os.path.isfile(os.path.join(a.base, nome)):
            shutil.copy2(os.path.join(a.base, nome), os.path.join(mod_dir, nome))
    mod_file = os.path.join(mod_dir, PESI_BASE)
    if not os.path.exists(mod_file):
        print("ricostruzione dei tensori modificati con apply_edit.py dei log (controlla le impronte)…", flush=True)
        subprocess.run([sys.executable, "-I", os.path.join(a.codice, "apply_edit.py"), base_file, a.modifica, mod_file],
                       check=True)
    manifesto = json.load(open(os.path.join(a.modifica, "manifest.json"), encoding="utf-8"))
    attesi = {t["name"]: t for t in manifesto["tensors"]}
    esito = confronta_pesi(base_file, mod_file)
    controlli = {
        "stessi_nomi": not esito["solo_base"] and not esito["solo_modificato"],
        "cambiano_solo_i_tensori_del_manifesto": set(esito["diversi"]) == set(attesi),
        "voci_diverse_come_nel_manifesto": all(n in esito["diversi"] and esito["diversi"][n]["voci_diverse"] == t["changed_entries"]
                                              for n, t in attesi.items()),
        "impronte_come_nel_manifesto": all(n in esito["diversi"] and esito["diversi"][n]["sha_base"] == t["base_sha256"]
                                           and esito["diversi"][n]["sha_modificato"] == t["edited_sha256"] for n, t in attesi.items()),
    }
    voci_cambiate = sum(v["voci_diverse"] or 0 for v in esito["diversi"].values())
    verifica = {"revisione_base": REVISIONE_BASE, "sha256_file_base": sha, "sha256_file_modificato": sha256_file(mod_file),
                "codice": {f: sha256_file(os.path.join(a.codice, f)) for f in ("apply_edit.py", "st_numpy.py")},
                "manifesto": {f: sha256_file(os.path.join(a.modifica, f)) for f in ("manifest.json", "edit.npz")},
                "confronto": esito, "controlli": controlli, "voci_cambiate": voci_cambiate,
                "quota_cambiata_su_tutto_il_file": voci_cambiate / esito["totale_voci"],
                "quota_cambiata_sul_modello_di_testo": voci_cambiate / max(1, esito["voci_modello_di_testo"])}
    json.dump(verifica, open(os.path.join(a.out, "verifica_pesi.json"), "w", encoding="utf-8"), indent=1)
    for k, v in controlli.items():
        print(f"  {k}: {'OK' if v else 'NO'}")
    print(f"  tensori confrontati {esito['tensori_confrontati']}, diversi {len(esito['diversi'])}, voci cambiate {voci_cambiate:,} "
          f"su {esito['totale_voci']:,} ({100 * verifica['quota_cambiata_su_tutto_il_file']:.4f} % del file; "
          f"{100 * verifica['quota_cambiata_sul_modello_di_testo']:.4f} % del modello di testo)")
    if not all(controlli.values()):
        raise SystemExit("la ricostruzione non corrisponde al manifesto")
    dest = os.path.join(a.out, "domande.jsonl")
    if not os.path.exists(dest):
        domande = scegli_domande(leggi_gsm8k(a.gsm8k), a.n, SEME_DOMANDE)
        with open(dest, "w", encoding="utf-8") as f:
            for d in domande:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"domande scritte: {len(domande)} (seme {SEME_DOMANDE})")
    print("prepara: fatto")


def leggi_jsonl(percorso):
    if not os.path.exists(percorso):
        return []
    return [json.loads(r) for r in open(percorso, encoding="utf-8") if r.strip()]


def cmd_genera(a):
    import torch
    from transformers import AutoTokenizer, LogitsProcessorList, Qwen3_5ForConditionalGeneration
    proto = PROTOCOLLI[a.protocollo]
    cartella = a.base if a.modello == "base" else os.path.join(a.out, "modificato")
    dest = os.path.join(a.out, "risposte", f"{a.modello}__{a.protocollo}__c{a.campione}.jsonl")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    domande = leggi_jsonl(os.path.join(a.out, "domande.jsonl"))[:a.primi]
    fatte = {r["i"] for r in leggi_jsonl(dest)}
    da_fare = [d for d in domande if d["i"] not in fatte]
    max_nuovi = a.max_nuovi or proto["max_nuovi"]
    print(f"{a.modello} {a.protocollo} campione {a.campione}: {len(da_fare)} domande da fare, lotto {a.lotto}, "
          f"tetto {max_nuovi} token", flush=True)
    if not da_fare:
        return
    prendi_lucchetto(f"rimisura_astra genera {a.modello} {a.protocollo} c{a.campione}")
    try:
        tok = AutoTokenizer.from_pretrained(a.base)
        tok.padding_side = "left"
        fine = [tok.convert_tokens_to_ids(t) for t in TOKEN_FINE]
        pad = tok.pad_token_id if tok.pad_token_id is not None else fine[-1]
        torch.cuda.set_per_process_memory_fraction(FRAZIONE_MEMORIA)   # sopra il limite l'allocatore libera la sua cache
        modello = Qwen3_5ForConditionalGeneration.from_pretrained(cartella, dtype=torch.bfloat16, device_map="cuda").eval()
        with open(dest, "a", encoding="utf-8") as f:
            for inizio in range(0, len(da_fare), a.lotto):
                lotto = da_fare[inizio:inizio + a.lotto]
                testi = [tok.apply_chat_template([{"role": "user", "content": d["domanda"] + "\n" + ISTRUZIONE}],
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=proto["pensiero"])
                         for d in lotto]
                ingresso = tok(testi, return_tensors="pt", padding=True).to(modello.device)
                lung = ingresso["input_ids"].shape[1]
                torch.manual_seed(SEMI[a.campione] + lotto[0]["i"])
                t0 = time.time()
                with torch.no_grad():
                    if proto["campiona"]:
                        uscita = modello.generate(**ingresso, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                                  logits_processor=LogitsProcessorList([Campionatore(lung, **CAMPIONAMENTO), Cronometro()]),
                                                  max_new_tokens=max_nuovi, eos_token_id=fine, pad_token_id=pad)
                    else:
                        uscita = modello.generate(**ingresso, do_sample=False, max_new_tokens=max_nuovi,
                                                  logits_processor=LogitsProcessorList([Cronometro()]),
                                                  eos_token_id=fine, pad_token_id=pad)
                secondi = round(time.time() - t0, 1)
                for k, d in enumerate(lotto):
                    ids = uscita[k, lung:].tolist()
                    n = conta_generati(ids, fine, pad)
                    tagliata = n >= max_nuovi and (n == 0 or ids[n - 1] not in fine)
                    testo = tok.decode(ids[:n], skip_special_tokens=False)   # tiene <think>/</think> per la misura indulgente
                    for t in TOKEN_FINE:
                        testo = testo.replace(t, "")
                    f.write(json.dumps({"i": d["i"], "campione": a.campione, "risposta": testo,
                                        "n_token": n, "tagliata": tagliata, "secondi_lotto": secondi, "prompt_token": lung,
                                        "tetto": max_nuovi},
                                       ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{inizio + len(lotto)}/{len(da_fare)}] {secondi} s, token medi "
                      f"{sum(conta_generati(uscita[k, lung:].tolist(), fine, pad) for k in range(len(lotto))) // len(lotto)}", flush=True)
    finally:
        lascia_lucchetto()
    print(f"{a.modello} {a.protocollo} c{a.campione}: fatto -> {dest}", flush=True)


def riassunto(righe, oro):
    n = len(righe)
    if n == 0:
        return None
    giudizi = [giudica(r["risposta"], oro[r["i"]]) for r in righe]
    stretti = sum(g[0] for g in giudizi)
    lenti = sum(g[1] for g in giudizi)
    return {"n": n, "giuste_strette": stretti, "acc_stretta": round(stretti / n, 4), "ic95_stretta": wilson(stretti, n),
            "acc_lenta": round(lenti / n, 4), "quota_tagliate": round(sum(bool(r["tagliata"]) for r in righe) / n, 4),
            "token_medi": round(sum(r["n_token"] for r in righe) / n, 1),
            "senza_boxed": sum(estrai_boxed(r["risposta"]) is None for r in righe),
            "tetti": sorted({r["tetto"] for r in righe if "tetto" in r})}


def appaia(b, m, oro, insieme):
    gb = {r["i"]: giudica(r["risposta"], oro[r["i"]])[0] for r in b if r["i"] in insieme}
    gm = {r["i"]: giudica(r["risposta"], oro[r["i"]])[0] for r in m if r["i"] in insieme}
    comuni = sorted(set(gb) & set(gm))
    return {"n": len(comuni), "entrambi_giusti": sum(gb[i] and gm[i] for i in comuni),
            "solo_modificato": sum(gm[i] and not gb[i] for i in comuni),
            "solo_base": sum(gb[i] and not gm[i] for i in comuni), "nessuno": sum(not gb[i] and not gm[i] for i in comuni)}


def analizza(out):
    domande = leggi_jsonl(os.path.join(out, "domande.jsonl"))
    oro = {d["i"]: d["oro"] for d in domande}
    insiemi = {"primarie": {d["i"] for d in domande[:PRIMARIE]}, "tutte": set(oro)}
    esito = {"n_domande": len(domande), "n_primarie": len(insiemi["primarie"]), "corse": {}, "appaiato": {}}
    corse = {}
    for nome in sorted(os.listdir(os.path.join(out, "risposte"))) if os.path.isdir(os.path.join(out, "risposte")) else []:
        if nome.endswith(".jsonl"):
            corse[nome[:-6]] = leggi_jsonl(os.path.join(out, "risposte", nome))
    for chiave, righe in corse.items():
        esito["corse"][chiave] = {nome: riassunto([r for r in righe if r["i"] in ins], oro) for nome, ins in insiemi.items()}
    for proto in PROTOCOLLI:
        for c in SEMI:
            b, m = corse.get(f"base__{proto}__c{c}"), corse.get(f"modificato__{proto}__c{c}")
            if b and m:
                esito["appaiato"][f"{proto}__c{c}"] = {nome: appaia(b, m, oro, ins) for nome, ins in insiemi.items()}
    esito["verdetto"] = verdetto(esito)
    return esito


def verdetto(esito):
    """Regola della premessa, sulle domande primarie, protocollo 'pensa', campione 1, giudizio stretto: REPLICA se il
    modificato supera la base di almeno 30 punti con intervalli di Wilson al 95 % disgiunti e almeno il 50 % di giuste
    (il 40 % se il tetto è sceso a 16.384: deviazione 1 della premessa); NON REPLICA se la differenza è al massimo 10 punti o il modificato resta sotto il 30 %; altrimenti INDIZIO.
    Meccanismo (secondario): la quota di risposte tagliate al tetto cala di almeno 50 punti dalla base al modificato."""
    b = (esito["corse"].get("base__pensa__c1") or {}).get("primarie")
    m = (esito["corse"].get("modificato__pensa__c1") or {}).get("primarie")
    if not b or not m:
        return {"esito": "INCOMPLETO", "motivo": "manca una delle due corse 'pensa' con campione 1"}
    if b["n"] < esito["n_primarie"] or m["n"] < esito["n_primarie"]:
        return {"esito": "INCOMPLETO", "motivo": f"base {b['n']} e modificato {m['n']} risposte su {esito['n_primarie']} primarie"}
    tetti = set(b.get("tetti", [])) | set(m.get("tetti", []))
    if len(tetti) > 1:
        return {"esito": "INCOMPLETO", "motivo": f"tetti diversi nelle corse 'pensa': {sorted(tetti)}"}
    soglia = 0.40 if tetti and max(tetti) <= 16384 else 0.50
    diff = m["acc_stretta"] - b["acc_stretta"]
    disgiunti = m["ic95_stretta"][0] > b["ic95_stretta"][1]
    if diff >= 0.30 and disgiunti and m["acc_stretta"] >= soglia:
        e = "REPLICA"
    elif diff <= 0.10 or m["acc_stretta"] < 0.30:
        e = "NON REPLICA"
    else:
        e = "INDIZIO"
    return {"esito": e, "differenza_punti": round(100 * diff, 1), "intervalli_disgiunti": disgiunti,
            "base": b["acc_stretta"], "modificato": m["acc_stretta"], "n": m["n"],
            "tetto": max(tetti) if tetti else None, "soglia_modificato": soglia,
            "meccanismo_tagliate": b["quota_tagliate"] - m["quota_tagliate"] >= 0.50,
            "tagliate_base": b["quota_tagliate"], "tagliate_modificato": m["quota_tagliate"]}


def cmd_analizza(a):
    esito = analizza(a.out)
    json.dump(esito, open(os.path.join(a.out, "analisi.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"domande: {esito['n_domande']} (primarie {esito['n_primarie']})")
    for k, v in esito["corse"].items():
        for nome, r in v.items():
            if r:
                print(f"  {k} [{nome}]: n {r['n']}, giuste (stretto) {r['acc_stretta']:.3f} IC95 {r['ic95_stretta']}, "
                      f"lento {r['acc_lenta']:.3f}, tagliate {r['quota_tagliate']:.3f}, token medi {r['token_medi']}, "
                      f"senza boxed {r['senza_boxed']}")
    for k, v in esito["appaiato"].items():
        print(f"  appaiato {k}: {v}")
    print(f"verdetto: {esito['verdetto']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("prepara")
    q.add_argument("--base", required=True)
    q.add_argument("--modifica", required=True, help="cartella con manifest.json ed edit.npz dei log")
    q.add_argument("--codice", required=True, help="cartella con apply_edit.py e st_numpy.py dei log")
    q.add_argument("--out", required=True)
    q.add_argument("--gsm8k", default=GSM8K_PREDEFINITO)
    q.add_argument("--n", type=int, default=N_DOMANDE)
    q.set_defaults(f=cmd_prepara)
    g = s.add_parser("genera")
    g.add_argument("--out", required=True)
    g.add_argument("--base", required=True)
    g.add_argument("--modello", choices=["base", "modificato"], required=True)
    g.add_argument("--protocollo", choices=list(PROTOCOLLI), required=True)
    g.add_argument("--campione", type=int, choices=list(SEMI), required=True)
    g.add_argument("--lotto", type=int, default=8)
    g.add_argument("--primi", type=int, default=None, help="solo le prime K domande dell'elenco (le primarie sono le prime 100)")
    g.add_argument("--max-nuovi", type=int, default=None, help="solo per deviazioni dichiarate nella premessa")
    g.set_defaults(f=cmd_genera)
    z = s.add_parser("analizza")
    z.add_argument("--out", required=True)
    z.set_defaults(f=cmd_analizza)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
