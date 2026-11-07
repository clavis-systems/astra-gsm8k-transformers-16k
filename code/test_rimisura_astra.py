"""Prove delle funzioni pure di rimisura_astra.py (senza GPU, senza modelli)."""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rimisura_astra as R  # noqa: E402


def scrivi_safetensors(percorso, tensori):
    """tensori: {nome: (dtype, bytes, forma)}; scrittore minimo, indipendente da quello dei log."""
    testa, blob, off = {}, [], 0
    for nome, (dt, dati, forma) in tensori.items():
        testa[nome] = {"dtype": dt, "shape": forma, "data_offsets": [off, off + len(dati)]}
        off += len(dati)
        blob.append(dati)
    h = json.dumps(testa).encode()
    with open(percorso, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blob))


class Estrazione(unittest.TestCase):
    def test_boxed_semplice(self):
        self.assertEqual(R.estrai_boxed("so the answer is \\boxed{18}."), "18")

    def test_boxed_ultimo_e_bilanciato(self):
        self.assertEqual(R.estrai_boxed("\\boxed{3} then \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")

    def test_boxed_assente(self):
        self.assertIsNone(R.estrai_boxed("the answer is 18"))

    def test_numero(self):
        self.assertEqual(R.numero("18"), 18.0)
        self.assertEqual(R.numero("\\$1,250"), 1250.0)
        self.assertEqual(R.numero("18 \\text{ dollars}"), 18.0)
        self.assertEqual(R.numero("40\\%"), 40.0)
        self.assertEqual(R.numero("\\frac{1}{2}"), 0.5)
        self.assertEqual(R.numero("\\dfrac{3}{4}"), 0.75)
        self.assertIsNone(R.numero("none"))

    def test_giudica_stretto_e_lento(self):
        self.assertEqual(R.giudica("… \\boxed{18}", "18"), (True, True))
        self.assertEqual(R.giudica("… \\boxed{17}", "18"), (False, False))
        self.assertEqual(R.giudica("<think>5 then 7</think> The answer is 18.", "18"), (False, True))
        self.assertEqual(R.giudica("<think>18</think> The answer is 7.", "18"), (False, False))

    def test_ultimo_numero_dopo_il_pensiero(self):
        self.assertEqual(R.ultimo_numero("<think>1 2 3</think> 4,000 then 5"), 5.0)


class Generazione(unittest.TestCase):
    def test_conta_generati(self):
        self.assertEqual(R.conta_generati([5, 6, 7, 1, 0, 0], fine=[1, 2], pad=0), 4)
        self.assertEqual(R.conta_generati([5, 6, 7], fine=[1, 2], pad=0), 3)
        self.assertEqual(R.conta_generati([5, 6, 0, 0], fine=[1, 2], pad=0), 2)

    def test_campionatore_ordine_vllm(self):
        import torch
        c = R.Campionatore(lunghezza_prompt=2, temperatura=0.5, top_p=0.9, top_k=2, presenza=1.0)
        ids = torch.tensor([[9, 9, 3]])              # prompt di 2 token, generato il token 3
        punteggi = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = c(ids, punteggi)
        # token 3 penalizzato (4-1=3), poi /0.5: [2, 4, 6, 6]; top-k 2 tiene i valori >= 6 -> token 2 e 3
        self.assertTrue(torch.isinf(out[0, 0]) and torch.isinf(out[0, 1]))
        self.assertAlmostEqual(float(out[0, 2]), 6.0)
        self.assertAlmostEqual(float(out[0, 3]), 6.0)

    def test_campionatore_top_p(self):
        import torch
        c = R.Campionatore(lunghezza_prompt=1, temperatura=1.0, top_p=0.5, top_k=0, presenza=0.0)
        out = c(torch.tensor([[0]]), torch.log(torch.tensor([[0.6, 0.3, 0.1]])))
        self.assertFalse(torch.isinf(out[0, 0]))
        self.assertTrue(torch.isinf(out[0, 1]) and torch.isinf(out[0, 2]))

    def test_cronometro_non_tocca_i_punteggi(self):
        import torch
        c = R.Cronometro(ogni=2)
        punteggi = torch.tensor([[1.0, 2.0, 3.0]])
        stato = torch.get_rng_state()
        for _ in range(3):
            self.assertIs(c(torch.tensor([[0]]), punteggi), punteggi)
        self.assertTrue(torch.equal(stato, torch.get_rng_state()))
        self.assertEqual(c.passi, 3)

    def test_scelta_domande_fissa(self):
        righe = [{"domanda": f"d{i}", "oro": str(i)} for i in range(50)]
        a = R.scegli_domande(righe, 10, 7)
        b = R.scegli_domande(righe, 10, 7)
        self.assertEqual(a, b)
        self.assertEqual([d["i"] for d in a], sorted(d["i"] for d in a))
        self.assertEqual(len({d["i"] for d in a}), 10)

    def test_wilson(self):
        basso, alto = R.wilson(50, 100)
        self.assertTrue(0.40 < basso < 0.41 and 0.59 < alto < 0.60)


class Pesi(unittest.TestCase):
    def test_confronta_pesi(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.safetensors")
            b = os.path.join(d, "b.safetensors")
            uguale = struct.pack("<4H", 1, 2, 3, 4)
            scrivi_safetensors(a, {"model.language_model.x": ("BF16", uguale, [2, 2]),
                                   "model.visual.y": ("BF16", struct.pack("<2H", 7, 8), [2])})
            scrivi_safetensors(b, {"model.language_model.x": ("BF16", struct.pack("<4H", 1, 9, 3, 9), [2, 2]),
                                   "model.visual.y": ("BF16", struct.pack("<2H", 7, 8), [2])})
            e = R.confronta_pesi(a, b)
            self.assertEqual(list(e["diversi"]), ["model.language_model.x"])
            self.assertEqual(e["diversi"]["model.language_model.x"]["voci_diverse"], 2)
            self.assertEqual(e["totale_voci"], 6)
            self.assertEqual(e["voci_modello_di_testo"], 4)
            self.assertEqual(e["solo_base"], [])


class Verdetto(unittest.TestCase):
    def esito(self, base, mod, n=100, tagliate=(0.85, 0.07), tetti=([], [])):
        kb, km = round(base * n), round(mod * n)
        return {"n_domande": 200, "n_primarie": n, "corse": {
            "base__pensa__c1": {"primarie": {"n": n, "acc_stretta": kb / n, "ic95_stretta": R.wilson(kb, n), "quota_tagliate": tagliate[0], "tetti": tetti[0]}},
            "modificato__pensa__c1": {"primarie": {"n": n, "acc_stretta": km / n, "ic95_stretta": R.wilson(km, n), "quota_tagliate": tagliate[1], "tetti": tetti[1]}}}}

    def test_replica(self):
        v = R.verdetto(self.esito(0.14, 0.65))
        self.assertEqual(v["esito"], "REPLICA")
        self.assertTrue(v["meccanismo_tagliate"])

    def test_non_replica(self):
        self.assertEqual(R.verdetto(self.esito(0.14, 0.20))["esito"], "NON REPLICA")
        self.assertEqual(R.verdetto(self.esito(0.05, 0.28))["esito"], "NON REPLICA")

    def test_indizio(self):
        v = R.verdetto(self.esito(0.14, 0.34, tagliate=(0.85, 0.60)))
        self.assertEqual(v["esito"], "INDIZIO")
        self.assertFalse(v["meccanismo_tagliate"])

    def test_soglia_della_deviazione_1(self):
        self.assertEqual(R.verdetto(self.esito(0.05, 0.45, tetti=([16384], [16384])))["esito"], "REPLICA")
        self.assertEqual(R.verdetto(self.esito(0.05, 0.45, tetti=([32768], [32768])))["esito"], "INDIZIO")
        self.assertEqual(R.verdetto(self.esito(0.05, 0.45))["soglia_modificato"], 0.50)
        self.assertEqual(R.verdetto(self.esito(0.05, 0.45, tetti=([16384], [32768])))["esito"], "INCOMPLETO")

    def test_incompleto(self):
        e = self.esito(0.14, 0.65)
        e["corse"]["base__pensa__c1"]["primarie"]["n"] = 50
        self.assertEqual(R.verdetto(e)["esito"], "INCOMPLETO")

    def test_analizza_su_cartella(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "risposte"))
            with open(os.path.join(d, "domande.jsonl"), "w", encoding="utf-8") as f:
                for i in range(4):
                    f.write(json.dumps({"i": i, "domanda": "?", "oro": str(i)}) + "\n")
            R.PRIMARIE, vecchio = 2, R.PRIMARIE
            try:
                righe = {"base__pensa__c1": ["\\boxed{9}", "\\boxed{1}", "\\boxed{9}", "\\boxed{3}"],
                         "modificato__pensa__c1": ["\\boxed{0}", "\\boxed{1}", "\\boxed{2}", "\\boxed{9}"]}
                for nome, risposte in righe.items():
                    with open(os.path.join(d, "risposte", nome + ".jsonl"), "w", encoding="utf-8") as f:
                        for i, r in enumerate(risposte):
                            f.write(json.dumps({"i": i, "campione": 1, "risposta": r, "n_token": 5, "tagliata": False}) + "\n")
                e = R.analizza(d)
            finally:
                R.PRIMARIE = vecchio
            self.assertEqual(e["corse"]["base__pensa__c1"]["primarie"]["giuste_strette"], 1)
            self.assertEqual(e["corse"]["modificato__pensa__c1"]["tutte"]["giuste_strette"], 3)
            self.assertEqual(e["appaiato"]["pensa__c1"]["tutte"], {"n": 4, "entrambi_giusti": 1, "solo_modificato": 2, "solo_base": 1, "nessuno": 0})
            self.assertEqual(e["verdetto"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
