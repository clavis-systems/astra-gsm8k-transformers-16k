#!/usr/bin/env bash
# Coda della rimisura di Astra: aspetta che la corsa C finisca (PID dato) e che il lucchetto GPU sia libero, poi le corse
# nell'ordine della premessa (domande primarie: le prime 100), infine l'analisi. Ripetibile: le domande fatte vengono saltate.
# Uso: bash coda_astra.sh CARTELLA_OUT CARTELLA_BASE [PID_DA_ASPETTARE] [TETTO_PENSA]   (dalla cartella astra_rimisura)
# TETTO_PENSA: solo per la deviazione 1 della premessa (16384); il protocollo diretto resta a 2048.
set -u
OUT="$1"; BASE="$2"; PID="${3:-}"; TETTO="${4:-}"
PY="${PY:-python}"
LOCK="../../risultati/.gpu_lock"
echo "== inizio $(date -Iseconds)"
if [ -n "$PID" ]; then
  while tasklist //FI "PID eq $PID" //NH 2>/dev/null | grep -q " $PID "; do sleep 60; done
  echo "== la corsa C (PID $PID) è finita $(date -Iseconds)"
fi
while [ -e "$LOCK" ]; do sleep 60; done
for spec in "base pensa 1" "modificato pensa 1" "base diretto 1" "modificato diretto 1"; do
  set -- $spec
  EXTRA=""
  if [ "$2" = "pensa" ] && [ -n "$TETTO" ]; then EXTRA="--max-nuovi $TETTO"; fi
  echo "== $(date -Iseconds) $1 $2 c$3"
  "$PY" -I -X utf8 rimisura_astra.py genera --out "$OUT" --base "$BASE" --modello "$1" --protocollo "$2" --campione "$3" --primi 100 $EXTRA \
    || echo "CORSA FALLITA: $spec"
done
echo "== analisi $(date -Iseconds)"
"$PY" -I -X utf8 rimisura_astra.py analizza --out "$OUT"
echo "== fine $(date -Iseconds)"
