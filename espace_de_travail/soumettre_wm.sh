#!/bin/bash
# Soumet UN fichier de masques au leaderboard WATERMARK avec la fonction submit() de submission_template.py (inchange),
# cle lue dans ~/.hack_api_key (jamais affichee). Autorise par E. le 22/11 a ~9h06 (voir CLAUDE.md).
# Garde-fous : fichier sous unlearning/v6N/subs/*.pt, verification du format (1020 masques 128x128 en 0/1, noms du test
# complets et uniques, < 20 Mo), 10 min au moins depuis la soumission watermark precedente faite par ce script,
# jamais deux fois le meme fichier. Journal : wm_journal.txt.
# Usage : bash soumettre_wm.sh v60/subs/xxx.pt
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning; W=/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task
cd $B || exit 1
[ -f $B/wm_STOP ] && { echo "arrete : wm_STOP present, aucun envoi"; exit 2; }
F="${1:?usage : bash soumettre_wm.sh v6N/subs/fichier.pt}"
case "$F" in v6[0-9]/subs/*.pt) ;; *) echo "refuse : le fichier doit etre de la forme v6N/subs/xxx.pt"; exit 2;; esac
[ -f "$F" ] || { echo "introuvable : $F"; exit 2; }
grep -q -x -F "$F" $B/wm_faits.txt 2>/dev/null && { echo "deja soumis : $F"; exit 2; }
T=$B/.wm_derniere
if [ -f $T ]; then D=$(( $(date +%s) - $(cat $T) )); [ $D -ge 600 ] || { echo "trop tot : derniere soumission watermark il y a ${D}s (attendre $((600 - D))s)"; exit 3; }; fi
cd $W || exit 1
$W/.venv/bin/python - "$B/$F" <<'PY' || { echo "refuse : format invalide"; exit 2; }
import sys, os, json, torch
p = sys.argv[1]; d = torch.load(p, map_location="cpu", weights_only=False); n, m = d["image_names"], torch.as_tensor(d["masks"])
assert os.path.getsize(p) < 20e6, "fichier trop gros"
assert tuple(m.shape) == (1020, 128, 128), m.shape
assert m.dtype in (torch.uint8, torch.bool), m.dtype
assert int(m.max()) <= 1 and int(m.min()) >= 0
assert len(n) == 1020 and len(set(n)) == 1020
ref = sorted(f for f in os.listdir("release/test/images")) if os.path.isdir("release/test/images") else None
if ref is not None: assert sorted(n) == ref or sorted(os.path.splitext(x)[0] for x in n) == sorted(os.path.splitext(x)[0] for x in ref), "noms differents de ceux du test"
print(f"format ok : {tuple(m.shape)} {m.dtype}, part de 1 = {m.float().mean():.3f}, {os.path.getsize(p) / 1e6:.1f} Mo")
PY
date +%s > $T
echo "[$(date '+%F %T')] $F" >> $B/wm_journal.txt
OUT=$(TEAM_API_KEY=$(cat ~/.hack_api_key) $W/.venv/bin/python -c "import sys, submission_template as s; s.submit(sys.argv[1])" "$B/$F" 2>&1)
echo "$OUT" | tee -a $B/wm_journal.txt
echo "$OUT" | grep -q "Successfully submitted" && echo "$F" >> $B/wm_faits.txt
date +%s > $T
