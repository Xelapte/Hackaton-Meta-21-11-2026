"""Tableau de bord statique : transforme le journal des scores (SCORES_CSV) en une page HTML.

Pas de serveur, pas d'internet : on régénère la page et on l'ouvre.
    python -m common.dashboard              # écrit results/dashboard.html
    python -m common.dashboard --bas met    # tâches où un score plus bas est meilleur
Astuce : `watch -n 60 python -m common.dashboard` la régénère chaque minute.
"""
import argparse
import csv
import datetime
import html

from common.scores import CSV, meilleurs

PAGE = """<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Scores de l'équipe</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}
th{{background:#f3f2ee}} .best td{{background:#eaf3de;font-weight:600}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:180px}}
.card b{{font-size:22px;display:block}} small{{color:#666}}
</style></head><body>
<h1>Scores de l'équipe</h1><small>Généré le {now}</small>
<h2>Meilleur score par tâche</h2><div class="cards">{cards}</div>
<h2>Tous les runs (du plus récent au plus ancien)</h2>
<table><tr><th>Heure</th><th>Tâche</th><th>Score</th><th>Commit</th>
<th>Configuration</th><th>Notes</th></tr>{rows}</table></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", nargs="*", default=[])
    args = ap.parse_args()
    best = meilleurs(args.bas)
    runs = list(csv.DictReader(CSV.open(newline=""))) if CSV.exists() else []
    best_keys = {(r["heure"], r["tache"], r["score"]) for r in best.values()}
    e = html.escape
    cards = "".join(
        f'<div class="card"><small>{e(t)}</small><b>{e(r["score"])}</b>'
        f'<small>{e(r["heure"])} · {e(r["commit"])}</small></div>'
        for t, r in sorted(best.items())) or "<small>Aucun score pour l'instant.</small>"
    rows = "".join(
        f'<tr class="{"best" if (r["heure"], r["tache"], r["score"]) in best_keys else ""}">'
        f'<td>{e(r["heure"])}</td><td>{e(r["tache"])}</td><td>{e(r["score"])}</td>'
        f'<td>{e(r["commit"])}</td><td>{e(r["config"])}</td><td>{e(r["notes"])}</td></tr>'
        for r in reversed(runs))
    out = CSV.parent / "dashboard.html"
    out.write_text(PAGE.format(now=datetime.datetime.now().strftime("%d/%m %H:%M:%S"),
                               cards=cards, rows=rows), encoding="utf-8")
    print("Écrit :", out)


if __name__ == "__main__":
    main()
