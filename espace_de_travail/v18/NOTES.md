# v18 — Ancrage plus faible, entraînement plus court (autour du record)

## Pourquoi
- v16/e2 (ancrage fort w_ret 6 + 8 époques) = 0.0959, très en dessous du record (0.1506) → tester l'inverse proprement.
- Record = v15/f20_k16_d6_b7 : 16 vues, δ_part 6, w_ret 3, 5 époques, aug_train 0.5.

## Runs (un élément à la fois, puis β=7)
| run | changement | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| k16_wr1_b7 | ancrage w_ret 1 | 0.340 | 0.928 | 0.76 | **0.1528** (22h50, record de peu ; soumis par Claude) |
| k16_ep3_b7 | 3 époques | 0.600 | 0.944 | 0.81 | |
| k16_wr1_at025_b7 | w_ret 1 + copies légères (aug_train 0.25) | 0.321 | 0.924 | 0.75 | **0.1552** (23h21, soumis par Claude) |
