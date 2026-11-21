"""Test de fumée Pixel Seal (fiche 6). À déposer dans la file du GPU 2 :
    bash slurm/soumettre.sh 2 smoke python pixelseal/smoke_test.py
Vérifie que les poids sont lus dans ckpts/ (sans téléchargement) et que le message
est relu sur une image non attaquée.
"""
import torch
import videoseal

m = videoseal.load("pixelseal").cuda().eval()
img = torch.rand(1, 3, 512, 512, device="cuda")      # image de test, valeurs dans [0, 1]
msg = m.get_random_msg(1)                           # message de 256 bits
with torch.no_grad():
    out = m.embed(img, msgs=msg)                    # out["imgs_w"] : image tatouée
    bits = (m.detect(out["imgs_w"])["preds"][0, 1:] > 0).float().cpu()
print("bits justes :", (bits == msg[0].float().cpu()).float().mean().item())
