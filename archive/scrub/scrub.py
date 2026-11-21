"""SCRUB, rembobinage SCRUB+R et attaque d'appartenance de base (fiche 3, section 4.5)."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def kl_teacher_student(student_logits, teacher_logits):
    """d(x) = KL( p_prof(x) || p_élève(x) ), moyennée sur le batch."""
    return F.kl_div(F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1), reduction="batchmean")


def scrub(teacher, forget_loader, retain_loader, max_steps=2, n_epochs=8,
          alpha=1.0, gamma=1.0, lr=5e-4):
    student = copy.deepcopy(teacher)        # l'élève part d'une copie du prof
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    checkpoints = []
    for epoch in range(n_epochs):
        if epoch < max_steps:               # max-step : s'éloigner du prof sur D_f
            for x, _ in forget_loader:
                with torch.no_grad():
                    t = teacher(x)
                loss = -kl_teacher_student(student(x), t)
                opt.zero_grad(); loss.backward(); opt.step()
        for x, y in retain_loader:          # min-step : rester proche du prof sur D_r
            with torch.no_grad():
                t = teacher(x)
            s = student(x)
            loss = alpha * kl_teacher_student(s, t) + gamma * F.cross_entropy(s, y)
            opt.zero_grad(); loss.backward(); opt.step()
        checkpoints.append(copy.deepcopy(student.state_dict()))   # pour SCRUB+R
    return student, checkpoints


def error(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(-1) != y).float().mean().item()


def rewind(model, checkpoints, X_forget, y_forget, X_val, y_val):
    """SCRUB+R : revenir au checkpoint dont l'erreur d'oubli est la plus proche
    de l'erreur de référence, mesurée sur un jeu de validation de même distribution."""
    model.load_state_dict(checkpoints[-1])
    reference = error(model, X_val, y_val)
    gaps = []
    for ck in checkpoints:
        model.load_state_dict(ck)
        gaps.append(abs(error(model, X_forget, y_forget) - reference))
    model.load_state_dict(checkpoints[int(np.argmin(gaps))])
    return model


def basic_mia(model, X_forget, y_forget, X_test, y_test, seed=0):
    """Attaque d'inférence d'appartenance simple : une régression logistique tente
    de distinguer les exemples oubliés d'exemples jamais vus à partir de leur loss.
    50 % = l'attaquant ne fait pas mieux que le hasard (c'est l'idéal)."""
    with torch.no_grad():
        l_in = F.cross_entropy(model(X_forget), y_forget, reduction="none").numpy()
        l_out = F.cross_entropy(model(X_test), y_test, reduction="none").numpy()
    n = min(len(l_in), len(l_out))
    X = np.clip(np.r_[l_in[:n], l_out[:n]], -400, 400).reshape(-1, 1)
    y = np.r_[np.ones(n), np.zeros(n)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(LogisticRegression(), X, y, cv=cv).mean()
