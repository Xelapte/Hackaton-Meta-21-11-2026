#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Pitch Epi84 (6 minutes, 8 diapositives) : fabrique pitch_Epi84.pptx avec python-pptx.

    /usr/bin/python3 build_deck.py            # écrit pitch_Epi84.pptx à côté du script (~2 s)

Les textes des diapositives (anglais) et les chiffres sont dans le bloc DATA ci-dessous.
Les notes de l'orateur sont la version française de pitch_script.md (marques [Diapo N]) : on retouche le
script, on relance, les notes suivent.
Pour mettre à jour un score : ajouter ou modifier une ligne de "steps", puis relancer. Le grand chiffre de
chaque tâche = le meilleur score de "steps" (sauf si "headline" est rempli à la main).
Le script vérifie que chaque texte tient dans sa boîte (avertissement « DÉBORDE » sinon).
"""

# =====================================================================================================
# ====================================  DATA : TOUT SE MODIFIE ICI  ===================================
# =====================================================================================================
DATA = {
    "meta": {
        "team": "Epi84",
        "subtitle": "Three tasks, one habit: measure honestly",
        "event": "European AI Championship · Paris · 21–22 November 2026",
        "lb_note": "Scores are from the live public leaderboard (public LB). The final ranking uses a hidden split.",
        "closing_title": "What we learned",
        "closing_line": "Throughout: say what is measured, and what is not. Thank you.",
    },

    # ------------------------------------------------------------------------------------------------
    # TÂCHE 1 — Watermark localisation (couleur : terre cuite)
    # ------------------------------------------------------------------------------------------------
    "task1": {
        # >>> SCORES À METTRE À JOUR : [étiquette de la barre, score public LB]
        "steps": [
            ["Sat. evening", 0.5042],
            ["09:27", 0.6538],
            ["09:47", 0.6814],
            ["09:58", 0.6936],
            ["10:17", 0.7252],
            ["10:37", 0.7363],
            ["10:58", 0.7400],
            ["11:23", 0.7549],
            ["11:38", 0.7610],
            ["Final (11:56)", 0.7621],
        ],
        "headline": None,                       # None = meilleur score de "steps" ; sinon un texte, ex. "0.7600"
        "headline_format": "{:.4f}",
        "headline_caption": "mIoU · public LB (final, 11:56)",
        "result_lines": [                       # 3 lignes courtes au maximum
            "Local cross-validation: 0.7506",
            "Public stays within about 0.02 of local",
            "StegaStamp: 0.648 → 0.841 locally",
        ],
        # <<< fin des scores
        "chart_title": "mIoU on the public LB, run after run",
        "chart_format": "0.000",
        "accent": "C2553A", "tint": "F5E3DC",
        "tag": "TASK 1 · WATERMARK LOCALISATION",
        "name": "Watermark localisation",
        "how_title": "We made our own training data",
        "how_problem": "Where is the invisible watermark? One 128×128 mask per image, three methods.",
        "how_strip": ["Images of the two other methods", "Official embedder + random message",
                      "Free ground-truth mask", "Segmentation network"],
        "how_line": "Like holding the official stamp: we know exactly where we put the mark.",
        "result_title": "Longer training kept paying off",
        "lesson": "With the official tool in hand, make your own data.",
    },

    # ------------------------------------------------------------------------------------------------
    # TÂCHE 2 — Model Equality (couleur : sarcelle)
    # ------------------------------------------------------------------------------------------------
    "task2": {
        # >>> SCORES À METTRE À JOUR : [étiquette de la barre, score public LB]
        "steps": [
            ["First rule (Sat. night)", 0.40],
            ["Proof of disguise", 0.70],
            ["+ base of one family", 0.85],
        ],
        "headline": None,
        "headline_format": "{:.2f}",
        "headline_caption": "public LB (final)",
        "result_lines": [
            "4 disguises found in each family, as in practice",
            "ep-29 recited its hidden prompt; our hunch ep-27 did not",
            "Base model found for 1 family of 3: the open limit",
        ],
        # <<< fin des scores
        "chart_title": "Score on the public LB",
        "chart_format": "0.00",
        "accent": "2F7F86", "tint": "DCECEC",
        "tag": "TASK 2 · MODEL EQUALITY",
        "name": "Model equality",
        "how_title": "We looked for proof of a disguise",
        "how_problem": "33 anonymous endpoints: which ones hide the same model?",
        "how_strip": ["Watermark: identical answers", "Temperature: rare words",
                      "System prompt: a fake identity", "No proof: the endpoint stays alone"],
        "how_line": "A disguise changes the text, not the model. Quantisation does the opposite.",
        "result_title": "The proof corrected our hunch",
        "lesson": "Ask who carries proof, not who looks alike.",
    },

    # ------------------------------------------------------------------------------------------------
    # TÂCHE 3 — Machine unlearning (couleur : indigo)
    # ------------------------------------------------------------------------------------------------
    "task3": {
        # >>> SCORES À METTRE À JOUR : [étiquette de la barre, score public LB] (records successifs)
        "steps": [
            ["First attempt", 0.0143],
            ["Forget more", 0.0453],
            ["Switch off 9 classes", 0.135],
            ["Average 16 views", 0.1506],
            ["Only 2 classes off", 0.2455],
            ["Fragile images, not classes", 0.2859],
            ["Two fragility indicators", 0.2937],
            ["09:31", 0.3144],
            ["Best (11:42)", 0.3236],
        ],
        "headline": None,
        "headline_format": "{:.4f}",
        "headline_caption": "public LB, best (11:42)",
        "result_lines": [
            "16th on this task at 11:56 (leader ≈ 0.48 at 8:00)",
            "Late gain: one margin setting, +0.009",
            "~75 submissions on the public split: overfitting risk",
        ],
        # <<< fin des scores
        "chart_title": "forget × retain on the public LB: our records",
        "chart_format": "0.000",
        "accent": "5F5AA2", "tint": "E6E5F3",
        "tag": "TASK 3 · MACHINE UNLEARNING",
        "name": "Machine unlearning",
        "how_title": "Forgetting is not failing everything",
        "how_problem": "Forget 3 500 images, as if retrained without them. No retain data, no labels.",
        "how_strip": ["Original model (teacher)", "Target: averaged views, true class lowered",
                      "Student imitates the target", "Push only fragile images to high loss"],
        "how_line": "Like a student who memorised the answer key: hesitate where others would.",
        "result_title": "A long climb, in small steps",
        "lesson": "Know what the metric rewards before optimising.",
    },
}
# =====================================================================================================
# ==========================================  FIN DE DATA  ============================================
# =====================================================================================================

import argparse
import io
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "pylib"))      # python-pptx installé dans ce dossier seulement
try:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
    from pptx.util import Inches, Pt
except ImportError as e:                             # pylib/ contient des roues pour Python 3.9
    sys.exit("python-pptx introuvable (%s). Lance : /usr/bin/python3 %s" % (e, os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------------- style
LARGEUR, HAUTEUR = 13.333, 7.5           # 16:9, en pouces
MX = 0.8                                 # marges généreuses
CW = LARGEUR - 2 * MX
POLICE = "Calibri"                       # texte et chiffres
POLICE_TITRE = "Georgia"                 # titres (pas de chiffres dedans : Georgia a des chiffres « bas de casse »)
FOND = "FAF8F4"                          # blanc cassé
TEXTE = "2F3437"                         # gris foncé, pas de noir pur
SOURD = "6E7479"
BLANC = "FFFFFF"
TOTAL = 8
SCRIPT = os.path.join(ICI, "pitch_script.md")


def rgb(h):
    return RGBColor.from_string(h)


# -------------------------------------------------------------------- mesure du texte (anti-débordement)
_POLICES = {}
ALERTES = []
NB_BOITES = [0]


def _police_mesure(gras):
    if gras not in _POLICES:
        f = None
        try:
            from PIL import ImageFont
            nom = "DejaVuSans-Bold.ttf" if gras else "DejaVuSans.ttf"
            for d in ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
                p = os.path.join(d, nom)
                if os.path.exists(p):
                    f = ImageFont.truetype(p, 100)
                    break
        except Exception:
            f = None
        _POLICES[gras] = f
    return _POLICES[gras]


def largeur_texte(txt, pt, gras=False):
    """Largeur estimée en pouces, calée sur Arial + 5 % (Calibri est plus étroit : marge confortable)."""
    f = _police_mesure(gras)
    if f is None:
        return len(txt) * pt * (0.60 if gras else 0.54) / 72.0
    return f.getlength(txt) * pt / 100.0 / 72.0 * 0.93


def nb_lignes(txt, pt, largeur, gras=False):
    n = 0
    for bloc in txt.split("\n"):
        ligne, n_bloc = "", 1
        for mot in bloc.split(" "):
            essai = (ligne + " " + mot).strip()
            if ligne and largeur_texte(essai, pt, gras) > largeur:
                n_bloc += 1
                ligne = mot
            else:
                ligne = essai
        n += n_bloc
    return n


def verifier(no, txt, besoin, dispo, quoi="hauteur"):
    NB_BOITES[0] += 1
    if besoin > dispo + 0.03:
        ALERTES.append("  diapo %d : DÉBORDE en %s (%.2f po pour %.2f po) : « %s »" % (no, quoi, besoin, dispo, txt[:60]))


# ------------------------------------------------------------------------------------------- briques
def _cadre(tf, anchor, marge=0.0):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(marge)
    tf.vertical_anchor = anchor


def _ecrire(tf, no, lignes, w, h, size, bold, color, italic, align, after, police):
    besoin = 0.0
    for i, ligne in enumerate(lignes):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after if i < len(lignes) - 1 else 0)
        r = p.add_run()
        r.text = ligne
        r.font.name = police
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = rgb(color)
        besoin += nb_lignes(ligne, size, w, bold) * size * 1.2 / 72.0 + (after / 72.0 if i < len(lignes) - 1 else 0.0)
    verifier(no, lignes[0], besoin, h)


def texte(slide, no, x, y, w, h, lignes, size=20, bold=False, color=TEXTE, italic=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, after=10, police=POLICE):
    """Zone de texte sans marge interne. lignes : une chaîne, ou une liste (un paragraphe par élément)."""
    if isinstance(lignes, str):
        lignes = [lignes]
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _cadre(tb.text_frame, anchor)
    _ecrire(tb.text_frame, no, lignes, w, h, size, bold, color, italic, align, after, police)
    return tb


def boite(slide, no, x, y, w, h, txt, fill, color=TEXTE, size=18, bold=False):
    """Boîte arrondie avec son texte centré (le texte est DANS la forme : facile à retoucher dans PowerPoint)."""
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.18
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    _cadre(s.text_frame, MSO_ANCHOR.MIDDLE, marge=0.15)
    _ecrire(s.text_frame, no, [txt], w - 0.3, h - 0.3, size, bold, color, False, PP_ALIGN.CENTER, 0, POLICE)
    return s


def fleche(slide, x, y, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def nouvelle(prs, no, titre, notes, tache=None, police=POLICE_TITRE, taille=32, y=1.0, h=0.85):
    slide = prs.slides.add_slide(prs.slide_layouts[5])          # « Title Only » : vrai titre de diapositive
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(FOND)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(MX), Inches(y), Inches(CW), Inches(h)
    tf = t.text_frame
    tf.text = titre
    _cadre(tf, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.runs[0]
    r.font.name = police
    r.font.size = Pt(taille)
    r.font.bold = True
    r.font.color.rgb = rgb(TEXTE)
    verifier(no, titre, nb_lignes(titre, taille, CW, True) * taille * 1.2 / 72.0, h)
    if tache:
        texte(slide, no, MX, 0.62, CW, 0.3, DATA[tache]["tag"], size=14, bold=True, color=DATA[tache]["accent"])
    if no > 1:
        texte(slide, no, MX, 6.98, 4.0, 0.25, DATA["meta"]["team"], size=12, color=SOURD)
        texte(slide, no, LARGEUR - MX - 1.5, 6.98, 1.5, 0.25, "%d / %d" % (no, TOTAL), size=12, color=SOURD,
              align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def meilleur(tache):
    t = DATA[tache]
    if t.get("headline"):
        return str(t["headline"])
    return t["headline_format"].format(max(s[1] for s in t["steps"]))


def bande(slide, no, y, h, etapes, accent, tint):
    """« Comment ça marche » : boîtes arrondies de gauche à droite, reliées par des flèches."""
    n = len(etapes)
    trou = 0.64
    w = (CW - (n - 1) * trou) / n
    for i, txt in enumerate(etapes):
        x = MX + i * (w + trou)
        dernier = i == n - 1
        boite(slide, no, x, y, w, h, txt, accent if dernier else tint, color=BLANC if dernier else TEXTE, bold=dernier)
        if not dernier:
            fleche(slide, x + w + 0.12, y + h / 2 - 0.18, trou - 0.24, 0.36, accent)


def graphique(slide, no, x, y, w, tache):
    """Barres horizontales natives (modifiables dans PowerPoint) : une seule série, le score public."""
    t = DATA[tache]
    n = len(t["steps"])
    h = min(4.15, max(3.1, 0.62 * n + 0.6))
    texte(slide, no, x, y, w, 0.36, t["chart_title"], size=18, bold=True)
    cd = CategoryChartData(number_format=t["chart_format"])
    cd.categories = [s[0] for s in t["steps"]]
    cd.add_series("Public LB", [s[1] for s in t["steps"]])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y + 0.5), Inches(w), Inches(h), cd)
    gf.name = "Graphique " + tache
    ch = gf.chart
    ch.font.name = POLICE
    ch.font.size = Pt(16)
    ch.font.color.rgb = rgb(TEXTE)
    ch.has_title = False
    ch.has_legend = False
    plot = ch.plots[0]
    plot.vary_by_categories = False
    plot.gap_width = 55
    cat = ch.category_axis
    cat.reverse_order = True                     # première étape en haut
    cat.has_major_gridlines = False
    cat.major_tick_mark = XL_TICK_MARK.NONE
    cat.format.line.color.rgb = rgb("C9C5BD")
    cat.tick_labels.font.size = Pt(16)
    cat.tick_labels.font.name = POLICE
    cat.tick_labels.font.color.rgb = rgb(TEXTE)
    val = ch.value_axis
    val.has_major_gridlines = False
    val.visible = False
    val.minimum_scale = 0
    val.maximum_scale = round(max(s[1] for s in t["steps"]) * 1.22, 3)   # place pour les étiquettes de valeur
    s = plot.series[0]
    s.format.fill.solid()
    s.format.fill.fore_color.rgb = rgb(t["accent"])
    s.invert_if_negative = False
    dl = s.data_labels
    dl.show_value = True
    dl.number_format = t["chart_format"]
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(16)
    dl.font.name = POLICE
    dl.font.bold = True
    dl.font.color.rgb = rgb(TEXTE)
    return gf


# ------------------------------------------------------------------------------------ les 8 diapositives
def d_titre(prs, notes):
    m = DATA["meta"]
    s = nouvelle(prs, 1, m["team"], notes.get(1, ""), police=POLICE, taille=66, y=0.95, h=1.2)
    texte(s, 1, MX, 2.2, CW, 0.5, m["subtitle"], size=26, color=TEXTE, police=POLICE_TITRE)
    texte(s, 1, MX, 2.85, CW, 0.36, m["event"], size=18, color=SOURD)
    w = (CW - 2 * 0.5) / 3
    for i, tache in enumerate(("task1", "task2", "task3")):
        t = DATA[tache]
        x = MX + i * (w + 0.5)
        texte(s, 1, x, 3.95, w, 0.36, t["name"], size=18, bold=True, color=t["accent"])
        texte(s, 1, x, 4.3, w, 0.95, meilleur(tache), size=54, bold=True, color=t["accent"])
        texte(s, 1, x, 5.3, w, 0.36, t["headline_caption"], size=18, color=SOURD)
    texte(s, 1, MX, 6.25, CW, 0.36, m["lb_note"], size=18, color=SOURD)


def d_comment(prs, no, tache, notes):
    t = DATA[tache]
    s = nouvelle(prs, no, t["how_title"], notes.get(no, ""), tache=tache)
    texte(s, no, MX, 2.0, CW, 0.75, t["how_problem"], size=20, color=SOURD)
    bande(s, no, 3.0, 1.9, t["how_strip"], t["accent"], t["tint"])
    texte(s, no, MX, 5.45, CW, 0.9, t["how_line"], size=24, italic=True, color=TEXTE, police=POLICE_TITRE)


def d_resultat(prs, no, tache, notes):
    t = DATA[tache]
    s = nouvelle(prs, no, t["result_title"], notes.get(no, ""), tache=tache)
    texte(s, no, MX, 2.0, 4.7, 1.3, meilleur(tache), size=76, bold=True, color=t["accent"])
    texte(s, no, MX, 3.35, 4.7, 0.36, t["headline_caption"], size=18, bold=True, color=TEXTE)
    texte(s, no, MX, 4.1, 4.7, 2.6, t["result_lines"][:3], size=20, color=TEXTE, after=12)
    graphique(s, no, 6.1, 2.1, LARGEUR - MX - 6.1, tache)


def d_conclusion(prs, notes):
    m = DATA["meta"]
    s = nouvelle(prs, 8, m["closing_title"], notes.get(8, ""), y=0.8)
    for i, tache in enumerate(("task1", "task2", "task3")):
        t = DATA[tache]
        y = 2.05 + i * 1.3
        texte(s, 8, MX, y, 2.7, 0.95, meilleur(tache), size=44, bold=True, color=t["accent"], anchor=MSO_ANCHOR.MIDDLE)
        texte(s, 8, MX + 3.0, y + 0.02, CW - 3.0, 0.3, t["name"].upper(), size=14, bold=True, color=t["accent"])
        texte(s, 8, MX + 3.0, y + 0.36, CW - 3.0, 0.5, t["lesson"], size=22, color=TEXTE)
    texte(s, 8, MX, 6.05, CW, 0.45, m["closing_line"], size=22, italic=True, color=SOURD, police=POLICE_TITRE)


def notes_du_script():
    """Version française de pitch_script.md, découpée aux marques [Diapo N] (les titres « # » sont ignorés)."""
    notes = {}
    try:
        fr = io.open(SCRIPT, encoding="utf-8").read().split("## English version")[0]
    except IOError:
        return {}
    courant = None
    for ligne in fr.splitlines():
        l = ligne.strip()
        m = re.match(r"^\[Diapo (\d+)\]$", l)
        if m:
            courant = int(m.group(1))
            notes.setdefault(courant, [])
        elif courant is not None and l and not l.startswith("#") and not l.startswith("---"):
            notes[courant].append(l)
    return {k: "\n\n".join(v) for k, v in notes.items()}


def main():
    ap = argparse.ArgumentParser(description="Fabrique le pitch Epi84 (8 diapositives, 16:9).")
    ap.add_argument("--out", default=os.path.join(ICI, "pitch_Epi84.pptx"))
    ap.add_argument("--debit", type=float, default=150.0, help="mots par minute, pour estimer la durée du script")
    args = ap.parse_args()
    notes = notes_du_script()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(LARGEUR), Inches(HAUTEUR)
    prs.core_properties.title = "Epi84 - European AI Championship 2026"
    prs.core_properties.author = "Epi84"
    d_titre(prs, notes)
    for no, tache in ((2, "task1"), (4, "task2"), (6, "task3")):
        d_comment(prs, no, tache, notes)
        d_resultat(prs, no + 1, tache, notes)
    d_conclusion(prs, notes)
    assert len(prs.slides) == TOTAL <= 8, "8 diapositives au maximum"
    prs.save(args.out)

    # relecture du fichier écrit : nombre de diapositives, titres, durée du script lu dans les notes
    relu = Presentation(args.out)
    print("Écrit : %s (%d diapositives, %.0f Ko)" % (args.out, len(relu.slides), os.path.getsize(args.out) / 1024.0))
    total = 0
    for i, sl in enumerate(relu.slides, 1):
        n = sum(1 for m in sl.notes_slide.notes_text_frame.text.split() if any(c.isalnum() for c in m))
        total += n
        print("  %d. %-42s notes : %3d mots, ~%3.0f s%s" % (i, sl.shapes.title.text, n, n * 60.0 / args.debit,
                                                           "" if n else "  <-- pas de [Diapo %d] dans pitch_script.md" % i))
    print("Script (notes) : %d mots, ~%d min %02d s à %.0f mots/min (limite : 6 min)"
          % (total, int(total * 60 / args.debit) // 60, int(total * 60 / args.debit) % 60, args.debit))
    print("Scores affichés (meilleur public LB) : tâche 1 = %s, tâche 2 = %s, tâche 3 = %s"
          % (meilleur("task1"), meilleur("task2"), meilleur("task3")))
    if ALERTES:
        print("ATTENTION, texte trop long (raccourcir dans DATA) :")
        print("\n".join(ALERTES))
    else:
        print("Mise en page : %d zones de texte vérifiées, aucun débordement estimé." % NB_BOITES[0])


if __name__ == "__main__":
    main()
