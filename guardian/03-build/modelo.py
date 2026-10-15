"""Regresion logistica hibrida (PASO 6). Requiere sklearn + numpy (RNF-01).

Matriz X SOLO con features de grupos CONDICIONADOS + score baseline:

- cifras (aislado en `cifras.feats_num`, CONDICIONADO): f_num, f_year.
  f_deriv se EXCLUYE (las fieles derivan mas; ver resultados_paso2.md:
  su signo lo aprenderia la logistica, pero con ~600 H y p pequeno se
  deja fuera hasta que la ablacion pida lo contrario).
- marcadores (`marcadores.feats_marc`, pendiente de gate): f_hedge,
  f_caus. f_alc y flip se EXCLUYEN (no condicionados para este paso).
- alineacion (`alineacion.feats_alin`, pendiente de gate): cobertura,
  dispersion, margen_min, f_dos_peores.
- baseline: `reglas.puntuar_caso` como feature (ancla la ordenacion NLI+juez).

FUNDAMENTO B.7: p = sigmoide(w.x + b), L2 (C = 1/lambda en sklearn),
estandarizar con mu/sigma del entreno y EMBEBERLOS (se devuelven en el
dict de parametros para `mi_bot.py`). C se elige por CV simple DENTRO
del train (GroupKFold interno por hash de fuentes), sin tocar H de
validacion. NaN/inf -> 0; todo acceso con .get()/try-except.

Coste 4:1 (B.2): UNA sola via (B.2 avisa de no componerlas). ELECCION:
entrenar SIN peso de clase y reescalar despues con
s = 4p / (1 + 3p) (p = 0,2 -> s = 0,5). Motivo: el peso 4 en el entreno
mueve el mismo logit en +ln 4, pero contamina la CV de C (el AUROC no
cambia, mas el corte interno si) y deja el desplazamiento escondido en
los coeficientes; el reescalado posterior es monotono (no toca el
AUROC), lleva el corte Bayes p > 0,2 al corte operativo 0,5 y queda
auditable en una linea (`reescalar`).

Politica H: igual que `validacion.evaluar_folds` por defecto
(`excluir_h_entreno=True`): al medir sobre H, ningun H entrena.
"""

import math
import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    LogisticRegression = None
    GroupKFold = None
    StandardScaler = None

try:
    from alineacion import feats_alin
    from cifras import feats_num
    from marcadores import feats_marc
    from reglas import FALLBACK, puntuar_caso
    from validacion import (
        comparar_scores,
        es_h,
        folds_por_grupo,
        grupos_de,
        verificar_disjuncion,
    )
except ImportError:  # carga directa por ruta (p. ej. desde otro cwd)
    import importlib.util

    _AQUI = os.path.dirname(os.path.abspath(__file__))

    def _cargar(nombre):
        _spec = importlib.util.spec_from_file_location(
            nombre, os.path.join(_AQUI, nombre + ".py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod

    _cifras = _cargar("cifras")
    _marc = _cargar("marcadores")
    _alin = _cargar("alineacion")
    _reglas = _cargar("reglas")
    _val = _cargar("validacion")
    feats_num = _cifras.feats_num
    feats_marc = _marc.feats_marc
    feats_alin = _alin.feats_alin
    FALLBACK = _reglas.FALLBACK
    puntuar_caso = _reglas.puntuar_caso
    comparar_scores = _val.comparar_scores
    es_h = _val.es_h
    folds_por_grupo = _val.folds_por_grupo
    grupos_de = _val.grupos_de
    verificar_disjuncion = _val.verificar_disjuncion

try:
    from evaluar_local import auroc
except ImportError:  # pragma: no cover
    _AQUI2 = os.path.dirname(os.path.abspath(__file__))
    import importlib.util as _ilu

    _spec2 = _ilu.spec_from_file_location(
        "evaluar_local", os.path.join(_AQUI2, "evaluar_local.py"))
    _ev = _ilu.module_from_spec(_spec2)
    _spec2.loader.exec_module(_ev)
    auroc = _ev.auroc

# Columnas de X, en orden fijo (9 = 2 cifras + 2 marcadores + 4 alineacion + 1 baseline).
COLUMNAS = ("f_num", "f_year", "f_hedge", "f_caus", "cobertura",
            "dispersion", "margen_min", "f_dos_peores", "score_base")
# Grupo -> indices en COLUMNAS (la ablacion quita un grupo cada vez).
GRUPOS = {"cifras": (0, 1), "marcadores": (2, 3), "alineacion": (4, 5, 6, 7)}
# Variantes de ablacion: completa + sin cada grupo.
VARIANTES = ("completo", "sin_cifras", "sin_marcadores", "sin_alineacion")
# Parrilla de C para la CV interna (lambda = 1/C, B.7).
CANDIDATAS_C = (0.01, 0.1, 1.0, 10.0)
N_SPLITS_INTERNA = 3
MAX_ITER = 5000
SEMILLA = 42

# Configuracion FINAL (PASO 7, reentreno limpio): baseline + alineacion.
# Motivo (ablacion paso 6, OOF en H, mismo GroupKFold; gate = IC95 > 0
# en AMBAS, AUROC y perdida):
# - cifras:      dAUC -0,000 (-0,004,+0,003); dPerd -0,006 (-0,023,+0,009)
#                -> FUERA (Delta ~= 0, no aporta nada).
# - marcadores:  dAUC -0,008 (-0,014,-0,002); dPerd -0,030 (-0,063,-0,003)
#                -> FUERA (resta: quitarlo MEJORA ambas).
# - alineacion:  dAUC +0,023 (+0,009,+0,038); dPerd +0,078 (+0,025,+0,137)
#                -> ENTRA (unico que cumple el gate en ambas).
# COLUMNAS/VARIANTES/oof_ablacion se conservan intactos como auditoria
# del paso 6; lo que se reentrena y mide en el paso 7 es COLUMNAS_FINAL.
COLUMNAS_FINAL = ("cobertura", "dispersion", "margen_min",
                  "f_dos_peores", "score_base")
# Posiciones de COLUMNAS_FINAL dentro de COLUMNAS (verificacion en
# 04-test/test_modelo.py: [COLUMNAS[i] for i in IDX_FINAL]).
IDX_FINAL = (4, 5, 6, 7, 8)
# Platt (B.8/B.16): sigmoide(a*z + b) con z = logit del score.
# C grande ~= maxima verosimilitud sin penalizacion (1-D, ~600 puntos H).
C_PLATT = 1e6
EPS_LOGIT = 1e-4


def _finito(x):
    """Float finito o 0.0 (NaN/inf/ilegible -> 0, B.7). Nunca lanza."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    try:
        if not math.isfinite(v):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return v


def reescalar(p):
    """Reescalado B.2: s = 4p/(1+3p); s > 0,5 <=> p > 0,2. Nunca lanza."""
    try:
        q = min(1.0, max(0.0, float(p)))
    except (TypeError, ValueError):
        return 0.5
    try:
        s = 4.0 * q / (1.0 + 3.0 * q)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.5
    if not math.isfinite(s):
        return 0.5
    return min(1.0, max(0.0, s))


def extraer_vector(caso):
    """Vector de 9 features de un caso (orden COLUMNAS). Nunca lanza.

    Solo lee pregunta/fuentes/respuesta/senales (sin etiqueta ni tipo).
    f_deriv (cifras), f_alc y flip (marcadores) quedan FUERA a proposito
    (ver docstring del modulo). Caso ilegible -> vector de ceros con
    score_base = FALLBACK.
    """
    try:
        if not isinstance(caso, dict):
            return [0.0] * 8 + [FALLBACK]
        try:
            respuesta = caso.get("respuesta", "") or ""
        except (TypeError, ValueError, AttributeError):
            respuesta = ""
        try:
            fuentes = caso.get("fuentes", []) or []
        except (TypeError, ValueError, AttributeError):
            fuentes = []
        try:
            f_num, _, f_year = feats_num(respuesta, fuentes)
        except (TypeError, ValueError, AttributeError):
            f_num, f_year = 0.0, 0.0
        try:
            f_caus, _, f_hedge, _ = feats_marc(caso)
        except (TypeError, ValueError, AttributeError):
            f_caus, f_hedge = 0, 0
        try:
            cobertura, dispersion, margen_min, f_dos = feats_alin(caso)
        except (TypeError, ValueError, AttributeError):
            cobertura, dispersion, margen_min, f_dos = 0.0, 0, 0.0, 0.0
        try:
            base = min(1.0, max(0.0, float(puntuar_caso(caso))))
            if not math.isfinite(base):
                base = FALLBACK
        except (TypeError, ValueError, AttributeError):
            base = FALLBACK
        return [_finito(f_num), _finito(f_year), _finito(f_hedge),
                _finito(f_caus), _finito(cobertura), _finito(dispersion),
                _finito(margen_min), _finito(f_dos), base]
    except (TypeError, ValueError, AttributeError):
        return [0.0] * 8 + [FALLBACK]


def construir_X(casos):
    """(X, y) de una lista de casos: X lista de vectores, y etiquetas 0/1."""
    X = [extraer_vector(c) for c in (casos or [])]
    y = []
    for c in (casos or []):
        try:
            y.append(int(c.get("etiqueta", 0)) if isinstance(c, dict) else 0)
        except (TypeError, ValueError, AttributeError):
            y.append(0)
    return X, y


def columnas_de(variante):
    """Indices de COLUMNAS que usa cada variante de ablacion."""
    if variante == "completo":
        return list(range(len(COLUMNAS)))
    if variante == "sin_cifras":
        fuera = set(GRUPOS["cifras"])
    elif variante == "sin_marcadores":
        fuera = set(GRUPOS["marcadores"])
    elif variante == "sin_alineacion":
        fuera = set(GRUPOS["alineacion"])
    else:
        raise ValueError("variante desconocida: %r" % (variante,))
    return [i for i in range(len(COLUMNAS)) if i not in fuera]


def elegir_C(X_tr, y_tr, grupos_tr, candidatas=CANDIDATAS_C):
    """Mejor C por CV interna GroupKFold (AUROC medio). No toca validacion.

    X_tr/y_tr/grupos_tr son SOLO entreno del fold externo. Devuelve la
    primera C con mejor AUROC medio (empate -> la menor, determinista).
    """
    if LogisticRegression is None or GroupKFold is None:
        raise RuntimeError("sklearn no disponible")
    Xt = np.asarray(X_tr, dtype=float)
    yt = np.asarray(y_tr, dtype=int)
    gt = list(grupos_tr)
    mejor_c, mejor_media = None, None
    for C in candidatas:
        medias = []
        try:
            pliegues = GroupKFold(n_splits=N_SPLITS_INTERNA).split(Xt, groups=gt)
        except (TypeError, ValueError):
            continue
        for itr, iva in pliegues:
            try:
                sc = StandardScaler().fit(Xt[itr])
                mod = LogisticRegression(C=float(C), max_iter=MAX_ITER)
                mod.fit(sc.transform(Xt[itr]), yt[itr])
                prob = [float(v) for v in mod.predict_proba(sc.transform(Xt[iva]))[:, 1]]
                medias.append(auroc([int(v) for v in yt[iva]], prob))
            except (TypeError, ValueError):
                continue
        if not medias:
            continue
        media = sum(medias) / len(medias)
        if mejor_media is None or media > mejor_media:
            mejor_media, mejor_c = media, float(C)
    if mejor_c is None:
        return float(candidatas[len(candidatas) // 2])
    return mejor_c


def ajustar(X, y, C):
    """Ajusta scaler + logistica y devuelve params EMBEBIBLES (B.7).

    Dict: {columnas, mu, sigma, coef, intercepto, C}. sigma es el scale_
    de StandardScaler (std con 0 -> 1.0, igual que sklearn): el score con
    estos params reproduce exacto a scaler+modelo.
    """
    if StandardScaler is None or LogisticRegression is None:
        raise RuntimeError("sklearn no disponible")
    Xt = np.asarray(X, dtype=float)
    yt = np.asarray(y, dtype=int)
    sc = StandardScaler().fit(Xt)
    mod = LogisticRegression(C=float(C), max_iter=MAX_ITER)
    mod.fit(sc.transform(Xt), yt)
    return {"columnas": list(COLUMNAS),
            "mu": [float(v) for v in sc.mean_],
            "sigma": [float(v) for v in sc.scale_],
            "coef": [float(v) for v in mod.coef_[0]],
            "intercepto": float(mod.intercept_[0]),
            "C": float(C)}


def puntuar_vector(params, x):
    """Score reescalado de un vector con params embebidos. Nunca lanza."""
    try:
        mu = params.get("mu", [])
        sigma = params.get("sigma", [])
        coef = params.get("coef", [])
        b = float(params.get("intercepto", 0.0))
        z = float(b)
        for i, v in enumerate(x or []):
            try:
                m = float(mu[i])
                s = float(sigma[i])
                w = float(coef[i])
            except (TypeError, ValueError, IndexError):
                continue
            try:
                if not math.isfinite(m):
                    m = 0.0
                if not math.isfinite(s) or s == 0.0:
                    s = 1.0
                if not math.isfinite(w):
                    w = 0.0
                v = _finito(v)
                z += w * ((v - m) / s)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        if not math.isfinite(z):
            return FALLBACK
        p = 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))
        return reescalar(p)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return FALLBACK


def puntuar_caso_logistica(params, caso):
    """Score [0,1] de un caso con params embebidos. Nunca lanza."""
    try:
        return puntuar_vector(params, extraer_vector(caso))
    except (TypeError, ValueError, AttributeError):
        return FALLBACK


def oof_ablacion(casos, n_splits=5):
    """Scores fuera-de-pliegue por variante, con el MISMO GroupKFold de validacion.

    Reusa `folds_por_grupo` + `verificar_disjuncion` (hash de fuentes):
    cada caso de validacion se mide una vez. El entreno de cada fold
    excluye H (misma politica que `evaluar_folds`); C se elige con CV
    interna solo sobre ese entreno. Devuelve dict con y, grupos,
    baseline (reglas) y scores por variante, alineados con `casos`.
    """
    folds = folds_por_grupo(casos, n_splits=n_splits)
    verificar_disjuncion(casos, folds)
    X, y = construir_X(casos)
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=int)
    grupos = grupos_de(casos)
    base = []
    for c in casos:
        try:
            s = min(1.0, max(0.0, float(puntuar_caso(c))))
        except (TypeError, ValueError, AttributeError):
            s = FALLBACK
        base.append(s)
    oof = {v: [0.0] * len(casos) for v in VARIANTES}
    elegidas = {}
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr = [int(j) for j in tr_idx if not es_h(casos[int(j)])]
        va = [int(j) for j in va_idx]
        gtr = [grupos[j] for j in tr]
        for variante in VARIANTES:
            cols = columnas_de(variante)
            C = elegir_C(Xa[tr][:, cols].tolist(), ya[tr].tolist(), gtr)
            elegidas[(variante, fi)] = C
            params = ajustar(Xa[tr][:, cols].tolist(), ya[tr].tolist(), C)
            for j in va:
                oof[variante][j] = puntuar_vector(params, [X[int(j)][i] for i in cols])
    return {"y": [int(v) for v in ya.tolist()], "grupos": grupos,
            "baseline": base, "oof": oof, "C_elegidas": elegidas,
            "n_splits": n_splits}


def _comparar_en_h_con_idx(res, idx_h, n_boot=2000, semilla=42):
    y_h = [res["y"][i] for i in idx_h]
    comp = [res["oof"]["completo"][i] for i in idx_h]
    base = [res["baseline"][i] for i in idx_h]
    salida = {}
    parejas = {"cifras": "sin_cifras", "marcadores": "sin_marcadores",
               "alineacion": "sin_alineacion"}
    for grupo, variante in parejas.items():
        sin = [res["oof"][variante][i] for i in idx_h]
        salida[grupo] = comparar_scores(y_h, sin, comp,
                                        n_boot=n_boot, semilla=semilla)
    salida["REQ-10"] = comparar_scores(y_h, base, comp,
                                       n_boot=n_boot, semilla=semilla)
    salida["n_h"] = len(idx_h)
    return salida


def ablacion_en_h(casos, res=None, n_boot=2000, semilla=42):
    """Atajo: oof_ablacion + `comparar_scores` restringido a H.

    Para cada grupo: base = sin_grupo, nuevo = completo (Delta > 0 =
    el grupo aporta). REQ-10: base = reglas, nuevo = completo.
    Devuelve (res, comp).
    """
    if res is None:
        res = oof_ablacion(casos)
    idx_h = [i for i, c in enumerate(casos) if es_h(c)]
    return res, _comparar_en_h_con_idx(res, idx_h, n_boot, semilla)


def sigmoide(z):
    """Sigmoide estable en (0, 1). Nunca lanza."""
    try:
        v = float(z)
    except (TypeError, ValueError):
        return 0.5
    try:
        if not math.isfinite(v):
            return 1.0 if v > 0.0 else 0.0
        if v >= 0.0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)
    except (TypeError, ValueError, OverflowError):
        return 1.0 if v > 0.0 else 0.0


def logit_clip(p, eps=EPS_LOGIT):
    """Logit con recorte a [eps, 1-eps] (mismo eps que B.4). Nunca lanza."""
    try:
        q = min(1.0 - float(eps), max(float(eps), float(p)))
    except (TypeError, ValueError):
        return 0.0
    try:
        return math.log(q / (1.0 - q))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def ece(y_true, scores, n_bins=10):
    """Error de calibracion esperado (B.8), bins uniformes. Nunca lanza.

    ECE = Σ_b (|B_b|/n)·|acc(B_b) − conf(B_b)|. Bins vacios aportan 0.
    Entrada vacia o ilegible -> 0.0.
    """
    try:
        y = [int(t) for t in (y_true or [])]
        s = [float(x) for x in (scores or [])]
    except (TypeError, ValueError):
        return 0.0
    try:
        n = int(n_bins)
    except (TypeError, ValueError):
        n = 10
    if n <= 0 or len(y) != len(s) or not y:
        return 0.0
    total = 0.0
    m = len(y)
    for b in range(n):
        lo = b / n
        hi = (b + 1) / n
        if b == n - 1:
            idx = [i for i, v in enumerate(s) if lo <= v <= hi]
        else:
            idx = [i for i, v in enumerate(s) if lo <= v < hi]
        if not idx:
            continue
        acc = sum(y[i] for i in idx) / len(idx)
        conf = sum(s[i] for i in idx) / len(idx)
        total += len(idx) / m * abs(acc - conf)
    try:
        return max(0.0, min(1.0, float(total)))
    except (TypeError, ValueError):
        return 0.0


def ajustar_platt(scores_val, y_val, C=C_PLATT):
    """Ajusta Platt sigmoide(a·z+b) EN VALIDACION, nunca en train.

    scores_val/y_val son predicciones OOF y etiquetas de validacion
    (paso 7: el subconjunto H del OOF). z = logit recortado del score.
    Una sola clase en y_val -> identidad (1.0, 0.0): no hay nada que
    calibrar. Devuelve (a, b).
    """
    if LogisticRegression is None:
        raise RuntimeError("sklearn no disponible")
    y = [int(t) for t in (y_val or [])]
    s = [float(x) for x in (scores_val or [])]
    if len(y) != len(s) or not y:
        raise ValueError("scores_val e y_val deben tener la misma "
                         "longitud no vacia")
    if len(set(y)) < 2:
        return 1.0, 0.0
    Z = np.array([[logit_clip(v)] for v in s], dtype=float)
    clf = LogisticRegression(C=float(C), max_iter=MAX_ITER)
    clf.fit(Z, np.asarray(y, dtype=int))
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def aplicar_platt(scores, a, b):
    """Aplica sigmoide(a·z+b) con z = logit recortado. Nunca lanza.

    Monotona creciente si a > 0 (no cambia el AUROC).
    """
    try:
        aa, bb = float(a), float(b)
    except (TypeError, ValueError):
        return [0.5 for _ in (scores or [])]
    sal = []
    for v in (scores or []):
        try:
            sal.append(sigmoide(aa * logit_clip(float(v)) + bb))
        except (TypeError, ValueError):
            sal.append(0.5)
    return sal


def extraer_vector_final(caso):
    """Vector de 5 features finales (orden COLUMNAS_FINAL). Nunca lanza.

    Proyeccion de `extraer_vector` a IDX_FINAL: cobertura, dispersion,
    margen_min, f_dos_peores (alineacion) + score_base (reglas).
    """
    try:
        v = extraer_vector(caso)
        return [v[i] for i in IDX_FINAL]
    except (TypeError, ValueError, IndexError, AttributeError):
        return [0.0] * 4 + [FALLBACK]


def oof_final(casos, n_splits=5):
    """Scores fuera-de-pliegue de la configuracion FINAL (5 columnas).

    MISMO GroupKFold de `validacion` (`folds_por_grupo` +
    `verificar_disjuncion`, hash de fuentes): cada caso de validacion
    se mide una vez; el entreno de cada fold excluye H; C por CV
    interna solo sobre ese entreno; reescalado B.2 via
    `puntuar_vector`. Devuelve dict con y, grupos, baseline (reglas),
    scores, C_elegidas y n_splits.
    """
    folds = folds_por_grupo(casos, n_splits=n_splits)
    verificar_disjuncion(casos, folds)
    X, y = construir_X(casos)
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=int)
    grupos = grupos_de(casos)
    cols = list(IDX_FINAL)
    base = []
    for c in casos:
        try:
            s = min(1.0, max(0.0, float(puntuar_caso(c))))
        except (TypeError, ValueError, AttributeError):
            s = FALLBACK
        base.append(s)
    oof = [0.0] * len(casos)
    elegidas = {}
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr = [int(j) for j in tr_idx if not es_h(casos[int(j)])]
        va = [int(j) for j in va_idx]
        gtr = [grupos[j] for j in tr]
        C = elegir_C(Xa[tr][:, cols].tolist(), ya[tr].tolist(), gtr)
        elegidas[fi] = C
        params = ajustar(Xa[tr][:, cols].tolist(), ya[tr].tolist(), C)
        for j in va:
            oof[j] = puntuar_vector(params, [X[int(j)][i] for i in cols])
    return {"y": [int(v) for v in ya.tolist()], "grupos": grupos,
            "baseline": base, "scores": oof, "C_elegidas": elegidas,
            "n_splits": n_splits}


if __name__ == "__main__":  # informe reproducible para resultados_paso6.md
    import json

    _ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "datos", "train.jsonl")
    with open(_ruta, encoding="utf-8") as _f:
        _casos = [json.loads(_l) for _l in _f if _l.strip()]
    _res, _comp = ablacion_en_h(_casos)
    print("C elegidas por (variante, fold):")
    for _k in sorted(_res["C_elegidas"]):
        print("  %s fold %d: C=%s" % (_k[0], _k[1], _res["C_elegidas"][_k]))
    print("H: n=%d" % _comp["n_h"])
    for _g in ("cifras", "marcadores", "alineacion"):
        _r = _comp[_g]
        print("completo vs sin_%s: dAUC=%.3f (%.3f,%.3f) dPerd=%.3f (%.3f,%.3f)"
              % (_g, _r["delta_auroc"], _r["ic_auroc"][0], _r["ic_auroc"][1],
                 _r["delta_perdida"], _r["ic_perdida"][0], _r["ic_perdida"][1]))
    _r = _comp["REQ-10"]
    print("REQ-10 completo vs reglas: dAUC=%.3f (%.3f,%.3f) dPerd=%.3f (%.3f,%.3f)"
          % (_r["delta_auroc"], _r["ic_auroc"][0], _r["ic_auroc"][1],
             _r["delta_perdida"], _r["ic_perdida"][0], _r["ic_perdida"][1]))
