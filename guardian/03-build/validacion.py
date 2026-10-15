"""Validacion honesta por grupos de fuentes + medicion en H (REQ-09).

Politica explicita: con `excluir_h_entreno=True` (valor por defecto),
ningun caso con id 'H' se usa para entrenar cuando se mide sobre H.
Los folds se construyen con GroupKFold de sklearn sobre todo train
usando como grupo el hash de las fuentes; despues se filtra H del
entreno de cada fold. El preprocesamiento (StandardScaler) y el
modelo (LogisticRegression) se ajustan SOLO con el entreno del fold;
la metrica se reporta sobre la validacion y sobre su subconjunto H.

Normalizacion exacta de cada fuente (documentada, no inventada):
  1. convertir a str,
  2. `strip()` de espacios extremos,
  3. colapsar cada racha de blancos (`\\s+`) a un solo espacio,
  4. `casefold()` (minusculas agresivas, independiente de locale).
El grupo es el sha256 hex de las fuentes normalizadas unidas con "\\x1f".

Limites honestos (no verificables con los datos entregados):
  - test_publico.jsonl no trae etiquetas: no se puede medir fuga
    test->train por rendimiento, solo impedirla por construccion
    (ver 03-build/carga.py, RNF-04).
  - Dos casos distintos con el mismo texto de fuentes comparten grupo
    a proposito: es la garantia buscada, no un falso positivo.

Comparacion rigurosa de dos sistemas (`perdida_mejor_corte`,
`comparar_scores`): sobre los MISMOS casos, Δ AUROC y Δ perdida con
mejor corte + IC 95 % por bootstrap pareado por casos (2000
remuestreos, percentil 2.5/97.5). Criterio en 04-test/CRITERIO_GATE.md.
"""
import hashlib
import random
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from evaluar_local import auroc, puntuacion

N_SPLITS = 5
SEMILLA = 42

_BLANCOS = re.compile(r"\s+")


def es_h(caso):
    """True si el caso es del subconjunto H (id empieza por 'H')."""
    return str(caso.get("id", "")).startswith("H")


def normalizar_fuente(texto):
    """Normaliza una fuente: strip, colapsa blancos, casefold."""
    return _BLANCOS.sub(" ", str(texto).strip()).casefold()


def grupo_fuentes(caso):
    """Hash sha256 hex de la tupla normalizada de `fuentes`."""
    fuentes = [normalizar_fuente(f) for f in caso.get("fuentes", [])]
    return hashlib.sha256("\x1f".join(fuentes).encode("utf-8")).hexdigest()


def grupos_de(casos):
    """Lista de grupos alineada con `casos`."""
    return [grupo_fuentes(c) for c in casos]


def folds_por_grupo(casos, n_splits=N_SPLITS):
    """Indices (entreno, validacion) con GroupKFold sobre grupos de fuentes."""
    grupos = grupos_de(casos)
    return list(GroupKFold(n_splits=n_splits).split(casos, groups=grupos))


def verificar_disjuncion(casos, folds):
    """Falla con ValueError si algun fold comparte grupos entre entreno y val."""
    grupos = grupos_de(casos)
    for i, (tr, va) in enumerate(folds):
        solape = set(grupos[j] for j in tr) & set(grupos[j] for j in va)
        if solape:
            raise ValueError(f"fold {i}: {len(solape)} grupos en entreno y validacion")
    return True


def extraer_features(caso):
    """Vector numerico solo de `senales` (nunca etiqueta ni tipo).

    Robusto ante claves ausentes (.get con defecto 0.0): minimo de
    respaldo conjunto por oracion, maximo de contradiccion conjunta,
    similitud media, conteos del veredicto llm ponderados por
    confianza y minimo de respaldo entre fragmentos.
    """
    senales = caso.get("senales", {}) or {}
    oraciones = senales.get("oraciones", []) or []
    fragmentos = senales.get("fragmentos", []) or []
    resp = [float(o.get("respaldo_todas", 0.0) or 0.0) for o in oraciones]
    cont = [float(o.get("contradiccion_todas", 0.0) or 0.0) for o in oraciones]
    sims = [float(s) for o in oraciones for s in (o.get("similitud_por_fuente", []) or [])]
    llm_resp = llm_mala = llm_contra = 0.0
    for o in oraciones:
        llm = o.get("llm", {}) or {}
        conf = float(llm.get("confianza", 0.0) or 0.0)
        veredicto = llm.get("veredicto")
        if veredicto == "respaldada":
            llm_resp += conf
        elif veredicto == "no_respaldada":
            llm_mala += conf
        elif veredicto == "contradicha":
            llm_contra += conf
    frag_resp = [float(f.get("respaldo_todas", 0.0) or 0.0) for f in fragmentos]
    n_or = float(len(oraciones)) or 1.0
    return [
        min(resp) if resp else 0.0,
        max(cont) if cont else 0.0,
        sum(sims) / len(sims) if sims else 0.0,
        llm_resp / n_or,
        llm_mala / n_or,
        llm_contra / n_or,
        min(frag_resp) if frag_resp else 0.0,
        float(len(oraciones)),
    ]


def _etiquetas(casos):
    return [int(c["etiqueta"]) for c in casos]


def evaluar_folds(casos, n_splits=N_SPLITS, excluir_h_entreno=True):
    """Entrena por fold (solo entreno del fold) y mide en validacion y en H.

    Devuelve una lista por fold con puntuacion/auroc globales y del
    subconjunto H de la validacion, mas tamanos. Si un fold no tiene
    casos H en validacion, sus metricas H son None (se informa, no se inventa).
    """
    folds = folds_por_grupo(casos, n_splits=n_splits)
    verificar_disjuncion(casos, folds)
    informe = []
    for tr_idx, va_idx in folds:
        if excluir_h_entreno:
            tr_idx = [j for j in tr_idx if not es_h(casos[j])]
        X_tr = [extraer_features(casos[j]) for j in tr_idx]
        y_tr = _etiquetas([casos[j] for j in tr_idx])
        scaler = StandardScaler().fit(X_tr)
        modelo = LogisticRegression(max_iter=2000, random_state=SEMILLA)
        modelo.fit(scaler.transform(X_tr), y_tr)
        X_va = scaler.transform([extraer_features(casos[j]) for j in va_idx])
        scores = [float(s) for s in modelo.predict_proba(X_va)[:, 1]]
        y_va = _etiquetas([casos[j] for j in va_idx])
        pred = [1 if s >= 0.5 else 0 for s in scores]
        idx_h = [k for k, j in enumerate(va_idx) if es_h(casos[j])]
        fila = {
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "n_h_val": len(idx_h),
            "puntuacion": puntuacion(y_va, pred),
            "auroc": auroc(y_va, scores),
            "puntuacion_h": None,
            "auroc_h": None,
        }
        if idx_h:
            y_h = [y_va[k] for k in idx_h]
            s_h = [scores[k] for k in idx_h]
            p_h = [pred[k] for k in idx_h]
            fila["puntuacion_h"] = puntuacion(y_h, p_h)
            fila["auroc_h"] = auroc(y_h, s_h)
        informe.append(fila)
    return informe


def evaluar_bot_folds(bot_cls, casos, n_splits=N_SPLITS, umbral=0.5):
    """Mide un bot sin entrenamiento por fold (para baselines de reglas).

    Reusa los mismos folds GroupKFold por hash de fuentes (con
    verificacion de disjuncion): como las reglas no entrenan, no hay
    fuga posible y cada caso H de validacion se mide exactamente una
    vez. Reporta puntuacion/AUROC globales y del subconjunto H.
    """
    from generar_predicciones import _caso_para_bot

    folds = folds_por_grupo(casos, n_splits=n_splits)
    verificar_disjuncion(casos, folds)
    bot = bot_cls()
    informe = []
    for tr_idx, va_idx in folds:
        val = [casos[j] for j in va_idx]
        scores = []
        for c in val:
            try:
                s = float(bot.score(_caso_para_bot(c)))
            except (TypeError, ValueError, AttributeError, KeyError):
                s = 0.55
            scores.append(min(1.0, max(0.0, s)))
        y_va = _etiquetas(val)
        pred = [1 if s >= umbral else 0 for s in scores]
        idx_h = [k for k, c in enumerate(val) if es_h(c)]
        fila = {
            "n_val": len(va_idx),
            "n_h_val": len(idx_h),
            "puntuacion": puntuacion(y_va, pred),
            "auroc": auroc(y_va, scores),
            "puntuacion_h": None,
            "auroc_h": None,
        }
        if idx_h:
            y_h = [y_va[k] for k in idx_h]
            s_h = [scores[k] for k in idx_h]
            p_h = [pred[k] for k in idx_h]
            fila["puntuacion_h"] = puntuacion(y_h, p_h)
            fila["auroc_h"] = auroc(y_h, s_h)
        informe.append(fila)
    return informe


N_BOOT = 2000
SEMILLA_BOOT = 42


def _auroc_rangos(y, s):
    """AUROC por rangos medios (Mann-Whitney, O(n log n)).

    Algebraicamente identico a `evaluar_local.auroc` (suma por pares
    con 0.5 por empate); 0.5 si solo hay una clase. Se usa dentro del
    bootstrap para que 2000 remuestreos sobre H (~631 casos) corran
    en segundos en stdlib.
    """
    try:
        n = len(y)
        n_pos = sum(1 for t in y if t == 1)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        orden = sorted(range(n), key=lambda k: s[k])
        suma_rangos = 0.0
        i = 0
        while i < n:
            j = i
            while j + 1 < n and s[orden[j + 1]] == s[orden[i]]:
                j += 1
            rango_medio = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                if y[orden[k]] == 1:
                    suma_rangos += rango_medio
            i = j + 1
        return (suma_rangos - n_pos * (n_pos + 1) / 2.0) / \
            (n_pos * n_neg)
    except (TypeError, ValueError, IndexError):
        return 0.5


def perdida_mejor_corte(y_true, scores, w_fn=4.0):
    """Perdida con mejor corte: min_t [4*FN(t)+FP(t)] / N_f (B.16).

    pred(t) = 1 si score >= t. Candidatos: cada score distinto mas el
    corte "todo a 0" (t por encima del maximo). N_f = n. de fieles
    (etiqueta 0); si N_f == 0 la perdida minima es 0.0 (t en el minimo
    deja FN = 0, igual que el borde de METRICA.md). Barrido O(n log n)
    tras ordenar: para cada corte, FN = positivos bajo t y FP =
    negativos sobre t. Nunca lanza ante entradas ilegibles: 0.0.
    """
    try:
        y = [int(t) for t in (y_true or [])]
        s = [float(x) for x in (scores or [])]
    except (TypeError, ValueError):
        return 0.0
    try:
        if len(y) != len(s) or not y:
            return 0.0
        n_f = sum(1 for t in y if t == 0)
        if n_f == 0:
            return 0.0
        try:
            peso = float(w_fn)
        except (TypeError, ValueError):
            peso = 4.0
        pares = sorted(zip(s, y), key=lambda p: p[0])
        n = len(pares)
        neg_total = n_f
        pos_total = n - n_f
        neg_bajo = 0
        pos_bajo = 0
        i = 0
        mejor = (peso * float(pos_total)) / float(n_f)  # todo a 0
        while i < n:
            # Corte t: FN = positivos bajo t, FP = negativos sobre t.
            fn = pos_bajo
            fp = neg_total - neg_bajo
            try:
                perdida = (peso * float(fn) + float(fp)) / float(n_f)
            except (TypeError, ValueError, ZeroDivisionError):
                perdida = mejor
            if perdida < mejor:
                mejor = perdida
            t = pares[i][0]
            while i < n and pares[i][0] == t:
                if pares[i][1] == 1:
                    pos_bajo += 1
                else:
                    neg_bajo += 1
                i += 1
        return max(0.0, mejor)
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return 0.0


def comparar_scores(y_true, scores_base, scores_nuevo, n_boot=N_BOOT,
                    semilla=SEMILLA_BOOT):
    """Compara dos vectores de scores sobre los MISMOS casos.

    Δ AUROC = AUROC(nuevo) - AUROC(base) (positivo = mejor ordenacion;
    AUROC por rangos medios, identico a `evaluar_local.auroc`).
    Δ perdida = perdida(base) - perdida(nuevo) con
    `perdida_mejor_corte` (positivo = menos perdida, mejor; el signo
    se define asi para que el gate pida IC > 0 en ambas).
    IC 95 % por bootstrap PAREADO por casos: cada remuestreo sortea n
    indices con reposicion y evalua ambos sistemas sobre los mismos,
    de modo que la correlacion por caso se conserva; el intervalo es
    el percentil 2.5/97.5 de las n_boot Δ (por defecto 2000).

    Criterio (ver 04-test/CRITERIO_GATE.md): la mejora es real solo si
    el limite inferior del IC 95 % queda claramente > 0 en AMBAS
    (o si arregla un tipo concreto de EJEMPLOS.md caso por caso).

    Devuelve dict con delta_auroc, ic_auroc (tupla), delta_perdida,
    ic_perdida (tupla), n y n_boot. Falla con ValueError si las
    longitudes difieren o no hay casos.
    """
    try:
        y = [int(t) for t in (y_true or [])]
        a = [float(x) for x in (scores_base or [])]
        b = [float(x) for x in (scores_nuevo or [])]
    except (TypeError, ValueError) as e:
        raise ValueError("entradas no numericas") from e
    if not y or len(y) != len(a) or len(y) != len(b):
        raise ValueError("y_true, scores_base y scores_nuevo deben "
                         "tener la misma longitud no vacia")
    try:
        n_rep = int(n_boot)
    except (TypeError, ValueError):
        n_rep = N_BOOT
    if n_rep <= 0:
        raise ValueError("n_boot debe ser > 0")
    try:
        rng = random.Random(int(semilla))
    except (TypeError, ValueError):
        rng = random.Random(SEMILLA_BOOT)
    n = len(y)
    d_auroc = _auroc_rangos(y, b) - _auroc_rangos(y, a)
    d_perd = perdida_mejor_corte(y, a) - perdida_mejor_corte(y, b)
    dist_a, dist_p = [], []
    for _ in range(n_rep):
        try:
            idx = [rng.randrange(n) for _ in range(n)]
        except (TypeError, ValueError):
            break
        try:
            yb = [y[k] for k in idx]
            ab = [a[k] for k in idx]
            bb = [b[k] for k in idx]
            dist_a.append(_auroc_rangos(yb, bb) - _auroc_rangos(yb, ab))
            dist_p.append(perdida_mejor_corte(yb, ab) -
                          perdida_mejor_corte(yb, bb))
        except (TypeError, ValueError, IndexError):
            continue
    if not dist_a:
        raise ValueError("bootstrap sin replicas validas")
    dist_a.sort()
    dist_p.sort()
    m = len(dist_a)
    lo_i = min(m - 1, max(0, int(0.025 * m)))
    hi_i = min(m - 1, max(0, int(0.975 * m)))
    return {
        "delta_auroc": float(d_auroc),
        "ic_auroc": (float(dist_a[lo_i]), float(dist_a[hi_i])),
        "delta_perdida": float(d_perd),
        "ic_perdida": (float(dist_p[lo_i]), float(dist_p[hi_i])),
        "n": n,
        "n_boot": m,
    }
