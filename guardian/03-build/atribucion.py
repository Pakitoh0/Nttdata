"""Atribucion de cifras + desacuerdo con el juez (PASO 5).

Solo libreria estandar. Espanol neutro.

FUNDAMENTO B.6 (atribucion) y B.4 (juez + desacuerdo):

- idf intra-caso SOLO sobre las fuentes (m = n. de fuentes):

      idf(t) = ln((m + 1) / (df(t) + 1)) + 1

- Por cada cifra de la respuesta, ventana de +-K tokens (K = 10) en la
  respuesta y en cada fuente. Una "cifra" es un token con digito
  ("42", "28" de "28.000") o una palabra-numero/magnitud en espanol
  ("tres", "veintiocho", "mil"); aproximacion documentada: el
  tokenizador parte "28.000" en ("28", "000") y cada parte cuenta como
  ancla propia con su ventana.
- attr(n) = maximo sobre las ventanas-ancla o de las fuentes del
  Jaccard ponderado por idf entre W_R(n) y W_S(o):

      J = suma_{t en inter} idf(t) / suma_{t en union} idf(t)

  (union vacia -> 0.0). La cifra NO se excluye de la ventana: el
  contexto compartido decide, y por eso la atribucion cruzada (misma
  cifra, entidad distinta) puntua a medio camino, no a cero.
- f_attr = min_n attr(n) (una sola cifra mal atribuida basta).
- margen(n) = attr(n) - mejor ventana de OTRA fuente (0.0 si ninguna
  otra fuente tiene ancla: distancia a "nada respalda", igual que
  alineacion.py con A(2) = 0). margen = min_n margen(n): margen bajo
  con f_attr medio = entidades confundibles (atribucion cruzada).
- Sin cifras en la respuesta -> (f_attr, margen) = (1.0, 1.0)
  (vacuidad: no hay nada que atribuir mal).

- Desacuerdo con el juez por oracion (B.4), con la MISMA convencion
  que reglas.py::prob_juez_llm (veredicto malo -> kappa, "respaldada"
  -> 1 - kappa, null/ilegible -> 0.5 neutral, todo recortado a
  [EPS, 1 - EPS] con EPS = 1e-4; Platt q' NO ajustada aqui: sin
  entrenamiento en este paso, se usa q_LLM cruda y la calibracion
  queda para la futura regresion logistica):

      q_LLM = P("no respaldada") segun el juez
      a_i   = max(max(respaldo_por_fuente), respaldo_todas)
      d1 = l(q_LLM) - l(1 - a_i)        juez y NLI discrepan
      d2 = (1 - q_LLM) * (1 - a_i)       "respaldada" con NLI vacia
      d3 = (1 - q_LLM) * contradiccion_NLI (contradiccion_todas)

  con l(x) = logit con recorte EPS. Agregacion "una falsa basta":
  d1min = min (el minimo captura la direccion "juez demasiado
  confiado", la que cuesta 4:1), d2max = max, d3max = max.
  Sin oraciones -> (0.0, 0.0, 0.0) neutral.

- Ausencia de respaldo/contradiccion -> 0.5 neutral (igual que
  reglas.py); valor presente pero NaN/inf -> 0.0; todo acceso con
  .get()/try-except. Caso ilegible -> (1.0, 1.0, 0.0, 0.0, 0.0).
  Nunca lanza.

ESTADO: pendiente de gate (ver 04-test/CRITERIO_GATE.md). Modulo
aislado: expone features via `feats_atrib` para la futura regresion
logistica; NO cableado a mano en `reglas.py`.
"""

import math
import re
import unicodedata

# Ventana de contexto alrededor de cada cifra (B.6: +-k, k = 10).
K_VENTANA = 10
# Recorte del logit (B.4, igual que reglas.py).
EPS = 1e-4

# Veredictos que afirman "no respaldada" (igual que reglas.py).
_VEREDICTO_MALO = ("no_respaldada", "contradicha")

_RE_TOKEN = re.compile(r"[a-z0-9]+")
_RE_DIGITO = re.compile(r"[0-9]")

# Palabras-numero y magnitudes (copiadas de cifras.py: anclas sin
# digito; el tokenizador ya quita acentos, ambas variantes valen).
_NUM_PALABRAS = frozenset((
    "cero", "uno", "un", "una", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
    "trece", "catorce", "quince", "dieciseis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "veintiuno", "veintiun",
    "veintidos", "veintitres", "veinticuatro", "veinticinco",
    "veintiseis", "veintisiete", "veintiocho", "veintinueve",
    "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
    "ochenta", "noventa", "cien", "ciento", "doscientos",
    "trescientos", "cuatrocientos", "quinientos", "seiscientos",
    "setecientos", "ochocientos", "novecientos",
))
_MAGNITUDES = frozenset(("mil", "millar", "millares", "millon",
                         "millones"))


def _norm(texto):
    """Minusculas sin acentos (NFKD + casefold). Ilegible -> ""."""
    try:
        if texto is None:
            return ""
        base = unicodedata.normalize("NFKD", str(texto))
        sin = "".join(c for c in base if not unicodedata.combining(c))
        return sin.casefold()
    except (TypeError, ValueError, AttributeError):
        return ""


def _tokens(texto):
    """Tokens [a-z0-9]+ del texto normalizado. Nunca lanza."""
    try:
        return _RE_TOKEN.findall(_norm(texto))
    except (TypeError, ValueError):
        return []


def _es_cifra(tok):
    """True si el token es ancla de cifra (digito o palabra-numero)."""
    try:
        if not tok:
            return False
        if _RE_DIGITO.search(tok):
            return True
        return tok in _NUM_PALABRAS or tok in _MAGNITUDES
    except (TypeError, ValueError):
        return False


def _recortar(p):
    """Recorta una probabilidad a [EPS, 1 - EPS]."""
    try:
        return min(1.0 - EPS, max(EPS, float(p)))
    except (TypeError, ValueError):
        return 0.5


def logit(p):
    """Logit con recorte EPS (B.4, igual que reglas.py)."""
    q = _recortar(p)
    return math.log(q / (1.0 - q))


def prob_juez_llm(veredicto, confianza):
    """P("no respaldada") segun el juez (misma convencion que reglas.py).

    Veredicto malo -> kappa; "respaldada" -> 1 - kappa; null o
    ilegible -> 0.5 neutral. Nunca lanza.
    """
    try:
        if veredicto is None:
            return 0.5
        if veredicto in _VEREDICTO_MALO:
            return _recortar(confianza)
        if veredicto == "respaldada":
            return _recortar(1.0 - float(confianza))
        return 0.5
    except (TypeError, ValueError):
        return 0.5


def _num_neutro(x, defecto=0.5):
    """Float de una senal NLI: ausente -> defecto neutral, NaN/inf -> 0.0."""
    try:
        if x is None:
            return defecto
        v = float(x)
    except (TypeError, ValueError):
        return defecto
    try:
        if not math.isfinite(v):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return v


def _finito(x, defecto=0.0):
    """x si es finito, si no defecto. Nunca lanza."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return defecto
    except (TypeError, ValueError):
        return defecto


def _idf_fuentes(docs_tokens):
    """idf intra-caso sobre las fuentes: ln((m+1)/(df+1)) + 1."""
    try:
        m = len(docs_tokens)
        if m == 0:
            return {}
        df = {}
        for doc in docs_tokens:
            try:
                for t in set(doc):
                    df[t] = df.get(t, 0) + 1
            except (TypeError, ValueError, AttributeError):
                continue
        pesos = {}
        for t, n in df.items():
            try:
                pesos[t] = math.log((m + 1.0) / (n + 1.0)) + 1.0
            except (TypeError, ValueError, ZeroDivisionError):
                pesos[t] = 1.0
        return pesos
    except (TypeError, ValueError, AttributeError):
        return {}


def _jaccard_pond(a, b, pesos):
    """Jaccard ponderado por idf. Union vacia -> 0.0. Nunca lanza."""
    try:
        union = set(a) | set(b)
        if not union:
            return 0.0
        inter = set(a) & set(b)
        num = sum(_finito(pesos.get(t, 1.0), 1.0) for t in inter)
        den = sum(_finito(pesos.get(t, 1.0), 1.0) for t in union)
        if den <= 0.0 or not math.isfinite(num / den):
            return 0.0
        return max(0.0, min(1.0, num / den))
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return 0.0


def _ventana(tokens, pos, k=K_VENTANA):
    """Ventana tokens[pos-k:pos+k+1]. Nunca lanza."""
    try:
        ini = max(0, int(pos) - int(k))
        return list(tokens[ini:int(pos) + int(k) + 1])
    except (TypeError, ValueError, AttributeError):
        return []


def _anclas(tokens):
    """Posiciones de tokens-cifra. Nunca lanza."""
    try:
        return [i for i, t in enumerate(tokens) if _es_cifra(t)]
    except (TypeError, ValueError, AttributeError):
        return []


def _attr_cifra(toks_r, pos_r, fuentes_toks, pesos):
    """(attr, margen) de una cifra en pos_r (B.6).

    attr = mejor Jaccard sobre ventanas-ancla de las fuentes;
    margen = mejor - mejor de otra fuente (0.0 si no hay otra).
    """
    try:
        vent_r = _ventana(toks_r, pos_r)
        if not vent_r:
            return 0.0, 0.0
        mejor = 0.0
        fuente_mejor = -1
        por_fuente = []
        for j, toks_s in enumerate(fuentes_toks):
            try:
                top = 0.0
                for o in _anclas(toks_s):
                    try:
                        s = _jaccard_pond(vent_r, _ventana(toks_s, o),
                                          pesos)
                    except (TypeError, ValueError):
                        continue
                    if s > top:
                        top = s
                por_fuente.append(top)
                if top > mejor:
                    mejor = top
                    fuente_mejor = j
            except (TypeError, ValueError, AttributeError):
                por_fuente.append(0.0)
                continue
        ajena = 0.0
        for j, top in enumerate(por_fuente):
            try:
                if j != fuente_mejor and top > ajena:
                    ajena = top
            except (TypeError, ValueError):
                continue
        return _finito(mejor), _finito(max(0.0, mejor - ajena))
    except (TypeError, ValueError, AttributeError):
        return 0.0, 0.0


def _desacuerdo_oracion(oracion):
    """(d1, d2, d3) de una oracion (B.4). Nunca lanza."""
    try:
        datos = oracion if isinstance(oracion, dict) else {}
        try:
            todas = _num_neutro(datos.get("respaldo_todas", 0.5))
        except (TypeError, ValueError, AttributeError):
            todas = 0.5
        try:
            brutas = datos.get("respaldo_por_fuente", None)
            if brutas is None:
                vals = []
            elif isinstance(brutas, (str, bytes)):
                vals = []
            else:
                vals = [_num_neutro(v, 0.5) for v in list(brutas)]
        except (TypeError, ValueError, AttributeError):
            vals = []
        try:
            a = max(vals + [todas])
        except (TypeError, ValueError):
            a = 0.5
        try:
            contra = _num_neutro(datos.get("contradiccion_todas", 0.5))
        except (TypeError, ValueError, AttributeError):
            contra = 0.5
        try:
            llm = datos.get("llm", {}) or {}
            if not isinstance(llm, dict):
                llm = {}
            q = prob_juez_llm(llm.get("veredicto"),
                              llm.get("confianza"))
        except (TypeError, ValueError, AttributeError):
            q = 0.5
        try:
            d1 = logit(q) - logit(1.0 - a)
        except (TypeError, ValueError):
            d1 = 0.0
        try:
            d2 = (1.0 - q) * (1.0 - a)
            d3 = (1.0 - q) * contra
        except (TypeError, ValueError):
            d2, d3 = 0.0, 0.0
        return _finito(d1), _finito(d2), _finito(d3)
    except (TypeError, ValueError, AttributeError):
        return 0.0, 0.0, 0.0


def feats_atrib(caso):
    """(f_attr, margen, d1min, d2max, d3max) de un caso (B.4 + B.6).

    Solo lee respuesta/fuentes/senales (nunca etiqueta ni tipo).
    Ilegible -> (1.0, 1.0, 0.0, 0.0, 0.0). Nunca lanza.
    """
    try:
        if not isinstance(caso, dict):
            return (1.0, 1.0, 0.0, 0.0, 0.0)
        try:
            respuesta = caso.get("respuesta", "") or ""
        except (TypeError, ValueError, AttributeError):
            respuesta = ""
        try:
            brutas = caso.get("fuentes", []) or []
            if isinstance(brutas, str):
                textos = [brutas]
            else:
                textos = [str(f) for f in list(brutas)]
        except (TypeError, ValueError, AttributeError):
            textos = []
        toks_r = _tokens(respuesta)
        fuentes_toks = [_tokens(t) for t in textos]
        pesos = _idf_fuentes(fuentes_toks)
        try:
            anclas_r = _anclas(toks_r)
        except (TypeError, ValueError):
            anclas_r = []
        if not anclas_r:
            f_attr, margen = 1.0, 1.0
        else:
            f_attr, margen = 1.0, 1.0
            for p in anclas_r:
                try:
                    a_n, m_n = _attr_cifra(toks_r, p, fuentes_toks,
                                           pesos)
                except (TypeError, ValueError, AttributeError):
                    continue
                if a_n < f_attr:
                    f_attr = a_n
                if m_n < margen:
                    margen = m_n
        try:
            senales = caso.get("senales", {}) or {}
            if not isinstance(senales, dict):
                oraciones = []
            else:
                oraciones = senales.get("oraciones", []) or []
                if not isinstance(oraciones, list):
                    oraciones = []
        except (TypeError, ValueError, AttributeError):
            oraciones = []
        if not oraciones:
            d1min, d2max, d3max = 0.0, 0.0, 0.0
        else:
            d1min, d2max, d3max = None, None, None
            for ora in oraciones:
                try:
                    d1, d2, d3 = _desacuerdo_oracion(ora)
                except (TypeError, ValueError, AttributeError):
                    continue
                if d1min is None or d1 < d1min:
                    d1min = d1
                if d2max is None or d2 > d2max:
                    d2max = d2
                if d3max is None or d3 > d3max:
                    d3max = d3
            if d1min is None:
                d1min, d2max, d3max = 0.0, 0.0, 0.0
        return (_finito(f_attr, 1.0), _finito(margen, 1.0),
                _finito(d1min, 0.0), _finito(d2max, 0.0),
                _finito(d3max, 0.0))
    except (TypeError, ValueError, AttributeError):
        return (1.0, 1.0, 0.0, 0.0, 0.0)
