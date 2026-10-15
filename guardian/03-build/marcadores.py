"""Marcadores diferenciales de causa, alcance y matiz + flip de polaridad (PASO 4).

Solo libreria estandar. Espanol neutro.

FUNDAMENTO B.6 (marcadores) y B.12 (polaridad):

- f_caus = 1[existe marcador causal en R ausente en la fuente citada].
  C = {gracias a, debido a, porque, provoco/provoca/provocan/provocado,
  a causa de, como consecuencia, por culpa de, a raiz de,
  como resultado, por causa de}.
- f_alc = 1[existe cuantificador en R ausente en la fuente citada].
  Q = {todos, todas, todo, toda, siempre, cada, ningun/ninguno/ninguna,
  nunca, jamas, nadie, la totalidad, en su totalidad}.
- f_hedge = n. de matices en la fuente citada ausentes en R.
  H = {podria/podrian, aproximadamente, mas de, al menos, solo/solamente,
  hasta, casi, alrededor de, en torno a, cerca de, quizas, posiblemente,
  probablemente, unos/unas, a lo sumo, como maximo}.
  Ventana +-3 tokens (aproximacion documentada): cada ocurrencia de un
  matiz en la fuente citada solo cuenta si alguna palabra de contenido
  de su ventana +-3 (excluidos los tokens del propio marcador) aparece
  tambien en R. Asi no se cuentan matices de frases que R ni retoma
  (p. ej. un "solo" de otra seccion no citada). Si la ventana no tiene
  contenido compartido, la ocurrencia no cuenta.
- flip (B.12) = 1[signo(v_R) * signo(v_S) = -1 y Jaccard ponderado del
  objeto >= 0,5]. Lexico firmado minimo: +1 {aumento, subio, crecio,
  gano, abrio, aprobo}; -1 {cayo, bajo, disminuyo, perdio, cerro,
  rechazo}. Solo formas base normalizadas (ver _norm). Aproximacion
  documentada del objeto: palabras de contenido en ventana +-5 tokens
  alrededor del verbo (excluido el verbo); Jaccard ponderado con idf
  intra-caso idf(t) = ln((m+1)/(df(t)+1)) + 1 sobre los documentos
  [R + cada fuente] (misma formula que B.6); union vacia -> 0,0.

"Fuente citada": la fuente indicada por el veredicto llm de cada oracion
(senales.oraciones[].llm.fuente, indice en `fuentes`); si varias
oraciones citan fuentes distintas se concatenan; si no hay ninguna
fuente valida citada, se concatenan todas. R = caso.respuesta.

Normalizacion: minusculas agresivas (casefold) + quitar acentos con
unicodedata NFKD (p. ej. "provoco" casa con "provocó", "mas de" con
"más de", "solo" con "sólo"). Deteccion por secuencia de tokens
(una palabra no casa dentro de otra: "cada" no casa en "cadaver").

Todo acceso defensivo (.get()/try-except); NaN/inf no aplica (conteos
enteros); ante caso ilegible devuelve (0, 0, 0, 0). Nunca lanza.

ESTADO: pendiente de gate (ver 04-test/CRITERIO_GATE.md). Modulo
aislado: expone features via `feats_marc` para la futura regresion
logistica; NO cableado a mano en `reglas.py`.
"""

import math
import re
import unicodedata

# Ventana de contexto del matiz (B.6: +-3 tokens).
VENTANA_MATIZ = 3
# Ventana del objeto del verbo (aproximacion documentada de B.12).
VENTANA_OBJETO = 5
# Umbral de objeto compartido (B.12).
UMBRAL_JACCARD = 0.5

# Marcadores causales (tuplas de tokens ya normalizados).
_CAUSALES = (
    ("gracias", "a"),
    ("debido", "a"),
    ("porque",),
    ("provoco",),
    ("provoca",),
    ("provocan",),
    ("provocado",),
    ("provocada",),
    ("a", "causa", "de"),
    ("como", "consecuencia"),
    ("por", "culpa", "de"),
    ("a", "raiz", "de"),
    ("como", "resultado"),
    ("por", "causa", "de"),
)

# Cuantificadores de alcance (tuplas de tokens ya normalizados).
_ALCANCE = (
    ("todos",),
    ("todas",),
    ("todo",),
    ("toda",),
    ("siempre",),
    ("cada",),
    ("ningun",),
    ("ninguno",),
    ("ninguna",),
    ("nunca",),
    ("jamas",),
    ("nadie",),
    ("la", "totalidad"),
    ("en", "su", "totalidad"),
)

# Matices/hedges (tuplas de tokens ya normalizados).
_MATICES = (
    ("podria",),
    ("podrian",),
    ("aproximadamente",),
    ("mas", "de"),
    ("al", "menos"),
    ("solo",),
    ("solamente",),
    ("hasta",),
    ("casi",),
    ("alrededor", "de"),
    ("en", "torno", "a"),
    ("cerca", "de"),
    ("quizas",),
    ("posiblemente",),
    ("probablemente",),
    ("unos",),
    ("unas",),
    ("a", "lo", "sumo"),
    ("como", "maximo"),
)

# Lexico firmado minimo (B.12), formas normalizadas (sin acentos).
_VERBO_POS = frozenset(("aumento", "subio", "crecio", "gano", "abrio",
                        "aprobo"))
_VERBO_NEG = frozenset(("cayo", "bajo", "disminuyo", "perdio", "cerro",
                        "rechazo"))

# Paradas minimas para objeto/ventana (no exhaustiva, documentada).
_STOP = frozenset((
    "el", "la", "los", "las", "del", "al", "un", "una", "unos", "unas",
    "de", "en", "y", "o", "a", "ante", "con", "por", "para", "segun",
    "que", "se", "su", "sus", "le", "les", "lo", "como", "es", "fue",
    "son", "fueron", "ha", "han", "hay", "este", "esta", "estos",
    "estas", "ese", "esa", "aquel", "muy", "mas", "menos", "tan",
    "pero", "porque", "si", "no", "ni", "ya", "el", "e",
))

_RE_TOKEN = re.compile(r"[a-z0-9]+")


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


def _contiene(tokens, marcador):
    """True si la secuencia `marcador` aparece en `tokens`."""
    try:
        n = len(marcador)
        if n == 0 or len(tokens) < n:
            return False
        primero = marcador[0]
        for i, tok in enumerate(tokens):
            try:
                if tok == primero and tuple(tokens[i:i + n]) == tuple(marcador):
                    return True
            except (TypeError, ValueError, IndexError):
                continue
        return False
    except (TypeError, ValueError, AttributeError):
        return False


def _ocurrencias(tokens, marcador):
    """Indices de inicio de cada ocurrencia de `marcador` en `tokens`."""
    try:
        n = len(marcador)
        if n == 0 or len(tokens) < n:
            return []
        primero = marcador[0]
        hall = []
        for i, tok in enumerate(tokens):
            try:
                if tok == primero and tuple(tokens[i:i + n]) == tuple(marcador):
                    hall.append(i)
            except (TypeError, ValueError, IndexError):
                continue
        return hall
    except (TypeError, ValueError, AttributeError):
        return []


def _es_contenido(tok):
    """True si el token aporta contenido (no parada, longitud > 2)."""
    try:
        return len(tok) > 2 and tok not in _STOP
    except (TypeError, ValueError):
        return False


def _fuente_citada(caso):
    """Texto de la fuente citada: la del veredicto llm si existe.

    Recorre senales.oraciones[].llm.fuente; cada entero valido es un
    indice en `fuentes`. Si hay al menos uno valido, concatena esas
    fuentes; si no, concatena todas. Ilegible -> "".
    """
    try:
        if not isinstance(caso, dict):
            return ""
        fuentes = caso.get("fuentes", []) or []
        try:
            textos = [str(f) for f in fuentes]
        except (TypeError, ValueError):
            textos = []
        if not textos:
            return ""
        senales = caso.get("senales", {}) or {}
        if not isinstance(senales, dict):
            senales = {}
        oraciones = senales.get("oraciones", []) or []
        indices = []
        try:
            for ora in oraciones:
                try:
                    if not isinstance(ora, dict):
                        continue
                    llm = ora.get("llm", {}) or {}
                    if not isinstance(llm, dict):
                        continue
                    idx = llm.get("fuente")
                    if isinstance(idx, bool):
                        continue
                    if isinstance(idx, int) and 0 <= idx < len(textos):
                        if idx not in indices:
                            indices.append(idx)
                except (TypeError, ValueError, AttributeError):
                    continue
        except (TypeError, ValueError, AttributeError):
            indices = []
        if indices:
            try:
                return "\n".join(textos[i] for i in indices)
            except (TypeError, ValueError, IndexError):
                pass
        return "\n".join(textos)
    except (TypeError, ValueError, AttributeError):
        return ""


def _f_causal(toks_r, toks_s):
    """1 si hay marcador causal en R ausente en la fuente citada."""
    try:
        for marc in _CAUSALES:
            try:
                if _contiene(toks_r, marc) and not _contiene(toks_s, marc):
                    return 1
            except (TypeError, ValueError):
                continue
        return 0
    except (TypeError, ValueError, AttributeError):
        return 0


def _f_alcance(toks_r, toks_s):
    """1 si hay cuantificador en R ausente en la fuente citada."""
    try:
        for marc in _ALCANCE:
            try:
                if _contiene(toks_r, marc) and not _contiene(toks_s, marc):
                    return 1
            except (TypeError, ValueError):
                continue
        return 0
    except (TypeError, ValueError, AttributeError):
        return 0


def _f_matices(toks_r, toks_s):
    """N. de matices en la fuente citada ausentes en R (ventana +-3).

    Cada ocurrencia de un matiz en S cuenta si el matiz falta en R y
    alguna palabra de contenido de su ventana +-3 (sin los tokens del
    marcador) aparece en R.
    """
    total = 0
    try:
        set_r = set(toks_r)
        for marc in _MATICES:
            try:
                if _contiene(toks_r, marc):
                    continue
                for ini in _ocurrencias(toks_s, marc):
                    try:
                        fin = ini + len(marc)
                        vent = toks_s[max(0, ini - VENTANA_MATIZ):ini] + \
                            toks_s[fin:fin + VENTANA_MATIZ]
                        if any(t in set_r for t in vent if _es_contenido(t)):
                            total += 1
                    except (TypeError, ValueError, IndexError):
                        continue
            except (TypeError, ValueError):
                continue
        return int(total)
    except (TypeError, ValueError, AttributeError):
        return 0


def _signo(tok):
    """+1/-1/0 segun el lexico firmado minimo (B.12)."""
    try:
        if tok in _VERBO_POS:
            return 1
        if tok in _VERBO_NEG:
            return -1
        return 0
    except (TypeError, ValueError):
        return 0


def _objeto(tokens, pos):
    """Contenido en ventana +-5 alrededor de `pos` (sin el verbo)."""
    try:
        ini = max(0, pos - VENTANA_OBJETO)
        fin = pos + VENTANA_OBJETO + 1
        return {t for i, t in enumerate(tokens[ini:fin], start=ini)
                if i != pos and _es_contenido(t)}
    except (TypeError, ValueError, AttributeError):
        return set()


def _idf(docs_tokens):
    """idf intra-caso: ln((m+1)/(df+1)) + 1 (misma formula que B.6)."""
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
            except (TypeError, ValueError):
                pesos[t] = 1.0
        return pesos
    except (TypeError, ValueError, AttributeError):
        return {}


def _jaccard_pond(a, b, pesos):
    """Jaccard ponderado por idf. Union vacia -> 0.0."""
    try:
        union = set(a) | set(b)
        if not union:
            return 0.0
        inter = set(a) & set(b)
        num = sum(float(pesos.get(t, 1.0)) for t in inter)
        den = sum(float(pesos.get(t, 1.0)) for t in union)
        if den <= 0.0:
            return 0.0
        return max(0.0, min(1.0, num / den))
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return 0.0


def _flip(toks_r, toks_s, docs_tokens):
    """1 si hay verbos de signo opuesto sobre el mismo objeto (B.12)."""
    try:
        pesos = _idf(docs_tokens)
        verbos_r = [(i, _signo(t)) for i, t in enumerate(toks_r)]
        verbos_s = [(i, _signo(t)) for i, t in enumerate(toks_s)]
        verbos_r = [(i, s) for i, s in verbos_r if s != 0]
        verbos_s = [(i, s) for i, s in verbos_s if s != 0]
        for i, sr in verbos_r:
            for j, ss in verbos_s:
                try:
                    if sr * ss != -1:
                        continue
                    obj_r = _objeto(toks_r, i)
                    obj_s = _objeto(toks_s, j)
                    if _jaccard_pond(obj_r, obj_s, pesos) >= UMBRAL_JACCARD:
                        return 1
                except (TypeError, ValueError):
                    continue
        return 0
    except (TypeError, ValueError, AttributeError):
        return 0


def feats_marc(caso):
    """(f_caus, f_alc, f_hedge, flip) de un caso (B.6 + B.12).

    Lee solo respuesta/fuentes/senales (sin etiqueta ni tipo).
    Caso ilegible -> (0, 0, 0, 0). Nunca lanza.
    """
    try:
        if not isinstance(caso, dict):
            return (0, 0, 0, 0)
        try:
            respuesta = caso.get("respuesta", "") or ""
        except (TypeError, ValueError, AttributeError):
            respuesta = ""
        texto_cit = _fuente_citada(caso)
        toks_r = _tokens(respuesta)
        toks_s = _tokens(texto_cit)
        try:
            fuentes = caso.get("fuentes", []) or []
            docs = [toks_r] + [_tokens(f) for f in fuentes]
        except (TypeError, ValueError, AttributeError):
            docs = [toks_r, toks_s]
        f_caus = _f_causal(toks_r, toks_s)
        f_alc = _f_alcance(toks_r, toks_s)
        f_hedge = _f_matices(toks_r, toks_s)
        flip = _flip(toks_r, toks_s, docs)
        return (int(f_caus), int(f_alc), int(f_hedge), int(flip))
    except (TypeError, ValueError, AttributeError):
        return (0, 0, 0, 0)
