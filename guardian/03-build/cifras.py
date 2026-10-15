"""Cifras con cierre por derivacion (PASO 2). Solo libreria estandar.

ESTADO: CONDICIONADO (conservado pero NO aprobado). Modulo aislado:
expone features via `feats_num`/`features_cifras` para la futura
regresion logistica; NO cableado a mano en `reglas.py` (ver
04-test/CRITERIO_GATE.md). Entrara solo si supera el gate en ablacion;
si no, se quita en Simplify.

FUNDAMENTO B.6 (cifras) y B.13 (redondeo declarado):

- Extrae numeros de textos espanoles: digitos con separadores
  ("28.000", "2.000", "3,50"), porcentajes ("40 %", "por ciento"),
  palabras ("tres", "catorce", "veintiocho") y magnitudes
  ("mil", "millon/millones").
- Normaliza a valor+unidad: "2.000 m" -> (2.0, "km"), "40 %" -> (0.4, "%").
- Anos separados del cierre aritmetico: tokens \\b(19|20)\\d{2}\\b van
  SOLO por coincidencia exacta contra las fuentes (sin derivacion ni
  tolerancia); N_S/N_R del cierre D(N_S) los excluyen; la senal de ano
  es f_year = 1[existe ano en R que no esta en S].
- Cierre por derivacion: D(N_S) = N_S u {a+b, |a-b|, a*b, a/b,
  100*a/b, 2a, a/2 : a,b en N_S}, mas sumas de subconjuntos hasta
  tamano 3 (itertools.combinations, |N_S| <= 12).
  100*a/b SOLO se consulta si la respuesta lleva "%" o "por ciento"
  (la unidad "%" ya lo impone; `features_cifras` lo exige ademas por
  bandera para que la intencion quede blindada).
- Features: f_num = |U| / max(1, |N_R|), f_deriv = n. de cifras de la
  respuesta que solo se explican por derivacion, f_year como arriba.
  Sin cifras (no-ano) -> f_num = f_deriv = 0 (f_year va aparte).
- f_deriv se mantiene disponible aunque las fieles derivan MAS
  (media 0.43 fieles frente a 0.22 alucinadas; ver
  04-test/resultados_paso2.md): el signo lo debe aprender el
  clasificador (evidencia negativa o interaccion z4 de B.15), nunca
  una penalizacion a mano que sumaria donde deberia restar.
- Compatibilidad con redondeo: |x - g| <= delta(x), con
  delta(x) = 0.5 * 10**(-d). En cifras redondas (sin decimales) se usa
  el ultimo digito significativo no cero (28000 -> delta 500).
  Con redondeo declarado ("unos", "casi", ...) se admite ademas una
  tolerancia relativa del 5 %.

Todo acceso defensivo (.get()/try-except); NaN e infinitos -> 0.
"""

import itertools
import math
import re

# Maximo de cifras de fuente (B.13: coste O(C(n,3))).
MAX_FUENTE = 12
# Tolerancia relativa solo con redondeo declarado (B.6).
TOL_REDONDEO = 0.05
# Epsilon numerico para porcentajes (B.13: delta(x) + eps).
EPS_PORC = 1e-9

_NUM_PALABRAS = {
    "cero": 0.0, "uno": 1.0, "un": 1.0, "una": 1.0, "dos": 2.0,
    "tres": 3.0, "cuatro": 4.0, "cinco": 5.0, "seis": 6.0,
    "siete": 7.0, "ocho": 8.0, "nueve": 9.0, "diez": 10.0,
    "once": 11.0, "doce": 12.0, "trece": 13.0, "catorce": 14.0,
    "quince": 15.0, "dieciseis": 16.0, "dieciséis": 16.0,
    "diecisiete": 17.0, "dieciocho": 18.0, "diecinueve": 19.0,
    "veinte": 20.0, "veintiuno": 21.0, "veintiun": 21.0,
    "veintidos": 22.0, "veintidós": 22.0, "veintitres": 23.0,
    "veintitrés": 23.0, "veinticuatro": 24.0, "veinticinco": 25.0,
    "veintiseis": 26.0, "veintiséis": 26.0, "veintisiete": 27.0,
    "veintiocho": 28.0, "veintinueve": 29.0, "treinta": 30.0,
    "cuarenta": 40.0, "cincuenta": 50.0, "sesenta": 60.0,
    "setenta": 70.0, "ochenta": 80.0, "noventa": 90.0,
    "cien": 100.0, "ciento": 100.0, "doscientos": 200.0,
    "trescientos": 300.0, "cuatrocientos": 400.0,
    "quinientos": 500.0, "seiscientos": 600.0,
    "setecientos": 700.0, "ochocientos": 800.0,
    "novecientos": 900.0,
}

_MAGNITUDES = {
    "mil": 1000.0, "millar": 1000.0, "millares": 1000.0,
    "millon": 1e6, "millón": 1e6, "millones": 1e6,
}

_MARCADORES_REDONDEO = (
    "unos", "unas", "casi", "alrededor", "aproximadamente",
    "cerca de", "en torno a", "algo mas de", "algo más de",
    "mas o menos", "más o menos", "unos/as",
)

_RE_DIGITOS = re.compile(r"\d[\d\.,]*")
_RE_ANO = re.compile(r"\b(19|20)\d{2}\b")
_RE_ANO_EXACTO = re.compile(r"(19|20)\d{2}")
_RE_PALABRA = re.compile(
    r"[a-záéíóúüñ]+|[€%]|\d[\d\.,]*", re.IGNORECASE
)

_UNIDADES_EURO = {"€", "euro", "euros"}
_UNIDADES_KM = {"km", "kilometro", "kilómetro", "kilometros",
                "kilómetros"}
_UNIDADES_M = {"m", "metro", "metros"}


def _es_finito(x):
    """True si x es numero finito (NaN/inf -> False)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _limpiar_numero(token):
    """Convierte un token de digitos con separadores es a float.

    "28.000" -> 28000.0; "3,50" -> 3.5; "2.000" -> 2000.0.
    Devuelve (valor, decimales) o (None, 0) si no parsea.
    """
    try:
        s = (token or "").strip().rstrip(".,;:")
        if not s or not s[0].isdigit():
            return None, 0
        if "," in s and "." in s:
            # Formato es: punto de miles, coma decimal ("1.234,56").
            enteros = s.split(",")[0].replace(".", "")
            dec = s.split(",")[-1]
            return float(enteros + "." + dec), len(dec)
        if "," in s:
            partes = s.split(",")
            # Coma decimal si la cola tiene 1-2 digitos ("3,50").
            if len(partes) == 2 and 1 <= len(partes[1]) <= 2:
                return float(partes[0].replace(".", "") + "." + partes[1]), \
                    len(partes[1])
            return float(s.replace(",", "")), 0
        if "." in s:
            if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
                return float(s.replace(".", "")), 0
            try:
                val = float(s)
            except ValueError:
                return None, 0
            dec = s.split(".")[-1] if len(s.split(".")) == 2 else ""
            return val, len(dec) if dec.isdigit() else 0
        return float(s), 0
    except (TypeError, ValueError):
        return None, 0


def _unidad_canonica(token):
    """Canoniza un token de unidad a {"€", "km", "m", "%", ""}."""
    try:
        t = (token or "").strip().lower()
    except (TypeError, ValueError, AttributeError):
        return ""
    if t in _UNIDADES_EURO:
        return "€"
    if t in _UNIDADES_KM:
        return "km"
    if t in _UNIDADES_M:
        return "m"
    if t in ("%", "porciento"):
        return "%"
    return ""


def normalizar_unidad(valor, unidad):
    """Normaliza valor+unidad ("2.000 m" -> (2.0, "km")).

    NaN/inf -> (0.0, unidad intacta). Nunca lanza.
    """
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return 0.0, (unidad or "")
    if not _es_finito(v):
        return 0.0, (unidad or "")
    try:
        if unidad == "m" and abs(v) >= 1000.0:
            return v / 1000.0, "km"
        return v, (unidad or "")
    except (TypeError, ValueError):
        return 0.0, (unidad or "")


def delta_para(valor, decimales):
    """Tolerancia delta(x) = 0.5 * 10**(-d) (B.13).

    Sin decimales, d se toma del ultimo digito significativo no cero
    (28000 -> 500; 86 -> 0.5). Nunca lanza; NaN -> 0.
    """
    try:
        d = int(decimales)
    except (TypeError, ValueError):
        d = 0
    try:
        if d > 0:
            return 0.5 * (10.0 ** (-d))
        v = abs(float(valor))
        if not _es_finito(v):
            return 0.0
        entero = int(round(v))
        if entero <= 0:
            return 0.5
        ceros = 0
        while entero % 10 == 0 and entero >= 10:
            entero //= 10
            ceros += 1
        return 0.5 * (10.0 ** ceros)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def detectar_redondeo(texto):
    """True si el texto declara redondeo ("unos", "casi", ...)."""
    try:
        bajo = str(texto).casefold()
    except (TypeError, ValueError):
        return False
    for marca in _MARCADORES_REDONDEO:
        try:
            if marca in bajo:
                return True
        except (TypeError, ValueError):
            continue
    return False


def extraer_anos(texto):
    """Anos de un texto: tokens \\b(19|20)\\d{2}\\b -> set de str.

    Solo digitos de 4 cifras (1900-2099); los anos en palabras no
    cuentan como ano. Ante texto ilegible -> set(). Nunca lanza.
    """
    try:
        if texto is None:
            return set()
        s = str(texto)
    except (TypeError, ValueError):
        return set()
    try:
        return {m.group(0) for m in _RE_ANO.finditer(s)}
    except (TypeError, ValueError):
        return set()


def tiene_porcentaje(texto):
    """True si la respuesta lleva "%", "por ciento" o "porciento"."""
    try:
        if texto is None:
            return False
        s = str(texto)
        bajo = s.casefold()
    except (TypeError, ValueError):
        return False
    try:
        return ("%" in s) or ("por ciento" in bajo) or \
            ("porciento" in bajo)
    except (TypeError, ValueError):
        return False


def _es_ano(texto_token):
    """True si el token extraido es un ano (coincidencia exacta)."""
    try:
        t = str(texto_token or "").strip()
    except (TypeError, ValueError, AttributeError):
        return False
    try:
        return _RE_ANO_EXACTO.fullmatch(t) is not None
    except (TypeError, ValueError):
        return False


def compatible(x_valor, x_decimales, g_valor, redondeo=False):
    """x compatible con g <=> |x - g| <= delta(x) (B.13).

    Con redondeo declarado admite ademas 5 % relativo.
    """
    try:
        x = float(x_valor)
        g = float(g_valor)
    except (TypeError, ValueError):
        return False
    if not _es_finito(x) or not _es_finito(g):
        return False
    try:
        tol = delta_para(x, x_decimales) + EPS_PORC
        if abs(x - g) <= tol:
            return True
        if redondeo and _es_finito(g) and g != 0.0:
            return abs(x - g) <= max(tol, TOL_REDONDEO * abs(g))
        return False
    except (TypeError, ValueError):
        return False


def extraer_cifras(texto):
    """Extrae cifras de un texto espanol -> lista de dicts.

    Cada cifra: {"valor": float, "unidad": str, "decimales": int,
    "texto": str}. Unidades canonicas {"€","km","m","%",""}.
    "40 %" y "40 por ciento" -> (0.4, "%"). "2.000 m" -> (2.0, "km").
    "veintiocho mil" -> (28000.0, ""). Ante texto ilegible -> [].
    """
    cifras = []
    try:
        if texto is None:
            return []
        original = str(texto)
        bajo = original.casefold()
    except (TypeError, ValueError):
        return []
    try:
        tokens = _RE_PALABRA.findall(bajo)
    except (TypeError, ValueError):
        return []
    i = 0
    n = len(tokens)
    while i < n:
        try:
            tok = tokens[i]
            # Caso 1: digitos con posible magnitud/unidad/porcentaje.
            if tok and tok[0].isdigit():
                valor, dec = _limpiar_numero(tok)
                if valor is None:
                    i += 1
                    continue
                resto = tokens[i + 1:i + 4]
                mag = 1.0
                j = i + 1
                if resto and resto[0] in _MAGNITUDES:
                    mag = _MAGNITUDES[resto[0]]
                    j = i + 2
                    resto = tokens[j:j + 2]
                valor *= mag
                unidad = ""
                decimales = dec
                cola = tokens[j:j + 3]
                if cola and cola[0] in ("%", "porciento"):
                    valor /= 100.0
                    unidad = "%"
                    decimales = dec
                    j += 1
                elif len(cola) >= 2 and cola[0] == "por" and \
                        cola[1] == "ciento":
                    valor /= 100.0
                    unidad = "%"
                    j += 2
                elif cola and _unidad_canonica(cola[0]):
                    unidad = _unidad_canonica(cola[0])
                    j += 1
                valor, unidad = normalizar_unidad(valor, unidad)
                if not _es_finito(valor):
                    valor = 0.0
                cifras.append({"valor": valor, "unidad": unidad,
                               "decimales": decimales, "texto": tok})
                i = j
                continue
            # Caso 2: palabras-numero con posible magnitud.
            if tok in _NUM_PALABRAS:
                suma = 0.0
                j = i
                while j < n and (
                        tokens[j] in _NUM_PALABRAS or tokens[j] == "y"):
                    if tokens[j] != "y":
                        suma += _NUM_PALABRAS[tokens[j]]
                    j += 1
                mag = 1.0
                if j < n and tokens[j] in _MAGNITUDES:
                    mag = _MAGNITUDES[tokens[j]]
                    j += 1
                valor = suma * mag
                unidad = ""
                cola = tokens[j:j + 3]
                if cola and cola[0] in ("%", "porciento"):
                    valor /= 100.0
                    unidad = "%"
                    j += 1
                elif len(cola) >= 2 and cola[0] == "por" and \
                        cola[1] == "ciento":
                    valor /= 100.0
                    unidad = "%"
                    j += 2
                elif cola and _unidad_canonica(cola[0]):
                    unidad = _unidad_canonica(cola[0])
                    j += 1
                valor, unidad = normalizar_unidad(valor, unidad)
                if not _es_finito(valor):
                    valor = 0.0
                cifras.append({"valor": valor, "unidad": unidad,
                               "decimales": 0, "texto": tok})
                i = j
                continue
            i += 1
        except (TypeError, ValueError, IndexError):
            try:
                i += 1
            except (TypeError, ValueError):
                break
            continue
    return cifras


def _derivadas_grupo(valores):
    """Cierre D(V) de un grupo de valores con la misma unidad.

    D(V) = V u {a+b, |a-b|, a*b, a/b, 2a, a/2} mas sumas de
    subconjuntos hasta tamano 3. Devuelve (derivadas, porcentajes),
    donde porcentajes son los 100*a/b (unidad "%"). Los porcentajes
    se consumen SOLO si la respuesta lleva "%"/"por ciento" (lo exige
    `features_cifras` por bandera; la unidad "%" ya lo impone).
    """
    derivadas = set()
    porcentajes = set()
    try:
        vals = [float(v) for v in (valores or [])]
        vals = [v for v in vals if _es_finito(v)][:MAX_FUENTE]
    except (TypeError, ValueError):
        return set(), set()
    for v in vals:
        try:
            derivadas.add(v)
            if _es_finito(2.0 * v):
                derivadas.add(2.0 * v)
            if _es_finito(v / 2.0):
                derivadas.add(v / 2.0)
        except (TypeError, ValueError, OverflowError):
            continue
    try:
        for a, b in itertools.combinations_with_replacement(vals, 2):
            try:
                if _es_finito(a + b):
                    derivadas.add(a + b)
                if _es_finito(abs(a - b)):
                    derivadas.add(abs(a - b))
                if _es_finito(a * b):
                    derivadas.add(a * b)
                if b != 0.0 and _es_finito(a / b):
                    derivadas.add(a / b)
                if a != 0.0 and _es_finito(b / a):
                    derivadas.add(b / a)
                # 100*a/b y 100*b/a explican porcentajes (unidad "%").
                if b != 0.0 and _es_finito(100.0 * a / b):
                    porcentajes.add(100.0 * a / b / 100.0)
                if a != 0.0 and _es_finito(100.0 * b / a):
                    porcentajes.add(100.0 * b / a / 100.0)
            except (TypeError, ValueError, OverflowError,
                    ZeroDivisionError):
                continue
        # Sumas de subconjuntos hasta tamano 3 (tamano 1 ya es V).
        for tam in (2, 3):
            try:
                for combo in itertools.combinations(vals, tam):
                    try:
                        s = sum(combo)
                        if _es_finito(s):
                            derivadas.add(s)
                    except (TypeError, ValueError, OverflowError):
                        continue
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        pass
    return derivadas, porcentajes


def _agrupar_por_unidad(cifras):
    """Agrupa valores por unidad canonica (defensivo)."""
    grupos = {}
    try:
        for c in cifras or []:
            try:
                if not isinstance(c, dict):
                    continue
                v = float(c.get("valor", 0.0))
                if not _es_finito(v):
                    v = 0.0
                u = str(c.get("unidad", "") or "")
                grupos.setdefault(u, []).append(v)
            except (TypeError, ValueError, AttributeError):
                continue
    except (TypeError, ValueError, AttributeError):
        return {}
    for u in list(grupos):
        grupos[u] = grupos[u][:MAX_FUENTE]
    return grupos


def explicar_cifra(resp, derivados_por_unidad, porc_derivados,
                   grupo_fuente, redondeo):
    """Clasifica una cifra de respuesta: "directa", "derivada" o None.

    "directa": compatible con un valor fuente de la misma unidad.
    "derivada": compatible solo via D(N_S) (B.6) o subconjuntos (B.13).
    """
    try:
        x = float(resp.get("valor", 0.0))
        u = str(resp.get("unidad", "") or "")
        dec = int(resp.get("decimales", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return None
    if not _es_finito(x):
        return None
    try:
        for g in grupo_fuente.get(u, []):
            if compatible(x, dec, g, redondeo):
                return "directa"
        if u == "%":
            for g in porc_derivados:
                if compatible(x, dec, g, redondeo):
                    return "derivada"
        for g in derivados_por_unidad.get(u, set()):
            if compatible(x, dec, g, redondeo):
                # Si tambien era directa ya se devolvio arriba.
                return "derivada"
        return None
    except (TypeError, ValueError, AttributeError):
        return None


def features_cifras(respuesta, fuentes):
    """f_num, f_deriv y f_year de una respuesta frente a sus fuentes (B.6).

    Anos (\\b(19|20)\\d{2}\\b) excluidos de N_S/N_R del cierre: solo
    cuentan por coincidencia exacta (f_year = 1.0 si existe un ano en
    la respuesta ausente en las fuentes, 0.0 en otro caso). 100*a/b
    solo se consulta si la respuesta lleva "%" o "por ciento".
    f_deriv se expone para que el clasificador aprenda su signo (las
    fieles derivan mas; ver docstring del modulo), nunca como castigo
    a mano. Sin cifras no-ano en la respuesta -> (0.0, 0.0, f_year).
    NaN/inf -> 0.0. Nunca lanza.
    """
    try:
        cifras_r_all = extraer_cifras(respuesta)
    except (TypeError, ValueError):
        cifras_r_all = []
    try:
        if isinstance(fuentes, str):
            textos = [fuentes]
        else:
            textos = list(fuentes or [])
    except (TypeError, ValueError):
        textos = []
    cifras_s_all = []
    try:
        for t in textos:
            try:
                cifras_s_all.extend(extraer_cifras(t))
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        cifras_s_all = []
    # Anos: coincidencia exacta, fuera del cierre.
    try:
        anos_r = extraer_anos(respuesta)
    except (TypeError, ValueError):
        anos_r = set()
    try:
        anos_s = set()
        for t in textos:
            try:
                anos_s |= extraer_anos(t)
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        anos_s = set()
    try:
        f_year = 1.0 if (anos_r - anos_s) else 0.0
    except (TypeError, ValueError, AttributeError):
        f_year = 0.0
    try:
        cifras_r = [c for c in cifras_r_all
                    if not _es_ano((c or {}).get("texto", ""))]
        cifras_s = [c for c in cifras_s_all
                    if not _es_ano((c or {}).get("texto", ""))]
    except (TypeError, ValueError, AttributeError):
        cifras_r, cifras_s = [], []
    if not cifras_r:
        return 0.0, 0.0, f_year
    try:
        tiene_pct = tiene_porcentaje(respuesta)
    except (TypeError, ValueError):
        tiene_pct = False
    try:
        redondeo = detectar_redondeo(respuesta)
    except (TypeError, ValueError):
        redondeo = False
    try:
        grupos = _agrupar_por_unidad(cifras_s)
        derivados = {}
        porc = set()
        for u, vals in grupos.items():
            try:
                der, por = _derivadas_grupo(vals)
                derivados[u] = der
                if u != "%":
                    porc |= por
                else:
                    derivados["%"] = derivados.get("%", set()) | por
            except (TypeError, ValueError):
                derivados[u] = set()
        if not tiene_pct:
            porc = set()
        sin_respaldo = 0
        solo_deriv = 0
        for r in cifras_r:
            try:
                estado = explicar_cifra(r, derivados, porc, grupos,
                                       redondeo)
                if estado is None:
                    sin_respaldo += 1
                elif estado == "derivada":
                    solo_deriv += 1
            except (TypeError, ValueError, AttributeError):
                sin_respaldo += 1
        n = max(1, len(cifras_r))
        f_num = float(sin_respaldo) / float(n)
        f_deriv = float(solo_deriv)
        if not _es_finito(f_num):
            f_num = 0.0
        if not _es_finito(f_deriv):
            f_deriv = 0.0
        if not _es_finito(f_year):
            f_year = 0.0
        return max(0.0, min(1.0, f_num)), max(0.0, f_deriv), \
            (1.0 if f_year else 0.0)
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return 0.0, 0.0, 0.0


def feats_num(respuesta, fuentes):
    """Punto de entrada aislado del grupo cifras (CONDICIONADO).

    Devuelve (f_num, f_deriv, f_year); ver `features_cifras`.
    Reservado para la futura regresion logistica (ablacion con gate);
    NO usar a mano en reglas (B.5: la logistica corrige la
    sobreconfianza; el signo de f_deriv lo aprende ella).
    Nunca lanza.
    """
    try:
        return features_cifras(respuesta, fuentes)
    except (TypeError, ValueError, AttributeError):
        return 0.0, 0.0, 0.0
