"""Baseline de solo-reglas (PASO 1). Solo libreria estandar, sin IA ni sklearn.

Combina por oracion la senal NLI conjunta (`respaldo_todas`,
`contradiccion_todas`) con el veredicto del juez LLM, todo en escala
logit con recorte epsilon = 1e-4 (FUNDAMENTO B.4). Agrega a nivel de
respuesta con maximo, noisy-OR y media de los dos peores mas n (B.3,
B.10). Accesos defensivos con .get() y try/except con fallback 0.55
(ligeramente sobre 0.5 por el coste asimetrico 4:1, B.9).

Baseline PASO 1 puro: sin penalizacion de cifras a mano. El grupo
cifras queda AISLADO en `cifras.py::feats_num` (f_num, f_deriv,
f_year, estado CONDICIONADO) para la futura regresion logistica;
entrara solo si supera el gate de 04-test/CRITERIO_GATE.md.
"""

import math

# Recorte para el logit (FUNDAMENTO B.4): evita log(0).
EPS = 1e-4
# Fallback ante caso ilegible: algo por encima de 0.5 (B.9, coste 4:1).
FALLBACK = 0.55

# Veredictos que afirman "no respaldada" (prob = confianza kappa).
_VEREDICTO_MALO = ("no_respaldada", "contradicha")


def _recortar(p):
    """Recorta una probabilidad a [EPS, 1 - EPS]."""
    try:
        return min(1.0 - EPS, max(EPS, float(p)))
    except (TypeError, ValueError):
        return 0.5


def logit(p):
    """Logit con recorte epsilon (B.4)."""
    q = _recortar(p)
    return math.log(q / (1.0 - q))


def sigmoide(z):
    """Sigmoide numericamente estable."""
    try:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)
    except (TypeError, ValueError, OverflowError):
        return 0.5


def prob_juez_llm(veredicto, confianza):
    """Probabilidad de "no respaldada" segun el juez LLM (B.4).

    - veredicto malo (no_respaldada/contradicha) -> kappa.
    - veredicto "respaldada" -> 1 - kappa.
    - veredicto null o confianza ilegible -> 0.5 (neutral).
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


def prob_no_respaldo_oracion(oracion):
    """Probabilidad de que una oracion no este respaldada (q_i).

    Senal NLI: q_nli = max(1 - respaldo_todas, contradiccion_todas).
    Se usa el maximo (no el promedio) porque una contradiccion baja con
    respaldo bajo no es evidencia de fidelidad, sino de falta de
    evidencia. Combinacion con el juez en logit con promedio simple
    (media geometrica de apuestas): evita sobrecontar dos evidencias
    correlacionadas (B.5). Ante dato ausente, cada senal aporta 0.5.
    """
    try:
        datos = oracion if isinstance(oracion, dict) else {}
        try:
            respaldo = float(datos.get("respaldo_todas", 0.5))
        except (TypeError, ValueError):
            respaldo = 0.5
        try:
            contradiccion = float(datos.get("contradiccion_todas", 0.5))
        except (TypeError, ValueError):
            contradiccion = 0.5
        llm = datos.get("llm", {}) or {}
        if not isinstance(llm, dict):
            llm = {}
        q_juez = prob_juez_llm(llm.get("veredicto"), llm.get("confianza"))
        q_nli = max(1.0 - respaldo, contradiccion)
        q = sigmoide((logit(q_nli) + logit(q_juez)) / 2.0)
        return min(1.0, max(0.0, q))
    except (TypeError, ValueError, AttributeError):
        return 0.5


def agregar_respuesta(qs):
    """Agrega las q_i de las oraciones a un score de respuesta (B.3, B.10).

    Calcula tres vistas: maximo M (una sola falsa basta, REQ-04),
    noisy-OR P = 1 - prod(1 - q_i) y media F de los dos peores
    (mas estable que el minimo solo, B.10; con n == 1, F = M).
    El tamano n entra por P (crece con n) y por la definicion de F.
    Score final: promedio simple (M + P + F) / 3, que conserva el
    maximo pero suaviza el sesgo de longitud del noisy-OR puro.
    Incluye el reescalado B.2 s = 4p / (1 + 3p), que lleva el corte
    optimo del coste 4:1 (p > 0.2) al corte operativo 0.5.
    """
    try:
        valores = []
        for q in qs or []:
            try:
                valores.append(min(1.0, max(0.0, float(q))))
            except (TypeError, ValueError):
                continue
        if not valores:
            return FALLBACK
        peor = max(valores)
        resto = 1.0
        for q in valores:
            resto *= 1.0 - q
        ruidoso = 1.0 - resto
        ordenadas = sorted(valores, reverse=True)
        if len(ordenadas) >= 2:
            dos_peores = (ordenadas[0] + ordenadas[1]) / 2.0
        else:
            dos_peores = ordenadas[0]
        combinado = (peor + ruidoso + dos_peores) / 3.0
        # Reescalado B.2 (una sola via, sin peso de entrenamiento).
        reescalado = 4.0 * combinado / (1.0 + 3.0 * combinado)
        return min(1.0, max(0.0, reescalado))
    except (TypeError, ValueError, AttributeError):
        return FALLBACK


def puntuar_caso(caso):
    """Score [0, 1] de un caso (probabilidad de alucinacion).

    Lee solo pregunta/fuentes/respuesta/senales (sin etiqueta ni tipo).
    Caso ilegible o sin oraciones -> FALLBACK (0.55, B.9).
    Baseline PASO 1 puro: sin termino de cifras (aisladas en
    `cifras.feats_num` para la futura logistica, ver CRITERIO_GATE.md).
    """
    try:
        if not isinstance(caso, dict):
            return FALLBACK
        senales = caso.get("senales", {}) or {}
        if not isinstance(senales, dict):
            return FALLBACK
        oraciones = senales.get("oraciones", []) or []
        if not oraciones:
            return FALLBACK
        qs = [prob_no_respaldo_oracion(o) for o in oraciones]
        return agregar_respuesta(qs)
    except (TypeError, ValueError, AttributeError):
        return FALLBACK
