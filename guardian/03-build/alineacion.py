"""Alineacion afirmacion <-> evidencia (PASO 3). Solo libreria estandar.

ESTADO: pendiente de gate (ver 04-test/CRITERIO_GATE.md). Modulo
aislado: expone features via `feats_alin` para la futura regresion
logistica; NO cableado a mano en `reglas.py` (B.5: la logistica
corrige la sobreconfianza). Entra solo si supera el gate en
ablacion; si no, se quita en Simplify.

FUNDAMENTO B.10, con las mismas cantidades que `reglas.py` (nombres
exactos verificados en `reglas.py` y en `datos/train.jsonl`):
por oracion, `respaldo_por_fuente` (lista por fuente) y
`respaldo_todas` (todas las fuentes juntas). Aproximacion con las
senales disponibles:

    A_ij = max(respaldo_fuente_j, respaldo_todas)

m_i = max_j A_ij (mejor respaldo del trozo i); el argmax
j*(i) = argmax_j A_ij se resuelve por trozo (en la practica basta
frente al hungaro, B.10). Features del caso:

    cobertura   = (1/n) * suma_i 1[m_i >= TAU]
    dispersion  = n. de fuentes distintas usadas como argmax
    margen_i    = A_i(1) - A_i(2) (mejor menos segunda mejor)
    margen_min  = min_i margen_i
    f_dos_peores = media de los dos peores m_i (con n == 1, f = m_1)

TAU = 0.5: punto medio de la escala 0-1 de respaldo y corte
operativo del resto del sistema; NO optimizado sobre train para
evitar sobreajuste con umbrales finos (B.18). Si una sola fuente
respalda el trozo (una unica A_ij), no hay segunda mejor: se toma
A_i(2) = 0, es decir, margen_i = A_i(1) (distancia a "nada
respalda"). Empates en el argmax se resuelven al primer indice
(margen 0 los delata igual).

Ausencia de evidencia = sin respaldo: claves ausentes o ilegibles
usan 0.0 (igual que `validacion.extraer_features`), NaN e
infinitos -> 0.0, y todo acceso va con .get()/try-except. Caso sin
oraciones o ilegible -> (0.0, 0, 0.0, 0.0).
"""

import math

# Umbral de cobertura (ver docstring: 0.5 sin optimizar, B.18).
TAU = 0.5


def _a_cero_uno(x):
    """Float recortado a [0, 1]; NaN/inf/ilegible -> 0.0. Nunca lanza."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    try:
        if not math.isfinite(v):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def alineaciones_oracion(oracion):
    """Lista A_j = max(fuente_j, todas) de una oracion. Nunca lanza."""
    try:
        datos = oracion if isinstance(oracion, dict) else {}
        try:
            todas = _a_cero_uno(datos.get("respaldo_todas", 0.0))
        except (TypeError, ValueError, AttributeError):
            todas = 0.0
        try:
            brutas = datos.get("respaldo_por_fuente", []) or []
            if isinstance(brutas, (str, bytes)):
                brutas = []
            valores = [_a_cero_uno(v) for v in list(brutas)]
        except (TypeError, ValueError, AttributeError):
            valores = []
        if not valores:
            return [todas]
        return [v if v >= todas else todas for v in valores]
    except (TypeError, ValueError, AttributeError):
        return [0.0]


def feats_alin(caso):
    """(cobertura, dispersion, margen_min, f_dos_peores) de un caso.

    Solo lee `senales.oraciones` (nunca etiqueta ni tipo). Nunca lanza.
    """
    try:
        if not isinstance(caso, dict):
            return 0.0, 0, 0.0, 0.0
        try:
            senales = caso.get("senales", {}) or {}
            if not isinstance(senales, dict):
                return 0.0, 0, 0.0, 0.0
            oraciones = senales.get("oraciones", []) or []
            if not isinstance(oraciones, list):
                return 0.0, 0, 0.0, 0.0
        except (TypeError, ValueError, AttributeError):
            return 0.0, 0, 0.0, 0.0
        if not oraciones:
            return 0.0, 0, 0.0, 0.0
        mejores = []
        segundos = []
        usados = []
        for oracion in oraciones:
            try:
                vals = alineaciones_oracion(oracion)
            except (TypeError, ValueError, AttributeError):
                vals = [0.0]
            if not vals:
                vals = [0.0]
            try:
                mejor = max(vals)
            except (TypeError, ValueError):
                mejor = 0.0
            try:
                orden = sorted(vals, reverse=True)
                segunda = orden[1] if len(orden) >= 2 else 0.0
            except (TypeError, ValueError, IndexError):
                segunda = 0.0
            try:
                arg = vals.index(mejor)
            except (TypeError, ValueError):
                arg = 0
            mejores.append(mejor)
            segundos.append(segunda)
            usados.append(arg)
        try:
            n = len(mejores)
            cobertura = sum(1 for m in mejores if m >= TAU) / float(n)
            if not math.isfinite(cobertura):
                cobertura = 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            cobertura = 0.0
        try:
            dispersion = len(set(usados))
        except (TypeError, ValueError):
            dispersion = 0
        try:
            margenes = [max(0.0, b - s) for b, s in zip(mejores, segundos)]
            margen_min = min(margenes) if margenes else 0.0
            if not math.isfinite(margen_min):
                margen_min = 0.0
        except (TypeError, ValueError):
            margen_min = 0.0
        try:
            ordenadas = sorted(mejores)
            if len(ordenadas) >= 2:
                f_dos = (ordenadas[0] + ordenadas[1]) / 2.0
            else:
                f_dos = ordenadas[0]
            if not math.isfinite(f_dos):
                f_dos = 0.0
        except (TypeError, ValueError, IndexError):
            f_dos = 0.0
        return cobertura, dispersion, margen_min, f_dos
    except (TypeError, ValueError, AttributeError):
        return 0.0, 0, 0.0, 0.0


def puntuar_auxiliar(caso):
    """Score auxiliar solo-medicion: 1 - f_dos_peores en [0, 1].

    f_dos_peores es respaldo (alto = fiel); el score de alucinacion
    lo invierte. No es baseline ni entra a mano en `reglas.py`: sirve
    para medir el poder predictivo del grupo con
    `validacion.comparar_scores` antes de la ablacion logistica.
    Nunca lanza.
    """
    try:
        _, _, _, f_dos = feats_alin(caso)
        return min(1.0, max(0.0, 1.0 - float(f_dos)))
    except (TypeError, ValueError, AttributeError):
        return 0.5
