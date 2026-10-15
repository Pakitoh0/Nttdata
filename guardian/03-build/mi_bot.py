"""Bot del equipo (PASO 1): baseline de solo-reglas sobre `reglas.py`.

Clase `Guardian` con `score(caso) -> float` en [0, 1] (con recorte).
Consume solo pregunta/fuentes/respuesta/senales, como en
`generar_predicciones._caso_para_bot` (sin etiqueta ni tipo).
"""

import os

try:
    from reglas import FALLBACK, puntuar_caso
except ImportError:  # carga directa por ruta (p. ej. desde otro cwd)
    import importlib.util

    _ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reglas.py")
    _spec = importlib.util.spec_from_file_location("reglas", _ruta)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FALLBACK = _mod.FALLBACK
    puntuar_caso = _mod.puntuar_caso


class Guardian:
    """Baseline de solo-reglas (FUNDAMENTO B.2-B.4, B.9-B.10)."""

    def score(self, caso):
        try:
            valor = float(puntuar_caso(caso))
        except (TypeError, ValueError, AttributeError):
            return FALLBACK
        return min(1.0, max(0.0, valor))
