"""Barrera contra fugas del test al entrenamiento (RNF-04).

Las funciones de carga separan conjuntos con etiqueta (train/val) de
conjuntos sin etiqueta (test) y fallan si se mezclan:

- `cargar_train(ruta)`: exige `etiqueta` 0/1 en TODOS los casos.
- `cargar_test(ruta)`: prohibe `etiqueta` en TODOS los casos.
- `preparar_entrenamiento(rutas)`: solo acepta ficheros de train
  validados; rechaza por nombre (`test_publico`, `giro`) y por
  contenido (cualquier caso sin `etiqueta`).

Criterio por-caso (de `material/kit/EJEMPLOS.md`, sin categorias
nuevas): fiel (0) solo si TODA afirmacion esta respaldada por las
fuentes en conjunto; basta una sin respaldo para alucinacion (1),
tipos: distorsion, contradiccion, no_respaldado, sobregeneralizacion.
"""
import json
import os

NOMBRES_PROHIBIDOS = ("test_publico", "giro")


class VetoTestError(ValueError):
    """Se intento mezclar conjuntos de test con entrenamiento."""


def _leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cargar_train(ruta):
    """Carga un fichero de entrenamiento; falla si algun caso no trae etiqueta."""
    casos = _leer(ruta)
    sin = [c.get("id", "?") for c in casos if c.get("etiqueta") not in (0, 1)]
    if sin:
        raise VetoTestError(
            f"{ruta}: {len(sin)} casos sin etiqueta valida (p. ej. {sin[0]}); "
            "es un fichero de test y no puede entrenar"
        )
    return casos


def cargar_test(ruta):
    """Carga un fichero de test; falla si algun caso trae etiqueta."""
    casos = _leer(ruta)
    con = [c.get("id", "?") for c in casos if "etiqueta" in c]
    if con:
        raise VetoTestError(
            f"{ruta}: {len(con)} casos con etiqueta (p. ej. {con[0]}); "
            "mezcla train en el test"
        )
    return casos


def preparar_entrenamiento(rutas):
    """Valida y concatena ficheros de entrenamiento; veta el test por nombre y contenido."""
    for ruta in rutas:
        base = os.path.basename(str(ruta)).lower()
        if any(n in base for n in NOMBRES_PROHIBIDOS):
            raise VetoTestError(
                f"{ruta}: nombre de fichero de test/giro, prohibido entrenar con el"
            )
    casos = []
    for ruta in rutas:
        casos.extend(cargar_train(ruta))
    return casos
