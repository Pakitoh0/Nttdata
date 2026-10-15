# METRICA — Métrica oficial y desempate (REQ-08)

## Fórmula

`puntuacion = 1 - (4·FN + FP) / n_fieles`, con corte `score >= 0.5`.

- `FN`: casos con `etiqueta == 1` (alucinación) predichos como fieles. Pesa 4.
- `FP`: casos con `etiqueta == 0` (fiel) predichos como alucinación. Pesa 1.
- `n_fieles`: casos con `etiqueta == 0`.
- Referencias: `1` = sin fallos; `0` = igual que marcar todo como
  alucinación; negativo = peor que no hacer nada.

Implementación: `puntuacion()` en `03-build/evaluar_local.py`.
Verificada contra la fórmula término a término; no se ha reescrito,
solo se ha documentado y se ha añadido `comparar()`.

## Caso borde `n_fieles == 0` (comportamiento definido)

La fórmula no define este caso (división por cero). El código define:

- Si no hay fieles pero hay alucinaciones: `1 - FN / n_alucinaciones`.
- Si no hay ningún caso: `1.0`.

No se ha corregido porque no discrepa de la fórmula: la cubre donde
esta no dice nada. Cambiarlo alteraría resultados sin base oficial.

## Desempate por AUROC

`auroc()` mide la ordenación de los scores al margen del corte
(Mann-Whitney con 0.5 por empate; 0.5 si solo hay una clase).
`comparar(a, b)` (dicts con `puntuacion` y `auroc`) devuelve
`1` si `a` es mejor, `-1` si `b` es mejor, `0` si empatan:
gana la mayor puntuación y, si empatan, el mayor AUROC.

## Tests

`04-test/test_metrica.py` (stdlib `unittest`, 12 tests con valores
calculados a mano): solo-FP (0.75), solo-FN (-1.0), sin errores (1.0),
todo-alucinación (0.0), `n_fieles == 0` (1 - 2/3), vacío (1.0),
AUROC perfecto/empate/una-clase y los tres casos de `comparar`.
