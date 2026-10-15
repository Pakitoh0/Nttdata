# 04 — Test

Evaluación y análisis de errores por versión del bot.
Guarda los resultados de `evaluar_local.py` (global y `--prefijo H`),
la calibración del corte 0.5 y el análisis de falsos negativos y falsos positivos.
Ninguna versión avanza a simplificación sin su registro aquí.

## Suites (stdlib `unittest`, `python3 test_*.py`)

- `test_metrica.py` (12 tests): fórmula exacta, borde `n_fieles == 0` y
  desempate AUROC. Detalle en `METRICA.md` (REQ-08).
- `test_validacion.py` (8 tests): GroupKFold por hash de fuentes,
  disjunción por fold, política sin-H-en-entreno y métricas reales en H
  de `datos/` (REQ-09).
- `test_veto.py` (8 tests): `cargar_train` / `cargar_test` /
  `preparar_entrenamiento` rechazan mezclas con `VetoTestError` (RNF-04).
- `test_reglas.py` (6 tests): baseline de solo-reglas (juez null → 0.5,
  agregación max una-falsa-basta, clipping [0,1], fallback 0.55).
  Resultados del PASO 1 en `resultados_paso1.md`.
