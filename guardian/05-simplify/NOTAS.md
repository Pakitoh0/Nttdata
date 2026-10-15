# NOTAS — Simplify del modelo final (PASO 7 → ship)

Fecha: 2026-10-15. Config final fijada en `03-build/modelo.py`
(`COLUMNAS_FINAL`) y medida en `04-test/resultados_paso7.md` (OOF en H:
puntuación −0,308 · AUROC 0,749 · FN 69 · FP 161; reglas −0,449 / 0,735).

## Camino final (lo que se embarca en `06-ship/mi_bot.py`)

SOLO baseline reglas + features de alineación, combinados por
regresión logística L2 (C = 1,0 por CV interna GroupKFold ×3 sobre el
entreno) con reescalado B.2 `s = 4p/(1+3p)` (corte operativo 0,5).
5 columnas: `cobertura, dispersion, margen_min, f_dos_peores`
(alineación) + `score_base` (reglas). Entrenado sobre TODO train no-H
(1076 casos; ningún H entrena, política REQ-09/`excluir_h_entreno`):
H queda 100 % held-out y la métrica del modelo embarcado en H
(−0,299 / 0,751 / FN 68 / FP 162) coincide con el OOF del paso 7
dentro de ruido. Métrica honesta del modelo embarcado = OOF paso 7.

## Queda FUERA (módulos conservados como historial, no borrados)

- cifras (`03-build/cifras.py`): ΔAUROC −0,000 / Δpérdida −0,006, no aporta nada (ablación paso 6, IC95 incluye el 0).
- marcadores (`03-build/marcadores.py`): ΔAUROC −0,008 / Δpérdida −0,030, resta (quitarlo mejora ambas).
- atribución (`03-build/atribucion.py`): no entró al gate de ablación; FN residual en cifras/atribución documentado en el pitch.
- flip (`f_alc`/flip de marcadores, `f_deriv` de cifras): excluidos a propósito desde el paso 6 (signo inestable con ~600 H).
- Platt (`a = 0,3785, b = −0,5538`): baja ECE (0,2254 → 0,0944) pero a corte fijo 0,5 da −0,566 (FN 105, FP 103); adoptarla exigiría re-derivar el umbral por pérdida esperada (B.16 t*), fuera de este paso.
- Peso de clase 4:1 en entreno: una sola vía de coste (B.2); el reescalado posterior es monótono y auditable, el peso escondería el desplazamiento en los coeficientes.

## Qué se conservó intacto y por qué

`COLUMNAS`/`VARIANTES`/`oof_ablacion` del paso 6 intactos como
auditoría; `reglas.py` y `alineacion.py` intactos como referencia del
cálculo que `mi_bot.py` replica autocontenido con stdlib.
