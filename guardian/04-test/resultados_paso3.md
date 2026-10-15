# Resultados PASO 3 — Alineación afirmación ↔ evidencia (B.10)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (H = 631: 334 fieles, 297 alucinaciones).

## Cambio

- Nuevo `03-build/alineacion.py` (solo stdlib, sin IA): por oración,
  `A_ij = max(respaldo_fuente_j, respaldo_todas)` con los mismos
  nombres que `reglas.py` (`respaldo_por_fuente`, `respaldo_todas`,
  verificados en `reglas.py` y en `datos/train.jsonl`).
  `feats_alin(caso)` devuelve
  (cobertura, dispersión, margen_min, f_dos_peores):
  cobertura = fracción de oraciones con `max_j A_ij >= τ` (τ = 0.5:
  punto medio de la escala 0-1 y corte operativo, NO optimizado sobre
  train para evitar sobreajuste, B.18); dispersión = n.º de fuentes
  distintas argmax por trozo; margen_i = A(1)−A(2) y su mínimo;
  f_dos_peores = media de los dos peores m_i (con n == 1, f = m_1;
  con una sola fuente, margen_i = A(1), es decir, A(2) = 0).
  Todo `.get()`/`try-except`, NaN/inf → 0, ausencias → 0.0
  (sin respaldo, igual que `validacion.extraer_features`); caso sin
  oraciones o ilegible → `(0.0, 0, 0.0, 0.0)`.
- `03-build/reglas.py` NO se toca: el grupo queda AISLADO para la
  futura regresión logística (B.5), igual que `cifras.py`.
- `puntuar_auxiliar(caso) = 1 − f_dos_peores`: score SOLO-medición
  (no baseline) para estimar el poder predictivo del grupo con
  `validacion.comparar_scores` antes de la ablación logística.

## Medición en H (auxiliar vs baseline, mismos 631 casos)

Comando (desde `guardian/03-build/`): el bloque `comparar_scores`
sobre H con `base = reglas.puntuar_caso` y
`nuevo = alineacion.puntuar_auxiliar` (bootstrap pareado, 2000
remuestreos, mismos casos).

| Sistema (H, corte 0.5) | Puntuación | AUROC |
| --- | --- | --- |
| Baseline PASO 1 (reglas) | -0.449 | 0.735 |
| Auxiliar (1 − f_dos_peores) | -0.829 | 0.667 |

`comparar_scores` (nuevo − base):

- Δ AUROC = −0.068, IC 95 % (−0.099, −0.037): entero bajo 0.
- Δ pérdida = −0.036, IC 95 % (−0.097, +0.021): puntual negativa,
  intervalo a caballo del 0.

Separación por clase en H (media fieles frente a alucinadas):
cobertura 0.64 frente a 0.41, f_dos_peores 0.64 frente a 0.43,
margen_min 0.18 frente a 0.10, dispersión ~1.0 en ambas
(H trae 1–4 oraciones y el argmax casi no varía: poca señal ahí).

## Veredicto del gate: CONDICIONADO

NO ENTRA como score a mano: el auxiliar solo pierde contra el
baseline (Δ AUROC con IC entero < 0; Δ pérdida sin límite inferior
> 0) y no se demuestra arreglo caso por caso de ningún tipo de
`EJEMPLOS.md`. Pero tampoco se elimina: las medias por clase
separan en la dirección correcta y el auxiliar va SIN la señal del
juez (medición de suelo, no ablación: el baseline combina NLI +
juez y el auxiliar solo respaldo). Queda conservado pero NO
aprobado, aislado en `03-build/alineacion.py::feats_alin` para la
futura regresión logística; entrará solo si supera este gate en
ablación (Δ AUROC y Δ pérdida con IC 95 % > 0); si no, se quita en
Simplify.

## Tests

78 tests (`python3 -m unittest discover -s 04-test -p "test_*.py"`
desde `03-build/`): OK. 61 previos + 17 nuevos en
`04-test/test_alineacion.py` (A_ij = max con nombres de reglas.py,
cobertura total/nula/parcial, margen 0 con dos fuentes empatadas,
dispersión 1/2, f_dos_peores 0.25 frente a mínimo 0.2, n == 1,
sin señales/ilegible → defaults, NaN/inf → finito, auxiliar en
[0, 1]).
