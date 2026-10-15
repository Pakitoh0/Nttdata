# Resultados PASO 1 — Baseline de solo-reglas (`03-build/reglas.py` + `mi_bot.py`)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (1707 casos; H = 631: 334 fieles, 297 alucinaciones).
Comandos (desde `guardian/03-build/`):
`python evaluar_local.py --bot mi_bot.py --datos ../datos/train.jsonl --prefijo H`
(mismo para `ejemplo_bot.py`; mas tabla global sin `--prefijo`).

## H directo (corte 0.5)

| Bot | Puntuación | FN | FP | AUROC |
| --- | --- | --- | --- | --- |
| `mi_bot` (reglas) | -0.449 | 85 | 144 | 0.735 |
| `ejemplo_bot` (referencia) | -0.443 | 78 | 170 | 0.676 |

Global (1707 casos): `mi_bot` 0.184 (FN 110, FP 227, AUROC 0.909);
`ejemplo_bot` 0.024 (FN 104, FP 381, AUROC 0.802).

## GroupKFold por hash de fuentes (5 folds, `validacion.evaluar_bot_folds`, sin entreno)

`mi_bot` por fold (val / h / punt_h / auroc_h):
0: 342/124/-0.815/0.659 · 1: 342/124/-0.561/0.713 · 2: 341/139/-0.400/0.728 ·
3: 341/133/-0.028/0.821 · 4: 341/111/-0.491/0.751.
Media H: puntuación -0.459, AUROC 0.734 (631 H medidos una vez cada uno).

`ejemplo_bot` media H: puntuación -0.454, AUROC 0.676
(folds: -0.785/0.606, -0.470/0.659, -0.307/0.696, -0.183/0.744, -0.526/0.674).

## Lectura

- Puntuación H: empate técnico (-0.449 frente a -0.443, Δ = 0.006 << SE ≈ 0.08).
  Las reglas cambian el perfil de error (FN 85 frente a 78, FP 144 frente a 170):
  `respaldo_todas` deja de penalizar respuestas fieles que combinan fuentes (REQ-03).
- AUROC H: +0.059 a favor de las reglas (0.735 frente a 0.676), consistente en
  los 5 folds. La ordenación mejora aunque el corte 0.5 aún no lo convierte en aciertos.
- El mejor corte en muestra sería 0.052 (puntuación 0.042), pero es optimista
  por ajuste a H: se mantiene el reescalado B.2 por principio (p > 0.2 → 0.5).
- Ningún baseline alcanza la meta REQ-10 mínima (≥ 0.00 en H): H es duro para
  NLI + juez solos; los FN restantes piden cifras/atribución (PASO 2, B.6/B.13).

## Validez como referencia REQ-10

Válida. Es honesta (GroupKFold sin fuga: las reglas no entrenan; 631 H medidos
una vez), reproducible (sin parámetros aprendidos, solo stdlib), defensiva
(`.get()` + `try/except` → 0.55) y documentada (B.2-B.5, B.9-B.10). Fija el
listón: el modelo híbrido deberá superarla en H por > 1 SE (≈ 0.08), es decir,
puntuación H ≥ -0.37 con AUROC registrado; si no, se entrega esta versión de reglas.

## Tests

34 tests (`python3 -m unittest discover -s 04-test -p "test_*.py"`): OK.
28 existentes (metrica 12 + validacion 8 + veto 8) + 6 nuevos en
`04-test/test_reglas.py` (juez null → 0.5, mapeo veredicto/confianza,
max una-falsa-basta, clipping [0,1], fallback 0.55, formato sin etiqueta).
