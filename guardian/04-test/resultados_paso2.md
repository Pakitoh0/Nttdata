# Resultados PASO 2 — Cifras con cierre por derivación (B.6 + B.13)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (H = 631: 334 fieles, 297 alucinaciones).
Comandos (desde `guardian/03-build/`):
`python evaluar_local.py --bot mi_bot.py --datos ../datos/train.jsonl --prefijo H`
+ `evaluar_bot_folds` (GroupKFold por hash de fuentes, 5 folds).

## Cambio

- Nuevo `03-build/cifras.py` (solo stdlib): extracción ES (dígitos con
  separadores, porcentajes, palabras, magnitudes mil/millones, unidades
  €/km/m/%), normalización (2.000 m → 2 km, 40 % → 0.4), cierre D(N_S)
  con subconjuntos hasta tamaño 3 (|N_S| ≤ 12), δ(x) = 0.5·10^(−d) con
  último dígito significativo no cero en redondas, +5 % relativo con
  redondeo declarado.
- `03-build/reglas.py`: penalización `PESO_CIFRAS * f_num` (máx. +0.10),
  solo suma, con f_num = 0 el score es idéntico al PASO 1. Las 6 reglas
  existentes no se tocan.
- `f_deriv` NO entra como penalización: las respuestas fieles derivan
  más (media 0.43 frente a 0.22 en alucinadas; P(f_deriv>0) 0.38 frente
  a 0.20). Queda disponible para uso futuro (evidencia negativa o
  interacción z4 de B.15).

## H directo (corte 0.5)

| Versión | Puntuación | FN | FP | AUROC |
| --- | --- | --- | --- | --- |
| PASO 1 (reglas) | -0.449 | 85 | 144 | 0.735 |
| PASO 2 (+0.10·f_num) | -0.428 | 82 | 149 | 0.728 |
| Δ | +0.021 | -3 | +5 | -0.007 |

## GroupKFold por hash de fuentes (`evaluar_bot_folds`, sin entreno)

PASO 2 por fold (punt_h / auroc_h):
0: -0.708/0.652 · 1: -0.576/0.706 · 2: -0.427/0.710 ·
3: 0.014/0.824 · 4: -0.491/0.755.
Media H: puntuación -0.437, AUROC 0.730.

PASO 1 media H: puntuación -0.459, AUROC 0.734
(folds: -0.815/0.659, -0.561/0.713, -0.400/0.728, -0.028/0.821, -0.491/0.751).

La puntuación mejora en 4/5 folds (fold 4 empata); el AUROC cede en 3/5.

## Veredicto del gate: NO ENTRA

Δ H = +0.021 << SE ≈ 0.08 (ruido) y el AUROC empeora (-0.007 directo,
-0.004 media en folds). No hay mejora por encima del ruido ni mejora
clara y estable: la dirección es consistente en puntuación pero el
efecto es ruido con coste de ordenación. Se mantiene el PASO 1 como
referencia REQ-10; `cifras.py` + tests quedan como infraestructura
medida para pasos siguientes (B.15: f_deriv como evidencia negativa,
interacción f_num·f_attr).

## Tests

47 tests (`python3 -m unittest discover -s 04-test -p "test_*.py"`): OK.
34 previos + 13 nuevos en `04-test/test_cifras.py` (28.000=28000,
tres=3, 40 %=0.4, 2.000 m=2 km, derivación suma/doble/porcentaje,
redondeo declarado, sin cifras → f=0).
