# Resultados PASO 7 — Reentreno limpio + calibración (B.2, B.8, B.16)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (1707 casos; H = 631:
334 fieles, 297 alucinaciones). Mismo GroupKFold por hash de fuentes
(`folds_por_grupo` + `verificar_disjuncion`); entreno de cada fold
excluye H; C por CV interna (GroupKFold ×3, AUROC medio, parrilla
0,01/0,1/1,0/10,0) solo sobre el entreno del fold. SIN entrega
(nada en `06-ship/`).

## 1. Configuración final (fijada en `03-build/modelo.py`)

`COLUMNAS_FINAL = (cobertura, dispersion, margen_min, f_dos_peores,
score_base)` = baseline reglas + grupo alineación. Reescalado B.2
`s = 4p/(1+3p)` (p = 0,2 → s = 0,5); UNA sola vía de coste 4:1
(entreno SIN peso de clase). `COLUMNAS`/`VARIANTES`/`oof_ablacion`
del paso 6 se conservan intactos como auditoría; lo reentrenado y
medido aquí es `oof_final` (5 columnas).

Motivo (ablación paso 6, OOF en H; gate = IC95 > 0 en AMBAS):

| Grupo | ΔAUROC [IC95] | Δpérdida [IC95] | Decisión |
| --- | --- | --- | --- |
| cifras | −0,000 (−0,004, +0,003) | −0,006 (−0,023, +0,009) | FUERA (Δ ≈ 0, no aporta) |
| marcadores | −0,008 (−0,014, −0,002) | −0,030 (−0,063, −0,003) | FUERA (resta: quitarlo mejora ambas) |
| alineación | +0,023 (+0,009, +0,038) | +0,078 (+0,025, +0,137) | ENTRA (único con IC95 > 0 en ambas) |

C elegidas final por fold: 1,0 / 1,0 / 1,0 / 1,0 / 10,0.

## 2. Calibración (B.8, B.16)

ECE en validación, bins = 10 uniformes, sobre OOF-H del mismo
GroupKFold (`modelo.ece`):

| Sistema | ECE |
| --- | --- |
| final sin calibrar | 0,2254 |
| reglas | 0,2104 |

Detalle pre (bin, n, acc, conf): [0,0,1): 50, 0,040, 0,044;
[0,1,0,2): 93, 0,312, 0,150; [0,2,0,3): 44, 0,432, 0,248;
[0,3,0,4): 30, 0,333, 0,353; [0,4,0,5): 25, 0,360, 0,452;
[0,5,0,6): 22, 0,364, 0,548; [0,6,0,7): 33, 0,394, 0,652;
[0,7,0,8): 38, 0,368, 0,747; [0,8,0,9): 42, 0,381, 0,855;
[0,9,1,0]: 254, 0,697, 0,969. Sobreconfianza arriba (0,969 vs 0,697).

ECE 0,2254 > 0,05 → Platt `σ(a·z+b)`, z = logit recortado (ε = 1e-4),
ajustada EN VALIDACIÓN (OOF-H, 631 puntos; nunca en train):
a = 0,3785 (< 1: aplasta la sobreconfianza, coherente con B.4),
b = −0,5538. Re-medición: ECE post = 0,0944 (mejora, sigue > 0,05).

Decisión operativa: el punto de corte 0,5 lo fijó el reescalado B.2
y Platt es monótona (AUROC idéntico 0,749; Δpérdida con mejor corte
invariante). A corte fijo 0,5 la calibrada da −0,566 (FN 105, FP 103):
empeora el operativo. Se documenta y NO se adopta a 0,5 (adoptarla
exigiría re-derivar el umbral por pérdida esperada, B.16 t*, fuera
de este paso). El modelo operativo es el final SIN Platt.

## 3. Final OOF en H (corte 0,5) y REQ-10

| Sistema | Puntuación | AUROC | FN | FP |
| --- | --- | --- | --- | --- |
| final (baseline + alineación) | −0,308 | 0,749 | 69 | 161 |
| reglas | −0,449 | 0,735 | 85 | 144 |

`comparar_scores` final vs reglas (pareado, 2000 remuestreos):
ΔAUROC +0,014 (IC95 −0,000, +0,029: incluye el 0, no significativo);
Δpérdida +0,111 (IC95 +0,056, +0,154: claramente > 0).

Veredicto REQ-10 final: CUMPLE (se sigue con el híbrido, no se
entrega reglas). Puntuación +0,141 sobre reglas (> 1 SE ≈ 0,08) y
Δpérdida con IC95 > 0; la ganancia entra por el corte reescalado,
no por la ordenación (igual que en el paso 6).

## 4. Tests

10 nuevos en `04-test/test_modelo.py`: ECE perfecto ≈ 0,
sobreconfianza 0,4, bins vacíos/rangos; Platt monótona (a > 0),
extremos finitos, calibración que baja ECE sintética; COLUMNAS_FINAL
(5, cifras/marcadores fuera), vector final consistente, humo
`oof_final`, y reproducción del número reportado
(−0,308 / 0,749 / FN 69 / FP 161).

Suite completa en verde desde `guardian/`:
`python3 -m unittest discover -s 04-test -p "test_*.py"`: 129 tests OK
(119 previos + 10 nuevos).
