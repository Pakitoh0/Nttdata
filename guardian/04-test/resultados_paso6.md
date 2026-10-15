# Resultados PASO 6 — Logística híbrida + ablación (B.7, B.8)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (1707 casos; H = 631:
334 fieles, 297 alucinaciones). Comando (desde `guardian/03-build/`):
`python3 modelo.py` (imprime C elegidas + tabla) + puntuación OOF a
corte 0,5 para REQ-10.

## Cambio

- Nuevo `03-build/modelo.py` (sklearn + numpy, RNF-01): X con 9
  columnas SOLO de grupos condicionados + baseline (`f_num`, `f_year`;
  `f_hedge`, `f_caus`; `cobertura`, `dispersion`, `margen_min`,
  `f_dos_peores`; `score_base` de `reglas.puntuar_caso`). Fuera a
  propósito: `f_deriv`, `f_alc`, `flip`. Estandariza con μ/σ de train
  y los devuelve embebibles (`ajustar` → dict con mu/sigma/coef/
  intercepto/C; sigma es el `scale_` de StandardScaler).
- Coste 4:1 por UNA sola vía (B.2): entreno SIN peso de clase +
  reescalado posterior `s = 4p/(1+3p)` (p = 0,2 → s = 0,5 exacto).
  C por CV simple interna (GroupKFold ×3, AUROC medio, parrilla
  0,01/0,1/1,0/10,0) solo sobre el entreno del fold, sin tocar H.
- Ablación con el MISMO GroupKFold de `validacion.py`
  (`folds_por_grupo` + `verificar_disjuncion`, hash de fuentes;
  entreno sin H, como `evaluar_folds`): completo vs sin cada grupo
  vs baseline, con `comparar_scores` (ΔAUROC + Δpérdida, IC95 pareado,
  2000 remuestreos). C elegidas: 1,0 en 19/20 (fold, variante);
  solo `sin_alineacion` pide 0,1 en 4 folds.

## Ablación en H (OOF, 631 casos; base = sin_grupo, nuevo = completo)

| Grupo retirado | ΔAUROC [IC 95 %] | Δpérdida [IC 95 %] | Veredicto |
| --- | --- | --- | --- |
| cifras | −0,000 (−0,004, +0,003) | −0,006 (−0,023, +0,009) | FUERA |
| marcadores | −0,008 (−0,014, −0,002) | −0,030 (−0,063, −0,003) | FUERA |
| alineación | +0,023 (+0,009, +0,038) | +0,078 (+0,025, +0,137) | ENTRA |

Gate (CRITERIO_GATE.md: IC95 claramente > 0 en AMBAS): solo
alineación lo cumple. Cifras no aporta nada (Δ ≈ 0). Marcadores resta:
quitarlo MEJORA ambas (límites superiores bajo 0). Ningún grupo arregla
un tipo de EJEMPLOS.md caso por caso (no se invoca la vía alternativa).

Puntuación OOF en H a corte 0,5 (contexto): reglas −0,449
(AUROC 0,735); completo −0,338 (0,742); sin_cifras −0,341 (0,742);
sin_marcadores −0,296 (0,749); sin_alineacion −0,302 (0,718).
Coherente: sin_marcadores es el mejor numérico (el grupo resta) y sin
alineación pierde ordenación (0,718 < 0,735 del baseline).

## Veredicto REQ-10

La logística completa supera a las reglas en H por +0,111 de
puntuación (−0,338 frente a −0,449), > 1 SE (≈0,08): la condición de
salida NO se activa (se sigue con el híbrido, no se entrega reglas).
Respaldo: Δpérdida +0,066 (IC95 +0,014, +0,114); ΔAUROC +0,007
(−0,008, +0,022), no significativo: la ganancia entra por el corte
reescalado, no por la ordenación. Nota para Simplify: el candidato a
entregar es el reentreno sin cifras ni marcadores (ENTRA solo
alineación + baseline).

## Tests

Suite completa en verde (`python3 -m unittest discover -s 04-test
-p "test_*.py"` desde `guardian/`: 119 tests OK, 108 previos + 11
nuevos en `04-test/test_modelo.py`): 9 columnas condicionadas
(f_deriv/f_alc/flip fuera), variantes que quitan un grupo,
reescalado p = 0,2 → s = 0,5 exacto + monotonía, overfit sintético
separable, μ/σ embebidos que reproducen score (10⁻⁹), C determinista
dentro de train, mismo GroupKFold sin solape y H medido una vez,
humo OOF con formas y rangos.
