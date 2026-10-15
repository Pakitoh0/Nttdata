# Resultados PASO 4 — Marcadores diferenciales + flip de polaridad (B.6 + B.12)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (H = 631: 334 fieles, 297 alucinaciones).

## Cambio

- Nuevo `03-build/marcadores.py` (solo stdlib, español neutro):
  `feats_marc(caso)` devuelve (f_caus, f_alc, f_hedge, flip).
  `f_caus`/`f_alc` = 1 si hay marcador causal/cuantificador en R ausente
  en la fuente citada; `f_hedge` = n.º de matices en la fuente citada
  ausentes en R (cada ocurrencia cuenta solo si su ventana ±3 comparte
  contenido con R, para no contar matices de frases no retomadas);
  `flip` = 1 si hay verbos de signo opuesto (léxico mínimo B.12) con
  Jaccard ponderado (idf intra-caso) del objeto ±5 ≥ 0,5.
  "Fuente citada" = la del veredicto llm si existe (índices
  `senales.oraciones[].llm.fuente`), si no todas concatenadas.
  Minúsculas + sin acentos (NFKD, documentado). Todo `.get()`/
  `try-except`; ilegible → `(0, 0, 0, 0)`.
- `03-build/reglas.py` NO se toca: el grupo queda AISLADO para la
  futura regresión logística (B.5), igual que `cifras.py` y
  `alineacion.py`.

## Medición en H (standalone por marcador, suelo documentado)

Comando (desde `guardian/03-build/`): bloque `comparar_scores` sobre H
con `base = reglas.puntuar_caso` y `nuevo` = valor del marcador como
score (bootstrap pareado, 2000 remuestreos, mismos casos). Es suelo, no
ablación: el baseline combina NLI + juez y cada marcador va solo.

| Marcador | P(>0\|fiel) | P(>0\|halu) | Δ AUROC [IC 95 %] | Δ pérdida [IC 95 %] |
| --- | --- | --- | --- | --- |
| f_caus | 0,030 (10/334) | 0,064 (19/297) | −0,218 (−0,255, −0,176) | −0,042 (−0,117, +0,000) |
| f_alc | 0,072 (24/334) | 0,020 (6/297) | −0,261 (−0,302, −0,218) | −0,042 (−0,117, +0,000) |
| f_hedge | 0,045 (15/334) | 0,168 (50/297) | −0,173 (−0,219, −0,126) | −0,042 (−0,117, +0,000) |
| flip | 0,000 (0/334) | 0,000 (0/297) | −0,235 (−0,272, −0,195) | −0,042 (−0,117, +0,000) |

Conductores: causal = "gracias a" (14) + "porque" (12); alcance =
"cada" (16, uso benigno tipo "cada año") + "todos" (5); matiz =
"solo" (37) + "podría" (22) + "al menos" (10). Flip: 43 casos con
verbos firmados en R y S, pero el mejor Jaccard ponderado del objeto
es 0,21 < 0,5 (la ventana ±5 con idf diluye el objeto): cero disparos.

## Veredicto del gate por marcador

Ningún marcador cumple ENTRA standalone (Δ AUROC y Δ pérdida con IC
95 % > 0 en ambas; todos pierden contra el baseline completo) ni se
demuestra arreglo caso por caso de un tipo de `EJEMPLOS.md`.

- f_hedge: CONDICIONADO. Separa en la dirección correcta (≈3,7× más en
  halu, 65 casos con señal). Conservado aislado en
  `03-build/marcadores.py::feats_marc` para la futura regresión
  logística; entrará solo si supera este gate en ablación; si no, se
  quita en Simplify.
- f_caus: CONDICIONADO (débil). Dirección correcta (≈2× más en halu)
  pero soporte bajo (29 casos). Mismo régimen que f_hedge.
- f_alc: FUERA. Dirección invertida (las fieles añaden "cada"/"todos"
  más que las halu): como penalización a mano sumaría donde debería
  restar. Se quita en Simplify (la logística futura podría rescatar su
  signo, pero no hay evidencia para conservarlo).
- flip: FUERA. Sin señal en H con la aproximación documentada (cero
  disparos con umbral 0,5 de B.12). Se quita en Simplify.

## Tests

Suite completa en verde (`python3 -m unittest discover -s 04-test -p "test_*.py"`):
previos + 18 nuevos en `04-test/test_marcadores.py` (causal Montejo
"gracias a" → 1, alcance todos frente a varias, matiz perdido/
conservado/múltiple/ventana, flip mismo/distinto objeto y mismo signo,
fuente citada selectiva y fallback a todas, sin marcadores → 0,
ilegible → 0 sin lanzar).
