# Resultados PASO 5 — Atribución + desacuerdo con el juez (B.4 + B.6)

Fecha: 2026-10-15. Datos: `datos/train.jsonl` (H = 631: 334 fieles, 297 alucinaciones).

## Cambio

- Nuevo `03-build/atribucion.py` (solo stdlib, español neutro):
  `feats_atrib(caso)` devuelve (f_attr, margen, d1min, d2max, d3max).
  idf intra-caso solo sobre fuentes (`ln((m+1)/(df+1)) + 1`); por
  cifra de la respuesta, ventanas ±10 tokens en respuesta y en cada
  fuente, `attr(n)` = mejor Jaccard ponderado, `f_attr` = mínimo;
  `margen(n)` = mejor − mejor de otra fuente (0.0 si no hay otra),
  `margen` = mínimo. Sin cifras → (1.0, 1.0) vacuidad. Desacuerdo
  por oración con la misma convención que `reglas.py::prob_juez_llm`
  (malo → κ, respaldada → 1−κ, null → 0.5; Platt sin ajustar: q_LLM
  cruda para la futura logística): `a_i` = max(por_fuente, todas),
  d1 = ℓ(q)−ℓ(1−a), d2 = (1−q)(1−a), d3 = (1−q)·contradicción;
  agregación una-falsa-basta (d1min, d2max, d3max). Todo `.get()`/
  `try-except`; NaN/inf → 0; ilegible → (1.0, 1.0, 0.0, 0.0, 0.0).
- `03-build/reglas.py` NO se toca: el grupo queda AISLADO para la
  futura regresión logística (B.5), igual que `cifras.py`,
  `alineacion.py` y `marcadores.py`.

## Medición en H (standalone vs baseline, mismos 631 casos)

Comando (desde `guardian/03-build/`): bloque `comparar_scores` sobre
H con `base = reglas.puntuar_caso` y `nuevo` = score de sospecha de
cada feature (1−f_attr, 1−margen, −d1min, d2max, d3max; bootstrap
pareado, 2000 remuestreos). Es suelo, no ablación: el baseline
combina NLI + juez y cada feature va sola.

| Feature | media fiel | media halu | Δ AUROC [IC 95 %] | Δ pérdida [IC 95 %] |
| --- | --- | --- | --- | --- |
| f_attr | 0,425 | 0,492 | −0,335 (−0,393, −0,276) | −0,042 (−0,117, +0,000) |
| margen | 0,248 | 0,320 | −0,319 (−0,376, −0,258) | −0,042 (−0,117, +0,000) |
| d1min | −3,281 | −1,439 | −0,363 (−0,417, −0,304) | −0,042 (−0,117, +0,000) |
| d2max | 0,340 | 0,294 | −0,259 (−0,308, −0,210) | −0,042 (−0,117, +0,000) |
| d3max | 0,293 | 0,139 | −0,397 (−0,452, −0,340) | −0,042 (−0,117, +0,000) |

Tasas (soporte alto, ~93 % de H con cifras, pero dirección
invertida en todas):

| Regla | P(fiel) | P(halu) |
| --- | --- | --- |
| f_attr < 0,5 | 0,746 (249/334) | 0,616 (183/297) |
| margen < 0,3 | 0,692 (231/334) | 0,562 (167/297) |
| margen < 0,3 y f_attr > 0,3 | 0,428 (143/334) | 0,401 (119/297) |
| d2 > 0,5 | 0,353 (118/334) | 0,279 (83/297) |
| d3 > 0,3 | 0,338 (113/334) | 0,162 (48/297) |
| d1 < −4 | 0,431 (144/334) | 0,306 (91/297) |

Lectura: las halu copian las cifras igual o mejor que las fieles
(f_attr mayor en halu: 0,492 frente a 0,425) y d2/d3 multiplican por
(1−q_LLM), que es grande justo cuando el juez dice "respaldada":
disparan más en fieles. Como penalización a mano sumarían donde
deberían restar. Ni la regla conjunta (margen bajo + f_attr medio)
separa (0,428 frente a 0,401).

## Veredicto del gate por feature

Ninguna feature cumple ENTRA standalone (Δ AUROC y Δ pérdida con IC
95 % > 0 en ambas; todas pierden contra el baseline completo) ni se
demuestra arreglo caso por caso de un tipo de `EJEMPLOS.md` (la
separación cruzada/bien solo se muestra en sintético, ver Tests).

- f_attr: FUERA. Dirección invertida en H. Se quita en Simplify (la
  logística futura podría rescatarla vía interacción z4 de B.15,
  pero no hay evidencia para conservarla).
- margen: FUERA. Misma inversión; ni conjunta separa. Se quita en
  Simplify.
- d1min, d2max, d3max: FUERA. Las tres disparan más en fieles que
  en halu con los umbrales naturales. Se quitan en Simplify.

## Tests

Suite completa en verde (`python3 -m unittest discover -s 04-test
-p "test_*.py"` desde `guardian/`: 108 tests OK, 96 previos + 12
nuevos en `04-test/test_atribucion.py`): cruzada B-factura-42-de-A
→ f_attr < 0,7, bien atribuida → > 0,75, orden cruz < bien en f_attr
y margen, d2 ≈ 0,81 Montejo-like ("respaldada" 0,9 + NLI 0,1),
convención q_LLM de `reglas.py`, veredicto null → q = 0,5 finito,
sin cifras → (1,0, 1,0), ilegible/NaN → finito sin lanzar, peor
trozo manda (d1min/d2max).
