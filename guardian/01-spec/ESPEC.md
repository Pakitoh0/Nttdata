# ESPEC — Especificación del Guardián

## Requisitos funcionales

- **REQ-01 — Contrato de puntuación:** `Guardian.score(caso)` devuelve un float
  en [0, 1] = probabilidad de que la respuesta alucine. Corte operativo 0.5
  (score >= 0.5 = alucina). Si el mejor corte empírico difiere, reescalar el
  score para que 0.5 sea el punto de decisión.
- **REQ-02 — Entrada completa:** el bot consume `pregunta`, `fuentes`,
  `respuesta` y `senales` (oraciones y fragmentos con `texto`,
  `respaldo_por_fuente`, `contradiccion_por_fuente`, `similitud_por_fuente`,
  `respaldo_todas`, `contradiccion_todas` y `llm` solo en oraciones). Debe
  funcionar aunque falten claves o el veredicto `llm` sea null.
- **REQ-03 — Combinación multi-fuente:** verificar cada afirmación contra las
  fuentes leídas en conjunto (`respaldo_todas`) y a nivel de fragmentos, no solo
  contra cada fuente por separado (`respaldo_por_fuente`), para no penalizar
  respuestas fieles que combinan varias fuentes.
- **REQ-04 — Peor afirmación manda:** agregar por mínimo/máximo de sospecha
  (la afirmación más débil determina el score), nunca por promedio sobre
  oraciones: una sola afirmación falsa basta para declarar alucinación.
- **REQ-05 — Desconfianza calibrada del LLM juez:** ponderar el veredicto `llm`
  por su `confianza` y contrastarlo con el NLI y con comprobaciones propias
  (cifras, fechas, atribución entidad-dato, alcance, fuerza del verbo, matices);
  ante conflicto o confianza baja, debe prevalecer la evidencia contraria.
  El coste 4:1 (FN frente a FP) exige priorizar la exhaustividad.
- **REQ-06 — Robustez ante el giro:** mantener el rendimiento con documentos
  técnicos y legales y respuestas más largas sin reglas dependientes del formato
  del train; accesos defensivos (`.get()`, valores por defecto, `try-except`) y
  ausencia de atajos no predictivos.
- **REQ-07 — Reproducibilidad:** parámetros embebidos en `mi_bot.py`, sin
  `pickle` ni ficheros externos; ejecución determinista sobre lotes nuevos con
  el mismo formato de `predicciones.csv` (`id,score`, coma, punto decimal).
- **REQ-08 — Métrica exacta + desempate AUROC (implementado):**
  `puntuacion()` en `03-build/evaluar_local.py` calcula
  `1 - (4·FN + FP) / n_fieles` (verificada contra la fórmula, no reescrita);
  caso borde `n_fieles == 0` definido como `1 - FN / n_alucinaciones`
  (1.0 si vacío), documentado en `04-test/METRICA.md`; `comparar(a, b)`
  desempatada por mayor AUROC; 12 tests en `04-test/test_metrica.py`.
- **REQ-09 — GroupKFold por hash de fuentes + evaluación en H
  (implementado):** `03-build/validacion.py` agrupa por sha256 de la tupla
  de `fuentes` normalizada (strip, colapso `\s+`, casefold), reparte con
  `GroupKFold` (5 folds), verifica disjunción de grupos por fold, ajusta
  scaler y modelo SOLO con el entreno del fold y reporta puntuación/AUROC
  global y del subconjunto H; política: con `excluir_h_entreno=True` ningún
  H entrena al medir H; 8 tests en `04-test/test_validacion.py` con datos
  reales (631 H medidos una vez cada uno).
- **REQ-10 — Enfoque híbrido:** las reglas (cifras, atribución, causalidad,
  alcance, matices, verbos) se implementan como features numéricas (≈15-25,
  todas normalizadas); una regresión logística L2 las combina. Corte p>0,2
  reescalado a 0,5. Condición de salida: si en H el modelo no supera a la
  baseline de solo-reglas por >1 SE (≈0,08), se entrega la versión de reglas.
  Metas en H: mínimo ≥0,00, objetivo ≥0,25, estirado ≥0,40, caída en giro
  ≤0,10; registrar AUROC.

## Requisitos no funcionales

- **RNF-01 — Dependencias:** solo librería estándar de Python, `numpy` y
  `scikit-learn`; Python 3.9 o superior; sin modelos de IA en tiempo de
  ejecución (ni APIs ni modelos locales).
- **RNF-02 — Legibilidad y defensa:** código claro, comentado en lo esencial y
  explicable en 5 minutos ante el jurado (qué señales usa, cuándo desconfía del
  NLI y del LLM juez, por qué falla donde falla, límites y generalización).
- **RNF-03 — Medición trazable:** cada versión se evalúa con `evaluar_local.py`
  sobre `train.jsonl` global y con `--prefijo H`, con registro de puntuación,
  FN, FP y AUROC en `04-test/` antes de avanzar de fase.
- **RNF-04 — Veto del test (implementado):** `03-build/carga.py` separa
  train/val (con `etiqueta`, vía `cargar_train()`) de test (sin `etiqueta`,
  vía `cargar_test()`) y falla con `VetoTestError` si se mezclan;
  `preparar_entrenamiento()` veta además por nombre (`test_publico`,
  `giro`); usos permitidos/prohibidos en `00-contexto/REGLAS.md` §7;
  8 tests en `04-test/test_veto.py`.
