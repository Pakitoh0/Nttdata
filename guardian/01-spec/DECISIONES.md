# DECISIONES — Decisiones de Spec (cerradas)

> Copia fiel del ANEXO A del orquestador. Sin resúmenes ni reinterpretaciones.
> Ante discrepancia, manda el orquestador.

ANEXO A — Decisiones de Spec (cerradas):

- Enfoque: híbrido. Las reglas (cifras, atribución, causalidad, alcance, matices, verbos) se implementan como features numéricas; un clasificador pequeño y regularizado (regresión logística L2) las combina. Pocas features (≈15-25), todas normalizadas.
- Condición de salida: si en H (validación agrupada) el modelo no supera a la línea base de solo reglas por más de 1 error estándar (≈0,08), se entrega la versión de reglas.
- Corte: p > 0,2 (coste 4:1) reescalado para que caiga en 0,5 (ver B.2).
- Metas de score medidas en H (GroupKFold por hash de fuentes, corte ya reescalado): Mínimo ≥0,00 (mejor que no hacer nada); Objetivo ≥0,25 (muy por encima del juez LLM a ciegas); Estirado ≥0,40 ("muy bueno" según el enunciado); Giro caída ≤0,10 respecto a H (robustez). Registrar también el AUROC. Con ~631 casos H el ruido de la nota es ≈ ±0,08: las metas son rangos.
- Checklist ESPEC.md antes de cerrar Spec: (1) entrada caso[pregunta|fuentes|respuesta|senales], salida float [0,1], umbral 0,5; (2) métrica exacta 1−(4·FN+FP)/N_f y AUROC; (3) solo Python 3.9+, estándar, numpy y sklearn, ninguna IA en el bot, coeficientes embebidos en mi_bot.py; (4) robustez al giro .get()/try-except/fallback algo por encima de 0,5; (5) medir con --prefijo H y validación agrupada sin fuga; (6) nada de etiquetar test ni reglas por caso.
