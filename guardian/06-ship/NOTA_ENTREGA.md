# Nota de entrega 2 — modelo final híbrido (PASO 7 + SIMPLIFY + SHIP)

Versión: baseline reglas + alineación combinados por logística L2
(C = 1,0), reescalado B.2, corte 0,5. `06-ship/mi_bot.py` autocontenido
(solo stdlib `math`; sigmoide y producto escalar a mano; parámetros
μ, σ, w, b como literales; `.get()`/`try-except` con fallback 0,55;
recorte [0,1]). Reentreno de entrega sobre TODO train no-H
(1076 casos; ningún H entrena — política REQ-09): H queda 100 %
held-out. Podas documentadas en `05-simplify/NOTAS.md` (cifras,
marcadores, atribución, flip y Platt fuera con motivo).

Métricas H (corte 0,5):
- Modelo embarcado en H (held-out): puntuación −0,299 · AUROC 0,751
  (FN 68, FP 162) — coincide con el OOF del paso 7 (−0,308 / 0,749 /
  FN 69 / FP 161, `04-test/resultados_paso7.md`) dentro de ruido.
- Baseline reglas en H: −0,449 / 0,735 (FN 85, FP 144).
- Mejora sobre reglas: +0,150 (> 1 SE ≈ 0,08) → veredicto REQ-10 CUMPLE.
- Réplica exacta: max|modelo − ship| = 0,00e+00 sobre los 1707 casos.

Comando usado (desde `guardian/`):

```sh
python3 03-build/generar_predicciones.py --bot 06-ship/mi_bot.py --test datos/test_publico.jsonl --salida 06-ship/predicciones.csv
```

Validaciones de formato (observadas):
- 441 líneas: cabecera `id,score` + 440 filas.
- Separador coma, punto decimal, 6 decimales, scores en [0,1]
  (rango observado 0,006979–0,992804).
- Sin ids repetidos; coinciden con `datos/test_publico.jsonl`.
- Prueba ciega OK: copia de `mi_bot.py` en dir temporal vacío puntúa
  3 casos train sin etiqueta
  (e01 0,004809, e01::par 0,002692, e01::distorsion 0,896992).
- `mi_bot.py` solo importa `math`; sin módulos propios, sin `open()`,
  sin pickle, sin sklearn/numpy. Clase única `Guardian` con
  `score(caso)->float`, fallback 0,55, recorte [0,1].

Pasos que debe hacer el usuario (no hechos por el agente):
1. Fork del repo del organizador.
2. Crear carpeta `NOMBRE_EQUIPO/entrega_2/`.
3. Subir los DOS ficheros: `mi_bot.py` y `predicciones.csv`.
4. Abrir PR titulado `NOMBRE_EQUIPO entrega 2`.
