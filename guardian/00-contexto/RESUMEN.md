# RESUMEN — El Guardián

Fuente: `material/kit/README.md`, `material/kit/EJEMPLOS.md` y
`material/enunciado_el_guarda.md` (verificado contra el `.docx` vía pandoc — mismo contenido).

## Qué se construye

Un detector de alucinaciones (guardrail): ante la respuesta de una IA y sus
fuentes, decide si la respuesta afirma algo que las fuentes no respaldan.

Contrato técnico: un fichero Python con una clase `Guardian` y un único método
`score(caso) -> float`, que devuelve la probabilidad de alucinación en [0, 1].
El organizador convierte cada score en decisión con corte 0.5
(score >= 0.5 = alucina). Si el mejor corte del bot no es 0.5, se reescala el
score para que lo sea.

## Entrada

Cada caso es un diccionario con cuatro claves:

- `pregunta`: texto de la pregunta del usuario.
- `fuentes`: lista de textos de referencia.
- `respuesta`: texto generado por la IA.
- `senales`: resultados precalculados de tres modelos (NLI, similitud, LLM juez),
  con dos niveles de segmentación:
  - `senales["oraciones"]`: una entrada por oración de la respuesta.
  - `senales["fragmentos"]`: trozos menores (cortados por comas y conjunciones).

Campos por oración:

- `texto`
- `respaldo_por_fuente[]`
- `contradiccion_por_fuente[]`
- `similitud_por_fuente[]`
- `respaldo_todas`
- `contradiccion_todas`
- `llm`: `{"veredicto": "respaldada" / "no_respaldada" / "contradicha" (o null si no contestó bien), "fuente": índice desde 0 o null, "confianza": 0–1}`.

Campos por fragmento: los mismos que por oración excepto `llm`, más el índice
`oracion` que indica de qué oración procede.

Advertencia del kit: las señales no son la respuesta. El NLI da falsas alarmas
con respuestas que combinan varias fuentes y se confunde con números; el LLM
juez acierta más pero se excede en confianza y da por buenas frases con detalles
añadidos.

## Los 4 tipos de alucinación

1. **Distorsión:** dato equivocado (cifra, fecha, unidad) o dato real atribuido a
   otra entidad.
2. **Contradicción:** afirma lo contrario que una fuente, a menudo sin negación
   explícita («aumentó» frente a «se redujo»).
3. **No respaldado:** añade algo plausible que ninguna fuente afirma (causa,
   fecha, atribución, detalle).
4. **Sobregeneralización:** amplía el alcance («en tres hospitales» →
   «en los hospitales del país»).

Criterio de oro: una respuesta es fiel (0) solo si **todo** lo que afirma está
respaldado por las fuentes leídas en conjunto. Basta **una** afirmación sin
respaldo para que sea alucinación (1). Cuenta lo que dicen las fuentes, no la
verdad externa. La respuesta fiel puede parafrasear, combinar fuentes, omitir
información y hacer aritmética directa o cambios de unidad exactos; no puede
endurecer verbos, eliminar matices, ampliar alcance, añadir causas ni cambiar
atribuciones.

## Métrica

`puntuación = 1 - (4·FN + FP) / n_fieles`, donde FN son alucinaciones que se
escapan y FP son falsas alarmas. 1 = sin fallos; 0 = equivalente a marcar todo
como alucinación; negativo = peor que no hacer nada. Dejar escapar una
alucinación pesa 4 veces más que una falsa alarma. El AUROC (ordenación de
casos al margen del corte) sirve para desempatar. Test y giro se dividen en
mitad pública (leaderboard en directo) y mitad privada (clasificación final).

## Datos

- `train.jsonl`: 1707 casos etiquetados. Los 631 con id `H…` están escritos como
  el test (muchas fuentes, combinación de datos, alucinaciones sutiles); el
  resto es más sencillo.
- `test_publico.jsonl`: 440 casos sin etiquetas, entregado el jueves a las 10:00.
- `giro_publico.jsonl`: 220 casos sin etiquetas, publicado el jueves a las 15:30
  (documentos técnicos y legales, respuestas más largas).

## Entregables

- `predicciones.csv`: una línea `id,score` por caso, separador coma y punto
  decimal (cabecera `id,score` opcional, sin ids repetidos). Se genera siempre
  con `generar_predicciones.py`; antes del giro solo con test, después con
  test + giro juntos. Cupo: 3 entregas antes del giro y 3 después.
- `mi_bot.py`: el bot, entregado junto con el CSV en cada entrega.

## Calendario

- Jueves 10:00 — presentación, entrega del kit y del test; inicio de entregas.
- Jueves 14:15 — comida.
- Jueves 15:30 — giro; a partir de aquí se entrega test + giro juntos.
- Jueves 19:00 — cierre de entregas y cálculo de la clasificación final (parte privada).
- Jueves 19:00–20:15 — reproducción: los 5 primeros ejecutan su bot sobre un lote nuevo.
- Jueves 20:30 — anuncio de los 5 finalistas.
- Viernes 9:30 — pitch de finalistas (5 minutos).
- Viernes 12:00 — entrega de premios y clausura.

## Nota

Nota final = 60 × puntuación final (mínimo 0) + hasta 40 puntos del jurado
(solo finalistas): Método 15, Calibración y giro 10, Originalidad 5, Código y
defensa 5, Propuesta de generalización 5.
