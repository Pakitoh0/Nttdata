# REGLAS — Restricciones duras del reto

1. **Entorno permitido:** solo la librería estándar de Python, `numpy` y
   `scikit-learn` (por ejemplo, para combinar señales o entrenar un clasificador
   con `train.jsonl`). No es obligatorio usarlas; se puede ganar con Python puro.
2. **IA prohibida en el bot, permitida para programar:** el bot no puede usar
   modelos de IA de ningún tipo (ni APIs externas, ni modelos locales, ni otros
   NLI o embeddings); decide únicamente a partir de las señales y de
   comprobaciones en Python. Sí está permitido usar asistentes como Copilot o
   ChatGPT para escribir el código.
3. **Test intocable:** se puede leer `test_publico.jsonl` (y el giro) para
   entender el formato, pero está prohibido etiquetarlo a mano o incluir reglas
   o datos pensados para casos concretos del test.
4. **Validación honesta:** validar con validación cruzada sin entrenar sobre los
   mismos casos con los que se mide; calibrar y medir sobre todo con
   `--prefijo H` (casos parecidos al test), porque calibrar con todo el train
   genera exceso de confianza.
5. **Robustez ante el giro:** el giro cambia de terreno (formato distinto,
   documentos técnicos y legales, respuestas más largas). El bot debe tolerar
   señales ausentes o con formato inesperado mediante `.get()`, valores por
   defecto y `try-except`; nada de atajos frágiles (longitud, superlativos,
   negaciones: no predicen la etiqueta).
6. **Reproducción autónoma:** todos los parámetros deben ir embebidos en
   `mi_bot.py`; prohibido `pickle` o ficheros externos de pesos. El bot debe
   ejecutarse delante del jurado sobre un lote nuevo con resultado similar; una
   caída anómala o un código distinto al entregado descalifica la entrega.

## 7. Veto del test en el entrenamiento (RNF-04, implementado en `03-build/carga.py`)

Usos permitidos: leer `test_publico.jsonl` para entender el formato;
cargarlo solo con `cargar_test()` (falla si algún caso trae `etiqueta`);
entrenar y validar solo con ficheros aceptados por
`preparar_entrenamiento()` (cada caso con `etiqueta` 0/1).

Usos prohibidos: entrenar, calibrar o seleccionar reglas con
`test_publico.jsonl` o `giro*.jsonl` (vetados por nombre y por contenido:
cualquier caso sin `etiqueta` dispara `VetoTestError`); etiquetar el test
a mano; colar casos con `etiqueta` en el test (`cargar_test()` falla si
los hay).

Criterio por-caso para etiquetar train/val (de `material/kit/EJEMPLOS.md`,
sin categorías nuevas): fiel (0) solo si TODA afirmación está respaldada
por las fuentes en conjunto; basta una sin respaldo para alucinación (1);
tipos: distorsión, contradicción, no_respaldado, sobregeneralización.
