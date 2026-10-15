# CRITERIO GATE — Aceptación de features (REQ-10)

## REGLA GATE (redacción exacta)

> Una feature (o grupo de features) ENTRA solo si, sobre los mismos
> casos, su Δ AUROC tiene el IC 95 % claramente > 0 Y su Δ pérdida con
> mejor corte tiene el IC 95 % claramente > 0, O si arregla un tipo
> concreto de error de EJEMPLOS.md con evidencia caso por caso; en
> cualquier otro caso la feature SE QUITA en Simplify.

## Definiciones

- Δ AUROC = AUROC(nuevo) − AUROC(base). Positivo = mejor ordenación.
- Δ pérdida = pérdida(base) − pérdida(nuevo), con
  pérdida = min_t [4·FN(t) + FP(t)] / N_f (B.16; N_f = n.º de fieles;
  borde N_f == 0 en `perdida_mejor_corte`). El signo se define así
  (base − nuevo) para que el gate pida IC > 0 en ambas: positivo =
  menos pérdida, mejor.
- «IC 95 % claramente > 0» = el límite inferior del intervalo supera 0
  (no basta la estimación puntual).
- Intervalo: bootstrap PAREADO por casos, 2000 remuestreos por defecto
  (percentil 2.5/97.5): cada remuestreo sortea n índices con reposición
  y evalúa ambos sistemas sobre los mismos, conservando la correlación
  por caso. Implementación: `perdida_mejor_corte` y `comparar_scores`
  en `03-build/validacion.py` (solo librería estándar; AUROC por rangos
  medios, idéntico a `auroc` de `evaluar_local.py`, en O(n log n) para
  que 2000 remuestreos sobre H corran en segundos).
- Vía alternativa: arreglo de un tipo concreto de `EJEMPLOS.md`
  (distorsión, contradicción, no respaldado, sobregeneralización, aguja
  en el pajar, atribución cruzada, contradicción sin «no», combinación
  de fuentes) demostrado caso por caso, no en agregado.

## Estado del grupo cifras: CONDICIONADO

Conservado pero NO aprobado. Aislado en `03-build/cifras.py::feats_num`
(f_num, f_deriv, f_year: años fuera del cierre, 100·a/b solo con
«%»/«por ciento», f_deriv con signo por aprender); fuera de la regla a
mano (`03-build/reglas.py` vuelve al baseline PASO 1 puro). Entrará
solo por regresión logística futura si supera este gate en ablación;
si no, se quita en Simplify.
