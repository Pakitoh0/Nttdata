# El Guardián — índice del proyecto

Detector de alucinaciones: clase `Guardian` con `score(caso) -> float` en
[0, 1]; corte 0.5. Métrica `1 - (4·FN + FP) / n_fieles` + AUROC de desempate.

## Estructura

- `00-contexto/` — reto y reglas (`RESUMEN.md`, `REGLAS.md`). Estado: completada (lectura).
- `01-spec/` — especificación (`ESPEC.md` + plantilla `SPECIFICATIONS.md`). Estado: cerrada (decisiones ANEXO A en `01-spec/DECISIONES.md`, checklist 6/6 cubierto en `ESPEC.md`).
- `02-plan/` — plan técnico (`ARCHITECTURE.md`, `task.template.md`, `memory.template.md`). Estado: siguiente (fundamento matemático en `02-plan/FUNDAMENTO.md`).
- `03-build/` — código (`ejemplo_bot.py`, `mi_bot.py`, `src/`, utilidades). Estado: pendiente.
- `04-test/` — evaluación y análisis de errores por versión. Estado: pendiente.
- `05-simplify/` — limpieza y refactor (`REVIEW.md`). Estado: hecha (camino final baseline + alineación; podas en `05-simplify/NOTAS.md`).
- `06-ship/` — entrega final (`mi_bot.py`, `predicciones.csv`, notas del pitch). Estado: hecha (modelo final híbrido + `PITCH.md`, ver `06-ship/NOTA_ENTREGA.md`).
- `datos/` — copia de trabajo de `train.jsonl` y `test_publico.jsonl` (no editar originales).
- `docs/` — documentación para el jurado (`JURADO.md`).

Estructura creada: completada.

## Ciclo de trabajo

Spec → Plan → Build → Test → Simplify → Ship. Cada cambio en `03-build/` se
mide en `04-test/` antes de simplificar (`05-simplify/`) y publicar (`06-ship/`).

## Cómo medir

Desde `guardian/03-build/`:

```
python evaluar_local.py --bot ejemplo_bot.py --datos ../datos/train.jsonl
python evaluar_local.py --bot mi_bot.py --datos ../datos/train.jsonl --prefijo H
```

Calibrar y validar sobre todo con `--prefijo H` (casos parecidos al test) y sin
entrenar sobre los mismos casos medidos.

## Próxima acción

Completar el plan técnico en `02-plan/` (a partir de `FUNDAMENTO.md`) antes
de escribir `mi_bot.py`. Pendiente formal: dar de alta REQ-10 (enfoque híbrido)
en `01-spec/ESPEC.md` para trazar la decisión de `DECISIONES.md`.
