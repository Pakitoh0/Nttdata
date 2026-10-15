# PITCH — El Guardián (modelo final)

## Método (3 líneas)

1. Baseline reglas: cada oración combina NLI conjunto (`respaldo_todas`/`contradiccion_todas`) con el juez LLM en escala logit, y la respuesta agrega por máximo + noisy-OR + media de los dos peores (la peor afirmación manda) con reescalado B.2 al corte 0,5.
2. Alineación afirmación↔evidencia (`cobertura, dispersion, margen_min, f_dos_peores`): mide si cada trozo tiene respaldo en alguna fuente y si las fuentes se usan de forma dispersa o redundante.
3. Regresión logística L2 (C = 1,0) combina las 5 señales; entrenada solo sobre no-H para que H quede held-out; `mi_bot.py` la embebe como literales con solo stdlib.

## Cuándo desconfía del juez / del NLI

- Juez con veredicto null o confianza ilegible → 0,5 neutral (no vota).
- Conflicto NLI↔juez → media en logit: ninguno decide solo; la evidencia contraria prevalece sobre un juez seguro pero contradicho por el NLI.
- NLI con respaldo bajo Y contradicción baja → sospecha (falta de evidencia, no fidelidad): `q_nli = max(1 − respaldo, contradicción)`.

## Dónde falla (FN cifras / atribución)

El gate de ablación dejó fuera cifras y marcadores (Δ ≈ 0 o negativo): con ~600 H no hay señal estable para distinguir una cifra inventada de una derivada legítima ni para juzgar atribución entidad↔dato. Resultado: FN 68 en H — alucinaciones de números/fechas/atribuciones con NLI alto se escapan. Los FP (162) son el precio del coste 4:1: ante duda, marcar.

## Límites + generalización a guardián real

- Límites: solo ve `senales` (NLI + juez), no el texto de las fuentes; ciego a cifras/atribución fina; sobreconfianza arriba (ECE 0,23, Platt documentada pero no adoptada a corte 0,5); giro técnico/legal pendiente de medir (meta caída ≤ 0,10).
- Generalización: el camino es portable — mismas 5 features + literales, sin modelos en ejecución; para un guardián real añadir verificación propia de cifras contra fuentes (extracción + comparación exacta) y atribución entidad-dato, cada una con su gate antes de entrar.

## Metas H vs logrado

| Meta H (REQ-10) | Logrado (modelo embarcado, H held-out) |
| --- | --- |
| mínimo ≥ 0,00 | −0,299 (no alcanza en absoluto) |
| objetivo ≥ 0,25 | no alcanza |
| estirado ≥ 0,40 | no alcanza |
| caída en giro ≤ 0,10 | pendiente (sin giro_publico) |
| superar reglas por > 1 SE (≈0,08) | SÍ: +0,150 (−0,299 vs −0,449), Δpérdida IC95 > 0 |

Veredicto REQ-10: CUMPLE por mejora sobre reglas (OOF paso 7: −0,308 / AUROC 0,749 / FN 69 / FP 161; embarcado: −0,299 / 0,751 / FN 68 / FP 162). Honesto: el detector ordena bien (AUROC 0,75) pero el coste 4:1 lo mantiene bajo cero; el valor frente a reglas es +0,15 de puntuación.
