# ANEXO B — Fundamento matemático (copia literal del orquestador)

> **Aviso:** estas fórmulas tratan las señales como cantidades abstractas. Antes de usarlas, **comprueba en los datos** los nombres exactos de las claves de `senales`, su escala (0-1) y si la `confianza` del juez es la del veredicto que da.
> **Regla de oro:** una feature solo entra si mejora el score en H por encima del error estándar (≈0,08). Si no, es ruido y resta robustez al giro.

Notación: fuentes S₁…Sₘ; respuesta troceada en afirmaciones (oraciones y fragmentos) c₁…cₙ; etiqueta y=1 si alucina, y=0 si es fiel.

## B.1 La métrica y qué implica

```
Score = 1 − (4·FN + FP) / N_f        N_f = nº de respuestas fieles
r = N_h / N_f,   TPR = 1 − FN/N_h,   FPR = FP/N_f
Score = (1 − 4r) + 4r·TPR − FPR
```

- Maximizar el score equivale a minimizar L = 4·FN + FP.
- Marcarlo todo como alucinación da TPR = FPR = 1 y score 0.
- El punto óptimo está donde la pendiente de la ROC es 1/(4r). Con r ≈ 0,66 (ejemplo del enunciado) la pendiente es ≈ 0,38: zona agresiva.

## B.2 Corte óptimo (Bayes) y reescalado a 0,5

Con p = P(y=1|x) calibrada, marcar alucinación si el coste esperado de dejarla pasar supera al de alarmar:

```
4·p > 1·(1−p)  ⟺  p > 1/5 = 0,2  ⟺  logit(p) > −ln 4 ≈ −1,386
```

Reescalado monótono (no cambia el AUROC):

```
s = σ( logit(p) + ln 4 ) = 4p / (1 + 3p)        s > 0,5 ⟺ p > 0,2        (p = 0,2 → s = 0,5 ✓)
```

- Alternativa equivalente: entrenar con peso 4 en la clase positiva (posterior p' = 4p/(1+3p)). **Usar una vía, no las dos**: se compondría el desplazamiento.
- Cambio de prevalencia (π_te vs π_tr), estimado con los casos H:

```
Δ = ln[π_te/(1−π_te)] − ln[π_tr/(1−π_tr)]        (se suma al logit)
```

## B.3 Del trozo a la respuesta: "una sola falsa basta"

Sea qᵢ la probabilidad de que la afirmación cᵢ no esté respaldada:

```
Máximo:      M = max_i qᵢ
Noisy-OR:    P = 1 − Π_i (1 − qᵢ) = 1 − exp( Σ_i ln(1 − qᵢ) )
Soft-max:    LSE_β = (1/β)·ln Σ_i exp(β·qᵢ)        (β→∞: máximo; β→0: media)
Media gen.:  ( (1/n)·Σ_i qᵢᵖ )^(1/p),   p ≥ 2
```

- Noisy-OR asume fallos independientes, crece con n y compone errores de calibración. El máximo no sufre sesgo de longitud pero tira información. Mete varias como features, además de n.

Respaldo (dual):

```
aᵢ = max_j r_ij                  (mejor fuente individual)
Aᵢ = respaldo_todas(cᵢ)          (todas las fuentes juntas)
mᵢ = max(aᵢ, Aᵢ)                 m = min_i mᵢ  (peor trozo)
soft-min = −(1/β)·ln Σ_i exp(−β·mᵢ)
```

## B.4 Transformación de las señales

```
ℓ(x) = ln( clip(x, ε, 1−ε) / (1 − clip(x, ε, 1−ε)) ),    ε = 1e-4
```

Juez LLM → probabilidad de "no respaldada" (veredicto v, confianza κ):

```
q_LLM = κ        si v ∈ {no_respaldada, contradicha}
q_LLM = 1 − κ    si v = respaldada
q'    = σ( a_v·ℓ(q_LLM) + b_v )        (Platt por veredicto, a_v < 1 aplasta la sobreconfianza)
```

Features de desacuerdo (enseñan al modelo cuándo desconfiar del juez):

```
d1 = ℓ(q_LLM) − ℓ(1 − aᵢ)         juez y NLI discrepan
d2 = (1 − q_LLM)·(1 − aᵢ)         "respaldada" con confianza y NLI sin respaldo
d3 = (1 − q_LLM)·contradicción_NLI
```

## B.5 Combinar expertos

Naive Bayes en logit (asume independencia, **sobrecuenta evidencia** si hay correlación):

```
logit P(y=1 | e₁…e_K) = logit(π) + Σ_k [ logit P(y=1 | e_k) − logit(π) ]
```

La regresión logística aprende pesos w_k que corrigen esa sobreconfianza; por eso se prefiere a sumar a mano.

## B.6 Comprobaciones léxicas (features)

**Cifras.** Normalizar a valor+unidad (`28.000`, `28000`, `veintiocho mil` → 28000 €; `tres` → 3; `40 %` → 0,4; `2.000 m` → 2 km). Cierre por derivación:

```
D(N_S) = N_S ∪ { a+b, |a−b|, a·b, a/b, 100·a/b, 2a, a/2 : a,b ∈ N_S }
U = { n ∈ N_R : n ∉ D(N_S) }
f_num = |U| / max(1, |N_R|)
f_deriv = nº de cifras de la respuesta que solo se explican por derivación
```

Tolerancia 0 tras normalizar; tolerancia relativa solo para redondeos declarados ("unos", "casi").

**Atribución (quién hizo qué).** Ventanas de ±k tokens alrededor de cada cifra; IDF intra-caso:

```
idf(t) = ln( (m+1) / (df(t)+1) ) + 1
attr(n) = max_o  Σ_{t∈W_R∩W_S(o)} idf(t) / Σ_{t∈W_R∪W_S(o)} idf(t)
f_attr  = min_n attr(n)
margen  = attr(mejor aparición) − attr(mejor aparición ajena)     (margen bajo → entidades confundibles)
```

**Novedad léxica** (medir antes de quedársela; el enunciado avisa de que longitud, superlativos y negaciones no predicen):

```
nov = Σ_{t ∈ R \ S_cit} idf(t) / Σ_{t ∈ R} idf(t)        (tomar el máximo por afirmación)
```

**Marcadores diferenciales** (cuentan solo si están en la respuesta y faltan en la fuente citada):

```
f_caus  = 1[∃ w∈C : w∈R ∧ w∉S_cit]    C = {gracias a, debido a, porque, provocó, a causa de, como consecuencia…}
f_alc   = 1[∃ w∈Q : w∈R ∧ w∉S_cit]    Q = {todos, siempre, cada, ningún, nunca, la totalidad…}
f_hedge = |H(S_cit) \ H(R)|            H = {podría, aproximadamente, más de, al menos, solo, hasta…}  (ventana ±3 tokens)
```

**Fuerza del verbo:**

```
f_verb = max sobre verbos alineados de  max(0, nivel(R) − nivel(S))      ("propuso" < "aprobó"; "vacunó" < "inmunizó")
```

Alinear verbos por el objeto compartido.

## B.7 Clasificador

```
p = σ(wᵀx + b)
J(w,b) = −(1/n)·Σ ωᵢ·[ yᵢ ln pᵢ + (1−yᵢ) ln(1−pᵢ) ] + (λ/2)·‖w‖²
∇_w J  = (1/n)·Xᵀ Ω (p − y) + λw
IRLS:   w ← w − ( XᵀWX + λI )⁻¹ ∇,    W = diag(pᵢ(1−pᵢ))
```

- Estandarizar con μ y σ del entrenamiento y **embeberlos** en `mi_bot.py`.
- Elegir λ (C = 1/λ en sklearn) por validación cruzada.
- NaN e infinitos → 0; todos los campos con `.get()`.

## B.8 Validación sin autoengañarse

```
Re-ponderación de dominio:   w(x) = [ g(x)/(1−g(x)) ]·( n_train / n_H ),   g(x) = P(H|x)
AUROC (Mann–Whitney):        ( Σ rank_pos − n_p(n_p+1)/2 ) / (n_p·n_n)
Ruido de la nota:            SE(Score) ≈ sqrt( 16·N_h·β(1−β) + N_f·α(1−α) ) / N_f
ECE:                         Σ_b (|B_b|/n)·| acc(B_b) − conf(B_b) |
```

- Más simple que re-ponderar: entrenar con todo y **evaluar solo sobre H**.
- Fuga entre casos: `GroupKFold` con el hash de `fuentes` como grupo.
- Con N_h = 175, N_f = 265, α = β = 0,2: SE ≈ 0,08 (≈ 0,12 en la mitad privada). **Diferencias < ~0,1 en el leaderboard público son ruido.**
- ECE alto → calibrar con Platt `σ(a·z+b)` o isotónica, ajustada en validación (no en entrenamiento).

## B.9 Robustez al giro

- Features normalizadas (ratios, máximos, mínimos, rangos dentro del caso), no conteos absolutos.
- Cambio de distribución train → giro: `PSI = Σ (pᵢ − qᵢ)·ln(pᵢ/qᵢ)`, o KS por feature. PSI > 0,25 → feature no fiable.
- Todo en `try/except`; ante un caso ilegible, devolver un score algo por encima de 0,5 (por el coste asimétrico).

## B.10 Alineación afirmación ↔ evidencia

```
C_ij = −ln( clip(A_ij, ε, 1) )
π* = argmin_π Σ_i C_{i,π(i)}          (húngaro; en la práctica basta j*(i) = argmax_j A_ij)
cobertura  = (1/n)·Σ_i 1[ max_j A_ij ≥ τ ]
dispersión = nº de fuentes distintas usadas
margen_i   = A_i(1) − A_i(2)           (margen bajo → dos fuentes respaldan por igual)
f = ( m_(1) + m_(2) ) / 2              (media de los dos peores trozos; más estable que el mínimo solo)
```

## B.11 Similitud léxica (sin IA)

```
BM25(c,S) = Σ_{t∈c} idf(t)·f(t,S)·(k₁+1) / ( f(t,S) + k₁·(1 − b + b·|S|/avgdl) ),   k₁ = 1,2   b = 0,75
            (normalizar dividiendo por BM25(c,c))
J₃(a,b)      = |G₃(a) ∩ G₃(b)| / |G₃(a) ∪ G₃(b)|          (n-gramas de caracteres, n=3, tras quitar acentos con unicodedata NFKD)
Lev_norm(a,b) = 1 − Lev(a,b) / max(|a|,|b|)
ent_cov       = Σ_{e∈E_R∩E_S} idf(e) / Σ_{e∈E_R} idf(e)     (E = tokens con mayúscula no inicial de frase)
```

## B.12 Polaridad y contradicción sin "no"

Léxico con signo σ(w) ∈ {+1, −1, 0}: `+1` aumentó, subió, creció, ganó, abrió, aprobó…; `−1` cayó, bajó, disminuyó, perdió, cerró, rechazó…

```
flip = 1[ σ(v_R)·σ(v_S) = −1  ∧  Jaccard_ponderado(objeto_R, objeto_S) ≥ 0,5 ]      (solo si hablan del mismo objeto)
viol = 1[ |ratio_R − ratio_S| / ratio_S > 0,01 ]         ("el doble", "la mitad", "mayor que"; ratio = a/b)
```

## B.13 Aritmética con redondeo declarado

```
δ(x) = 0,5·10^(−d)        (d = decimales; 12,5 → δ = 0,05; en cifras redondas, la del último dígito significativo no cero)
x compatible con g  ⟺  |x − g| ≤ δ(x)
porcentaje: |x − 100·a/b| ≤ δ(x) + ε
```

Sumas de subconjuntos hasta tamaño 3 con `itertools.combinations` (limitar |N_S| ≤ 12; coste O(C(n,3))).

## B.14 Peso de evidencia (WoE) de marcadores

```
WoE(w) = ln( (n_halu(w)+α) / (N_halu+2α) ) − ln( (n_fiel(w)+α) / (N_fiel+2α) ),    α = 1
```

`n_halu(w)` = respuestas alucinadas con w en la respuesta y **no** en la fuente citada. Quedarse con |WoE| alto y soporte n ≥ 10; la suma de WoE del caso es una feature. **Calcular dentro de cada pliegue**, nunca con todo el train (se filtraría la etiqueta).

## B.15 Modelo más expresivo (acotado)

Interacciones con el juez:

```
z1 = (1 − q_LLM)·f_caus        z2 = (1 − q_LLM)·f_alc
z3 = (1 − q_LLM)·(1 − m)       z4 = f_num·f_attr
```

- Plan B: `GradientBoostingClassifier` (profundidad 2-3, ≤ 50-100 árboles); pérdida `Σ ωᵢ·ln(1 + exp(−ỹᵢ·F(xᵢ)))`. Embeberlo exige volcar cada árbol a literales.
- Stacking con predicciones **fuera de pliegue** (OOF): `p_final = σ( β₀ + Σ_k β_k·logit(p_k^OOF) )`.
- Bagging (B = 30 logísticas con bootstrap) y penalización por incertidumbre, coherente con el coste 4:1:

```
z_final = μ_z + k·σ_z        k ≈ 0,5 (ajustar con CV)
```

## B.16 Optimizar lo que se puntúa

```
L_rank = Σ_{i∈P, j∈N} ln(1 + exp( −(s_i − s_j) ))          (logística sobre diferencias x_i − x_j; solo si el AUROC está bajo)
t* = argmin_t (1/K)·Σ_k [ 4·FN_k(t) + FP_k(t) ] / N_f,k       (umbral por pérdida esperada con CV)
```

- Si t* se aleja mucho de 0,2, el modelo está mal calibrado: revisar B.8.
- Calibración: Platt (2 parámetros) → isotónica solo con ≳ 1.000 puntos (con ~600 casos H queda justa) → alternativa con pocos datos, **calibración beta**: `p̂ = σ( a·ln s − b·ln(1−s) + c )`, a, b ≥ 0.

## B.17 Detección de anomalías (red de seguridad para el giro)

```
d²(x) = (x − μ_f)ᵀ Σ_f⁻¹ (x − μ_f)
Σ_f = (1−γ)·S + γ·(tr(S)/p)·I        (γ ≈ 0,1; evita S casi singular)
```

`d²` alto = "no se parece a una respuesta fiel típica": subir su score. Solo sobre features normalizadas.

## B.18 Lo que parece listo y no lo es

- Más features sin control: con ~600 casos H, p > 40 sobreajusta.
- Umbrales finos sobre el train fácil: fallan en H y en el giro.
- Fiarse del leaderboard público: la nota final usa la mitad privada y el ruido (~0,08-0,12) es del tamaño de las mejoras.

## B.19 Orden de implementación (retorno / riesgo)

1. Logit, "peor trozo" y corte 0,2 → 0,5 (B.2, B.3, B.4).
2. Cifras con cierre por derivación (B.6) + aritmética con redondeo y subconjuntos (B.13).
3. Estadísticos de orden y márgenes (B.10).
4. Marcadores de causa y alcance (B.6) y polaridad por antónimos (B.12).
5. Atribución (B.6) y features de desacuerdo con el juez (B.4).
6. WoE dentro de pliegue (B.14) e interacciones z1…z4 (B.15).
7. Bagging con penalización por incertidumbre (B.15) y calibración (B.16).
8. Detección de anomalías (B.17), solo para blindar el giro.
