import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def extraer_features(caso):
    senales = caso.get("senales", {})
    oraciones = senales.get("oraciones", [])
    fragmentos = senales.get("fragmentos", [])
    
    # Cobertura léxica (Tokens / Shingles básicos)
    resp_texto = caso.get("respuesta", "")
    fuentes_texto = " ".join(caso.get("fuentes", []))
    r_words = set(w.lower() for w in resp_texto.split() if len(w) > 3)
    f_words = set(w.lower() for w in fuentes_texto.split() if len(w) > 3)
    overlap = len(r_words.intersection(f_words)) / max(len(r_words), 1)

    if not oraciones:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, overlap]

    contra_vals = []
    resp_todas_vals = []
    conf_llm_vals = []
    similitudes = []
    resp_frag = []

    for o in oraciones:
        contras = o.get("contradiccion_por_fuente", [0])
        if contras:
            contra_vals.extend([c if c is not None else 0.0 for c in contras])
        resp = o.get("respaldo_todas", 1.0)
        resp_todas_vals.append(resp if resp is not None else 1.0)
        llm = o.get("llm", {})
        conf = llm.get("confianza", 0.0)
        conf_llm_vals.append(conf if conf is not None else 0.0)
        sims = o.get("similitud_por_fuente", [0])
        if sims:
            similitudes.extend([s if s is not None else 0.0 for s in sims])

    for f in fragmentos:
        r_f = f.get("respaldo_todas", 1.0)
        resp_frag.append(r_f if r_f is not None else 1.0)

    max_contra = max(contra_vals) if contra_vals else 0.0
    mean_contra = float(np.mean(contra_vals)) if contra_vals else 0.0
    min_resp = min(resp_todas_vals) if resp_todas_vals else 1.0
    max_conf = max(conf_llm_vals) if conf_llm_vals else 0.0
    min_sim = min(similitudes) if similitudes else 0.0
    min_resp_frag = min(resp_frag) if resp_frag else min_resp
    
    brecha_frag = float(min_resp - min_resp_frag)
    num_fragmentos = len(fragmentos) if fragmentos else 0
    num_oraciones = len(oraciones) if oraciones else 0

    val_conf = float(max_conf) if max_conf is not None else 0.0
    val_contra = float(max_contra) if max_contra is not None else 0.0
    interaccion = val_conf * val_contra
    flag_contra = 1.0 if val_contra > 0.5 else 0.0
    divergencia_juez = float(abs(val_conf - min_resp))

    return [
        float(val_contra),
        mean_contra,
        float(min_resp),
        float(max_conf),
        float(min_sim),
        float(min_resp_frag),
        float(num_fragmentos),
        float(num_oraciones),
        float(flag_contra),
        float(interaccion),
        float(brecha_frag),
        float(divergencia_juez),
        float(overlap)
    ]

def main():
    print("Leyendo nuevosDatos.jsonl...")
    casos = [json.loads(l) for l in open("train.jsonl", encoding="utf-8") if l.strip()]
    
    X = np.array([extraer_features(c) for c in casos])
    y = np.array([c.get("etiqueta", 0) for c in casos])

    print(f"Entrenando modelo con {len(casos)} casos...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight={0: 1.0, 1: 4.0},
        random_state=42
    )
    modelo.fit(X, y)

    # Control de seguridad si el dataset tiene una sola clase
    if len(modelo.classes_) < 2:
        print("\n[AVISO] El archivo analizado contiene una sola clase de etiquetas (ej. solo fieles o solo alucinaciones).")
        preds = np.zeros(len(y))
    else:
        preds = modelo.predict_proba(X)[:, 1]
        print(f"AUROC obtenido en entrenamiento: {roc_auc_score(y, preds):.3f}")

    # Búsqueda del mejor umbral de corte local para maximizar la puntuación
    mejor_puntuacion = -999.0
    mejor_umbral = 0.5
    
    for t in np.linspace(0.1, 0.99, 100):
        pred_bin = [1 if p >= t else 0 for p in preds]
        fn = sum(1 for true, pred in zip(y, pred_bin) if true == 1 and pred == 0)
        fp = sum(1 for true, pred in zip(y, pred_bin) if true == 0 and pred == 1)
        n_neg = sum(1 for true in y if true == 0)
        puntuacion = 1.0 - (4.0 * fn + fp) / n_neg if n_neg > 0 else 0.0
        
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_umbral = t

    print("\n" + "="*50)
    print(f"¡RESULTADO ÓPTIMO ENCONTRADO!")
    print(f"Mejor umbral a pasar a mano (umbral_optimo): {mejor_umbral:.3f}")
    print(f"Puntuación estimada en train: {mejor_puntuacion:.3f}")
    print("="*50)
    
    importancias = modelo.feature_importances_
    nombres = ['val_contra', 'mean_contra', 'min_resp', 'max_conf', 'min_sim', 'min_resp_frag', 
               'num_frag', 'num_oraciones', 'flag_contra', 'interaccion', 'brecha_frag', 'div_juez', 'overlap']
    print("\nImportancia de características:")
    for nombre, imp in sorted(zip(nombres, importancias), key=lambda x: x[1], reverse=True):
        print(f"  - {nombre}: {imp:.3f}")

if __name__ == "__main__":
    main()