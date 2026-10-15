"""Tests de la logistica hibrida (PASO 6, REQ-10)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from modelo import (
    COLUMNAS,
    VARIANTES,
    ajustar,
    ablacion_en_h,
    construir_X,
    columnas_de,
    elegir_C,
    extraer_vector,
    puntuar_vector,
    reescalar,
)
from validacion import (
    es_h,
    folds_por_grupo,
    grupos_de,
    verificar_disjuncion,
)

DATOS = os.path.join(os.path.dirname(__file__), "..", "datos", "train.jsonl")


def cargar_train():
    with open(DATOS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestColumnas(unittest.TestCase):
    def test_nueve_columnas_condicionadas(self):
        self.assertEqual(list(COLUMNAS),
                         ["f_num", "f_year", "f_hedge", "f_caus",
                          "cobertura", "dispersion", "margen_min",
                          "f_dos_peores", "score_base"])
        # f_deriv, f_alc y flip quedan fuera a proposito.
        for excluida in ("f_deriv", "f_alc", "flip"):
            self.assertNotIn(excluida, COLUMNAS)

    def test_variantes_quitan_un_grupo(self):
        self.assertEqual(columnas_de("completo"), list(range(9)))
        self.assertEqual(len(columnas_de("sin_cifras")), 7)
        self.assertEqual(len(columnas_de("sin_marcadores")), 7)
        self.assertEqual(len(columnas_de("sin_alineacion")), 5)
        # Sin cifras quita f_num y f_year (indices 0 y 1).
        self.assertNotIn(0, columnas_de("sin_cifras"))
        self.assertNotIn(1, columnas_de("sin_cifras"))
        with self.assertRaises(ValueError):
            columnas_de("sin_lo_que_sea")

    def test_vector_real_nueve_finitos(self):
        casos = cargar_train()
        for c in casos[:20]:
            v = extraer_vector(c)
            self.assertEqual(len(v), 9)
            for x in v:
                self.assertTrue(-1e9 < x < 1e9)
        # Ilegible no lanza y cae al fallback en la base.
        from reglas import FALLBACK
        self.assertEqual(extraer_vector(None)[-1], FALLBACK)


class TestReescalado(unittest.TestCase):
    def test_corte_bayes_a_operativo(self):
        # B.2: p = 0,2 -> s = 0,5 exacto; s > 0,5 <=> p > 0,2.
        self.assertAlmostEqual(reescalar(0.2), 0.5, places=12)
        self.assertLess(reescalar(0.19), 0.5)
        self.assertGreater(reescalar(0.21), 0.5)

    def test_monotono_y_acotado(self):
        prev = -1.0
        for k in range(101):
            s = reescalar(k / 100.0)
            self.assertGreaterEqual(s, prev)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
            prev = s
        self.assertAlmostEqual(reescalar(0.0), 0.0)
        self.assertAlmostEqual(reescalar(1.0), 1.0)


class TestAjuste(unittest.TestCase):
    def test_overfit_sintetico_separable(self):
        # Dos nubes separables: la logistica las clava en entreno.
        X = [[float(i), 0.0] + [0.0] * 7 for i in range(10)]
        X += [[float(i), 5.0] + [0.0] * 7 for i in range(10)]
        y = [0] * 10 + [1] * 10
        params = ajustar(X, y, 1.0)
        pred = [1 if puntuar_vector(params, v) >= 0.5 else 0 for v in X]
        self.assertEqual(pred, y)

    def test_mu_sigma_embebidos_reproducen_score(self):
        # El score con mu/sigma embebidos == scaler + modelo + reescalado.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        casos = cargar_train()[:60]
        X, y = construir_X(casos)
        params = ajustar(X, y, 1.0)
        import numpy as np
        Xa = np.array(X)
        sc = StandardScaler().fit(Xa)
        mod = LogisticRegression(C=1.0, max_iter=5000)
        mod.fit(sc.transform(Xa), np.array(y))
        for v in X[:10]:
            z = sc.transform([v])[0]
            p = float(mod.predict_proba([z])[0][1])
            self.assertAlmostEqual(puntuar_vector(params, v), reescalar(p),
                                   places=9)

    def test_elegir_c_no_toca_validacion(self):
        # La C elegida solo depende del train que se le pasa.
        casos = cargar_train()[:120]
        X, y = construir_X(casos)
        grupos = grupos_de(casos)
        c1 = elegir_C(X[:100], y[:100], grupos[:100])
        c2 = elegir_C(X[:100], y[:100], grupos[:100])
        self.assertEqual(c1, c2)
        self.assertIn(c1, (0.01, 0.1, 1.0, 10.0))


class TestSinFuga(unittest.TestCase):
    def test_mismo_groupkfold_sin_solape(self):
        # Reusa folds_por_grupo/verificar_disjuncion de validacion.py:
        # ningun grupo en entreno y validacion a la vez.
        casos = cargar_train()
        folds = folds_por_grupo(casos)
        self.assertEqual(len(folds), 5)
        self.assertTrue(verificar_disjuncion(casos, folds))

    def test_h_medido_una_vez(self):
        casos = cargar_train()
        folds = folds_por_grupo(casos)
        total_h = sum(sum(1 for j in va if es_h(casos[j])) for _, va in folds)
        self.assertEqual(total_h, sum(1 for c in casos if es_h(c)))


class TestOOF(unittest.TestCase):
    def test_oof_rapido_una_variante(self):
        # Humo sobre pocos casos: formas y rangos, sin fuga de H al entreno.
        casos = [c for c in cargar_train() if not es_h(c)][:40]
        casos += [c for c in cargar_train() if es_h(c)][:10]
        res, comp = ablacion_en_h(casos, n_boot=50, semilla=1)
        for v in VARIANTES:
            self.assertEqual(len(res["oof"][v]), len(casos))
            for s in res["oof"][v]:
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)
        for g in ("cifras", "marcadores", "alineacion", "REQ-10"):
            self.assertIn("delta_auroc", comp[g])
            self.assertIn("ic_auroc", comp[g])


if __name__ == "__main__":
    unittest.main()


class TestECE(unittest.TestCase):
    def test_perfecto_cero(self):
        from modelo import ece
        # Prediccion perfecta en extremos: cada bin con casos tiene
        # acc == conf exactas -> ECE == 0.
        self.assertAlmostEqual(
            ece([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]), 0.0, places=9)

    def test_sobreconfianza_alta(self):
        from modelo import ece
        # Tasa base 0,5 con todo a 0,9: |0,5-0,9| = 0,4.
        y = [0] * 10 + [1] * 10
        self.assertAlmostEqual(ece(y, [0.9] * 20), 0.4, places=9)

    def test_bins_vacios_y_rangos(self):
        from modelo import ece
        # Bins vacios aportan 0; el resultado vive en [0, 1].
        v = ece([0, 1], [0.05, 0.95])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)
        self.assertAlmostEqual(ece([], []), 0.0)


class TestPlatt(unittest.TestCase):
    def test_monotona(self):
        from modelo import ajustar_platt, aplicar_platt
        s_val = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
        y_val = [0, 0, 0, 1, 1, 1, 1]
        a, b = ajustar_platt(s_val, y_val)
        # A mayor score, mayor tasa: la pendiente sale positiva.
        self.assertGreater(a, 0.0)
        cal = aplicar_platt(s_val, a, b)
        for antes, despues in zip(cal, cal[1:]):
            self.assertLessEqual(antes, despues)

    def test_extremos_finitos(self):
        from modelo import ajustar_platt, aplicar_platt
        import math
        a, b = ajustar_platt([0.2, 0.8], [0, 1])
        for s in aplicar_platt([0.0, 0.5, 1.0], a, b):
            # Saturacion a 0/1 en extremos: finita y en [0, 1].
            self.assertTrue(math.isfinite(s))
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_calibracion_baja_ece_sintetica(self):
        from modelo import aplicar_platt, ece
        # Sobreconfianza sistematica: todo a 0,9 con tasa 0,5.
        # Platt ajustada ahi (a < 1) acerca confianza y tasa.
        y = [0] * 10 + [1] * 10
        s = [0.9] * 20
        from modelo import ajustar_platt
        a, b = ajustar_platt(s, y)
        self.assertLess(ece(y, aplicar_platt(s, a, b)), ece(y, s))


class TestConfigFinal(unittest.TestCase):
    def test_columnas_final_baseline_mas_alineacion(self):
        from modelo import COLUMNAS, COLUMNAS_FINAL, IDX_FINAL
        self.assertEqual(list(COLUMNAS_FINAL),
                         ["cobertura", "dispersion", "margen_min",
                          "f_dos_peores", "score_base"])
        # Los indices apuntan a los mismos nombres en COLUMNAS (9).
        self.assertEqual([COLUMNAS[i] for i in IDX_FINAL],
                         list(COLUMNAS_FINAL))
        # Cifras (f_num, f_year) y marcadores (f_hedge, f_caus) fuera.
        for fuera in ("f_num", "f_year", "f_hedge", "f_caus"):
            self.assertNotIn(fuera, COLUMNAS_FINAL)

    def test_vector_final_consistente(self):
        from modelo import IDX_FINAL, extraer_vector, extraer_vector_final
        casos = cargar_train()
        for c in casos[:20]:
            v = extraer_vector(c)
            self.assertEqual(extraer_vector_final(c),
                             [v[i] for i in IDX_FINAL])
        v = extraer_vector_final(None)
        self.assertEqual(len(v), 5)
        for x in v:
            self.assertTrue(-1e9 < x < 1e9)

    def test_oof_final_humo(self):
        from modelo import oof_final
        casos = [c for c in cargar_train() if not es_h(c)][:40]
        casos += [c for c in cargar_train() if es_h(c)][:10]
        res = oof_final(casos)
        self.assertEqual(len(res["scores"]), len(casos))
        for s in res["scores"]:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        for c in res["C_elegidas"].values():
            self.assertIn(c, (0.01, 0.1, 1.0, 10.0))

    def test_reproduce_numero_reportado(self):
        # Reentreno limpio OOF en H (resultados_paso7.md): la
        # configuracion final reproduce el numero reportado.
        from modelo import oof_final
        from evaluar_local import auroc, puntuacion
        casos = cargar_train()
        res = oof_final(casos)
        idx_h = [i for i, c in enumerate(casos) if es_h(c)]
        y_h = [res["y"][i] for i in idx_h]
        s_h = [res["scores"][i] for i in idx_h]
        pred = [1 if s >= 0.5 else 0 for s in s_h]
        fn = sum(1 for t, p in zip(y_h, pred) if t == 1 and p == 0)
        fp = sum(1 for t, p in zip(y_h, pred) if t == 0 and p == 1)
        self.assertAlmostEqual(puntuacion(y_h, pred), -0.308, places=2)
        self.assertAlmostEqual(auroc(y_h, s_h), 0.749, places=2)
        self.assertEqual((fn, fp), (69, 161))
