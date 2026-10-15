"""Tests de validacion por grupos de fuentes + medicion en H (REQ-09)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from validacion import (
    comparar_scores,
    es_h,
    evaluar_folds,
    folds_por_grupo,
    grupo_fuentes,
    grupos_de,
    normalizar_fuente,
    perdida_mejor_corte,
    verificar_disjuncion,
)

DATOS = os.path.join(os.path.dirname(__file__), "..", "datos", "train.jsonl")


def cargar_train():
    with open(DATOS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestNormalizacion(unittest.TestCase):
    def test_exacta(self):
        # strip + colapso de blancos + casefold
        self.assertEqual(normalizar_fuente("  Hola   MUNDO\n\tfin  "), "hola mundo fin")

    def test_grupo_estable(self):
        a = {"fuentes": ["  El Maraton  recorre 42 KM ", "Segunda"]}
        b = {"fuentes": ["el maraton recorre 42 km", "segunda"]}
        self.assertEqual(grupo_fuentes(a), grupo_fuentes(b))

    def test_grupo_distinto(self):
        a = {"fuentes": ["texto uno"]}
        b = {"fuentes": ["texto dos"]}
        self.assertNotEqual(grupo_fuentes(a), grupo_fuentes(b))


class TestFolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.casos = cargar_train()
        cls.folds = folds_por_grupo(cls.casos)

    def test_cinco_folds(self):
        self.assertEqual(len(self.folds), 5)

    def test_hashes_disjuntos_todos_los_folds(self):
        self.assertTrue(verificar_disjuncion(self.casos, self.folds))

    def test_verificador_detecta_solape(self):
        n = len(self.casos)
        with self.assertRaises(ValueError):
            verificar_disjuncion(self.casos, [(list(range(n)), list(range(n)))])

    def test_ningun_h_en_entreno_con_politica(self):
        for tr_idx, _ in self.folds:
            filtrado = [j for j in tr_idx if not es_h(self.casos[j])]
            self.assertFalse(any(es_h(self.casos[j]) for j in filtrado))
            # La politica tiene efecto real: hay H en train sin filtrar
            self.assertTrue(any(es_h(self.casos[j]) for j in tr_idx))


class TestEvaluacionH(unittest.TestCase):
    def test_evaluacion_real_sobre_h(self):
        casos = cargar_train()
        folds = folds_por_grupo(casos)
        informe = evaluar_folds(casos, n_splits=5, excluir_h_entreno=True)
        self.assertEqual(len(informe), 5)
        for (tr_idx, va_idx), f in zip(folds, informe):
            # La politica se aplica dentro de evaluar_folds: el entreno
            # usado no contiene ningun H aunque el fold bruto si los trae
            self.assertEqual(
                f["n_train"], sum(1 for j in tr_idx if not es_h(casos[j]))
            )
            self.assertEqual(f["n_val"], len(va_idx))
        total_h = sum(f["n_h_val"] for f in informe)
        # Todo H de validacion se mide exactamente una vez por ronda completa
        self.assertEqual(total_h, sum(1 for c in casos if es_h(c)))
        for f in informe:
            self.assertGreater(f["n_h_val"], 0)
            self.assertIsNotNone(f["puntuacion_h"])
            self.assertIsNotNone(f["auroc_h"])
            self.assertGreaterEqual(f["auroc_h"], 0.0)
            self.assertLessEqual(f["auroc_h"], 1.0)
        # Evidencia impresa para el informe (numeros reales, no inventados)
        print()
        for i, f in enumerate(informe):
            print(
                "fold %d: train=%d val=%d h_val=%d punt=%.3f auroc=%.3f "
                "punt_h=%.3f auroc_h=%.3f"
                % (i, f["n_train"], f["n_val"], f["n_h_val"], f["puntuacion"],
                   f["auroc"], f["puntuacion_h"], f["auroc_h"])
            )


class TestCompararScores(unittest.TestCase):
    def test_perdida_mejor_corte_manual(self):
        # Separación perfecta: existe un corte con pérdida 0.
        self.assertAlmostEqual(
            perdida_mejor_corte([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.0)
        # Sin separación (todos empatan): mejor corte = todo a un lado.
        # Todo-1: FN=0, FP=2, N_f=2 -> 1.0; todo-0: FN=2, FP=0 -> 4.0.
        self.assertAlmostEqual(
            perdida_mejor_corte([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]), 1.0)

    def test_perdida_mejor_corte_intermedio(self):
        # y=[0,1,0,1], s=[0.1,0.2,0.8,0.9]: corte t=0.2 da FN=0, FP=1,
        # N_f=2 -> 0.5, mejor que todo-1 (1.0) y que todo-0 (4.0).
        self.assertAlmostEqual(
            perdida_mejor_corte([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9]), 0.5)

    def test_delta_auroc_coincide_con_auroc(self):
        from evaluar_local import auroc

        y = [0, 0, 1, 1, 0, 1]
        a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.1, 0.4, 0.35, 0.8, 0.6, 0.55]
        r = comparar_scores(y, a, b, n_boot=50, semilla=3)
        self.assertAlmostEqual(r["delta_auroc"], auroc(y, b) - auroc(y, a))

    def test_delta_positivo_con_mejora(self):
        y = [0, 0, 1, 1]
        base = [0.5, 0.5, 0.5, 0.5]
        nuevo = [0.1, 0.2, 0.8, 0.9]
        r = comparar_scores(y, base, nuevo, n_boot=200, semilla=7)
        self.assertGreater(r["delta_auroc"], 0.0)
        # El nuevo separa perfecto (pérdida 0) y la base no.
        self.assertGreater(r["delta_perdida"], 0.0)

    def test_identicos_dan_delta_cero_e_ic_cero(self):
        y = [0, 0, 1, 1, 0, 1]
        s = [0.1, 0.4, 0.35, 0.8, 0.6, 0.55]
        r = comparar_scores(y, s, s, n_boot=200, semilla=1)
        self.assertAlmostEqual(r["delta_auroc"], 0.0)
        self.assertAlmostEqual(r["delta_perdida"], 0.0)
        self.assertEqual(r["ic_auroc"], (0.0, 0.0))
        self.assertEqual(r["ic_perdida"], (0.0, 0.0))

    def test_longitudes_distintas_fallan(self):
        with self.assertRaises(ValueError):
            comparar_scores([0, 1], [0.2], [0.3, 0.8])


if __name__ == "__main__":
    unittest.main()
