"""Tests de alineacion afirmacion <-> evidencia (PASO 3, B.10)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from alineacion import TAU, alineaciones_oracion, feats_alin, puntuar_auxiliar


def _caso(oraciones):
    return {"senales": {"oraciones": oraciones}}


def _oracion(fuentes, todas):
    return {"respaldo_por_fuente": list(fuentes), "respaldo_todas": todas}


class TestAij(unittest.TestCase):
    def test_max_fuente_y_todas(self):
        # A_ij = max(fuente_j, todas), mismos nombres que reglas.py.
        self.assertEqual(
            alineaciones_oracion(_oracion([0.2, 0.9], 0.5)), [0.5, 0.9])

    def test_sin_lista_usa_todas(self):
        self.assertEqual(alineaciones_oracion({"respaldo_todas": 0.7}), [0.7])

    def test_nan_a_cero(self):
        vals = alineaciones_oracion(_oracion([float("nan")], float("nan")))
        self.assertEqual(vals, [0.0])


class TestCobertura(unittest.TestCase):
    def test_cobertura_total(self):
        caso = _caso([_oracion([0.9], 0.1), _oracion([0.2], 0.8)])
        cobertura, _, _, _ = feats_alin(caso)
        self.assertEqual(cobertura, 1.0)

    def test_cobertura_nula(self):
        caso = _caso([_oracion([0.1], 0.2), _oracion([0.4], 0.3)])
        cobertura, _, _, _ = feats_alin(caso)
        self.assertEqual(cobertura, 0.0)

    def test_cobertura_parcial(self):
        caso = _caso([_oracion([0.9], 0.1), _oracion([0.1], 0.2)])
        cobertura, _, _, _ = feats_alin(caso)
        self.assertEqual(cobertura, 0.5)

    def test_tau_documentado(self):
        self.assertEqual(TAU, 0.5)


class TestMargenYDispersion(unittest.TestCase):
    def test_margen_bajo_con_empate(self):
        # Dos fuentes respaldan por igual: margen 0.
        caso = _caso([_oracion([0.9, 0.9], 0.1)])
        _, _, margen_min, _ = feats_alin(caso)
        self.assertAlmostEqual(margen_min, 0.0)

    def test_margen_alto_sin_competencia(self):
        caso = _caso([_oracion([0.9, 0.1], 0.0)])
        _, _, margen_min, _ = feats_alin(caso)
        self.assertAlmostEqual(margen_min, 0.8)

    def test_dispersion_cuenta_argmax_distintos(self):
        caso = _caso([_oracion([0.9, 0.1], 0.0),
                      _oracion([0.1, 0.8], 0.0)])
        _, dispersion, _, _ = feats_alin(caso)
        self.assertEqual(dispersion, 2)

    def test_dispersion_una_fuente(self):
        caso = _caso([_oracion([0.9, 0.1], 0.0),
                      _oracion([0.8, 0.2], 0.0)])
        _, dispersion, _, _ = feats_alin(caso)
        self.assertEqual(dispersion, 1)


class TestDosPeores(unittest.TestCase):
    def test_dos_peores_vs_minimo(self):
        # m = 0.9, 0.3, 0.2: f = (0.2+0.3)/2 = 0.25, no el minimo 0.2.
        caso = _caso([_oracion([0.9], 0.0),
                      _oracion([0.3], 0.0),
                      _oracion([0.2], 0.0)])
        _, _, _, f_dos = feats_alin(caso)
        self.assertAlmostEqual(f_dos, 0.25)
        self.assertNotAlmostEqual(f_dos, 0.2)

    def test_una_oracion_f_es_m(self):
        caso = _caso([_oracion([0.4], 0.6)])
        _, _, _, f_dos = feats_alin(caso)
        self.assertAlmostEqual(f_dos, 0.6)


class TestDefaults(unittest.TestCase):
    def test_sin_senales(self):
        self.assertEqual(feats_alin({}), (0.0, 0, 0.0, 0.0))
        self.assertEqual(feats_alin({"senales": {}}), (0.0, 0, 0.0, 0.0))
        self.assertEqual(feats_alin({"senales": {"oraciones": []}}),
                         (0.0, 0, 0.0, 0.0))

    def test_caso_ilegible(self):
        self.assertEqual(feats_alin(None), (0.0, 0, 0.0, 0.0))
        self.assertEqual(feats_alin("texto"), (0.0, 0, 0.0, 0.0))

    def test_sin_nan_ni_inf(self):
        for caso in ({}, {"senales": {"oraciones": [
                _oracion([float("nan"), float("inf")], float("-inf"))]}}):
            for v in feats_alin(caso):
                self.assertTrue(math.isfinite(float(v)))

    def test_auxiliar_en_rango(self):
        caso = _caso([_oracion([0.9, 0.9], 0.1)])
        s = puntuar_auxiliar(caso)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)
        # f_dos = 0.9 -> score 0.1 (poco sospechoso).
        self.assertAlmostEqual(s, 0.1)


if __name__ == "__main__":
    unittest.main()
