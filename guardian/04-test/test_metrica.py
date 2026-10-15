"""Tests de la metrica oficial (REQ-08). Solo libreria estandar (unittest).

Valores calculados a mano a partir de:
    puntuacion = 1 - (4*FN + FP) / n_fieles, corte score >= 0.5.
    FN = etiqueta 1 (alucinacion) predicha 0; FP = etiqueta 0 (fiel) predicha 1.
    n_fieles = casos con etiqueta 0.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from evaluar_local import auroc, comparar, puntuacion


class TestPuntuacion(unittest.TestCase):
    def test_sin_errores(self):
        # FN=0, FP=0, n_fieles=2 -> 1.0
        self.assertEqual(puntuacion([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)

    def test_solo_fp(self):
        # FN=0, FP=1, n_fieles=4 -> 1 - 1/4 = 0.75
        self.assertAlmostEqual(puntuacion([0, 0, 0, 0], [1, 0, 0, 0]), 0.75)

    def test_solo_fn(self):
        # FN=1, FP=0, n_fieles=2 -> 1 - 4/2 = -1.0 (el FN pesa 4)
        self.assertAlmostEqual(puntuacion([0, 0, 1], [0, 0, 0]), -1.0)

    def test_todo_alucinacion_da_cero(self):
        # FN=0, FP=2, n_fieles=2 -> 1 - 2/2 = 0.0
        self.assertAlmostEqual(puntuacion([0, 0, 1, 1], [1, 1, 1, 1]), 0.0)

    def test_sin_fieles(self):
        # Rama n_fieles == 0: 1 - FN/n_pos = 1 - 2/3
        self.assertAlmostEqual(puntuacion([1, 1, 1], [1, 0, 0]), 1.0 - 2.0 / 3.0)

    def test_vacio(self):
        # Sin casos: definido como 1.0 (ver METRICA.md)
        self.assertEqual(puntuacion([], []), 1.0)


class TestAuroc(unittest.TestCase):
    def test_perfecto(self):
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_empate_total(self):
        self.assertAlmostEqual(auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_una_clase(self):
        # Sin las dos clases no hay ordenacion posible: definido 0.5
        self.assertEqual(auroc([1, 1], [0.9, 0.1]), 0.5)
        self.assertEqual(auroc([0, 0], [0.9, 0.1]), 0.5)


class TestComparar(unittest.TestCase):
    def test_gana_mayor_puntuacion(self):
        a = {"puntuacion": 0.6, "auroc": 0.5}
        b = {"puntuacion": 0.4, "auroc": 0.9}
        self.assertEqual(comparar(a, b), 1)
        self.assertEqual(comparar(b, a), -1)

    def test_empate_resuelto_por_auroc(self):
        a = {"puntuacion": 0.5, "auroc": 0.8}
        b = {"puntuacion": 0.5, "auroc": 0.6}
        self.assertEqual(comparar(a, b), 1)
        self.assertEqual(comparar(b, a), -1)

    def test_empate_total(self):
        a = {"puntuacion": 0.5, "auroc": 0.7}
        self.assertEqual(comparar(a, dict(a)), 0)


if __name__ == "__main__":
    unittest.main()
