"""Tests de cifras con cierre por derivacion (PASO 2, B.6 + B.13)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from cifras import (compatible, delta_para, detectar_redondeo,
                    explicar_cifra, extraer_anos, extraer_cifras,
                    feats_num, features_cifras)


def _valores(texto):
    return [(c["valor"], c["unidad"]) for c in extraer_cifras(texto)]


class TestExtraccion(unittest.TestCase):
    def test_miles_con_punto(self):
        self.assertIn((28000.0, ""), _valores("se presentaron 28.000 quesos"))

    def test_palabra_tres(self):
        self.assertIn((3.0, ""), _valores("tres queserias"))

    def test_palabra_catorce(self):
        self.assertIn((14.0, ""), _valores("14 meses de maduración"))

    def test_porcentaje_a_tanto_por_uno(self):
        self.assertIn((0.4, "%"), _valores("el 40 % de los casos"))

    def test_metros_a_km(self):
        self.assertIn((2.0, "km"), _valores("recorrió 2.000 m"))


class TestDerivacion(unittest.TestCase):
    def test_suma(self):
        f_num, _, _ = features_cifras("72 y 14 suman 86",
                                      ["hubo 72 quesos y 14 mas"])
        self.assertEqual(f_num, 0.0)

    def test_doble(self):
        f_num, f_der, _ = features_cifras("produjo 50 toneladas",
                                          ["produce unas 25 toneladas"])
        self.assertEqual(f_num, 0.0)
        self.assertGreaterEqual(f_der, 1.0)

    def test_porcentaje_derivado(self):
        f_num, _, _ = features_cifras("el 50 % aprobo",
                                      ["de 10 casos, 5 salieron bien"])
        self.assertEqual(f_num, 0.0)

    def test_sin_respaldo(self):
        f_num, _, _ = features_cifras("valoro 86 quesos de 34 queserias",
                                      ["se presentaron 86 quesos de 31 queserias"])
        self.assertGreater(f_num, 0.0)


class TestRedondeo(unittest.TestCase):
    def test_delta_decimales(self):
        self.assertAlmostEqual(delta_para(12.5, 1), 0.05)

    def test_redonda_usa_ultimo_significativo(self):
        self.assertAlmostEqual(delta_para(28000.0, 0), 500.0)

    def test_redondeo_declarado_tolera(self):
        self.assertTrue(detectar_redondeo("unos 90 casos"))
        self.assertTrue(compatible(83.0, 0, 80.0, redondeo=True))
        self.assertFalse(compatible(83.0, 0, 80.0, redondeo=False))

    def test_sin_cifras_da_cero(self):
        self.assertEqual(features_cifras("gano el primer premio",
                                         ["obtuvo el primer premio"]), (0.0, 0.0, 0.0))
        self.assertEqual(features_cifras("", ["hay 5 casos"]), (0.0, 0.0, 0.0))


class TestAnos(unittest.TestCase):
    def test_extraer_anos(self):
        self.assertEqual(extraer_anos("abrió en 1987 y cerró en 2021"),
                         {"1987", "2021"})
        self.assertEqual(extraer_anos("sin años aquí"), set())

    def test_ano_nuevo_marca_f_year(self):
        f_num, f_der, f_year = features_cifras(
            "ganó en 2021", ["ganó en 2020"])
        self.assertEqual(f_year, 1.0)
        # Los años no entran al cierre: no cuentan como cifra sin respaldo.
        self.assertEqual(f_num, 0.0)
        self.assertEqual(f_der, 0.0)

    def test_ano_coincidente_no_marca(self):
        self.assertEqual(features_cifras("ganó en 2020", ["ganó en 2020"]),
                         (0.0, 0.0, 0.0))

    def test_ano_excluido_del_cierre(self):
        # 86 respaldada + año nuevo: f_num ignora el año, f_year lo señala.
        f_num, _, f_year = features_cifras(
            "en 2021 hubo 86 casos", ["en 2020 hubo 86 casos"])
        self.assertEqual(f_num, 0.0)
        self.assertEqual(f_year, 1.0)

    def test_solo_anos_no_cuenta_cifras(self):
        self.assertEqual(features_cifras("fue en 2021", ["fue en 2020"]),
                         (0.0, 0.0, 1.0))

    def test_feats_num_aislado(self):
        # Punto de entrada para la futura logística: misma tupla, sin cableado.
        self.assertEqual(feats_num("ganó en 2021", ["ganó en 2020"]),
                         features_cifras("ganó en 2021", ["ganó en 2020"]))


class TestPorcentajeCondicionado(unittest.TestCase):
    def test_porcentaje_solo_con_marca(self):
        # 100*1/3 solo explica el 33 % con marca, no el 33 pelado.
        f_num_pct, _, _ = features_cifras(
            "el 33 % aprobó", ["de 3 casos, 1 salió bien"])
        self.assertEqual(f_num_pct, 0.0)
        f_num_sin, _, _ = features_cifras(
            "aprobó 33", ["de 3 casos, 1 salió bien"])
        self.assertGreater(f_num_sin, 0.0)

    def test_por_ciento_en_palabras_abre_derivacion(self):
        f_num, _, _ = features_cifras(
            "el 33 por ciento aprobó", ["de 3 casos, 1 salió bien"])
        self.assertEqual(f_num, 0.0)


if __name__ == "__main__":
    unittest.main()
