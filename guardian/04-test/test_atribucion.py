"""Tests de atribucion de cifras + desacuerdo con el juez (PASO 5, B.4 + B.6)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from atribucion import feats_atrib


def _oracion(veredicto="respaldada", confianza=0.9, respaldo=0.9,
             contradiccion=0.05, por_fuente=None):
    """Oracion minima con senales NLI y veredicto del juez."""
    sen = {"texto": "x", "respaldo_todas": respaldo,
           "contradiccion_todas": contradiccion,
           "llm": {"veredicto": veredicto, "confianza": confianza}}
    if por_fuente is not None:
        sen["respaldo_por_fuente"] = list(por_fuente)
    return sen


def _caso(respuesta, fuentes, oraciones):
    return {"respuesta": respuesta, "fuentes": list(fuentes),
            "senales": {"oraciones": list(oraciones)}}


FUENTES_CRUZ = [
    "La empresa Alfa facturo 42 millones en 2025 tras vender su filial.",
    "La empresa Beta contrato 15 ingenieros y abrio sede en Oporto.",
]
RESP_CRUZ = "La empresa Beta facturo 42 millones en 2025."

FUENTES_BIEN = [
    "La queseria Penalisa obtuvo el primer premio con un queso de 14 meses de maduracion.",
    "El jurado estuvo formado por 9 catadores de la comarca.",
]
RESP_BIEN = "Penalisa obtuvo el primer premio con un queso de 14 meses de maduracion."


class TestAtribucionCruzada(unittest.TestCase):
    def test_cruzada_f_attr_bajo(self):
        # B factura 42 de A: la cifra cuadra pero el contexto no es suyo.
        caso = _caso(RESP_CRUZ, FUENTES_CRUZ,
                     [_oracion(respaldo=0.5, contradiccion=0.1)])
        f_attr, _, _, _, _ = feats_atrib(caso)
        self.assertLess(f_attr, 0.7)

    def test_bien_atribuida_f_attr_alto(self):
        caso = _caso(RESP_BIEN, FUENTES_BIEN,
                     [_oracion(respaldo=0.9, contradiccion=0.05)])
        f_attr, _, _, _, _ = feats_atrib(caso)
        self.assertGreater(f_attr, 0.75)

    def test_orden_cruz_menor_que_bien(self):
        f_cruz = feats_atrib(_caso(RESP_CRUZ, FUENTES_CRUZ, [_oracion()]))[0]
        f_bien = feats_atrib(_caso(RESP_BIEN, FUENTES_BIEN, [_oracion()]))[0]
        self.assertLess(f_cruz, f_bien)

    def test_margen_cruz_menor_que_bien(self):
        m_cruz = feats_atrib(_caso(RESP_CRUZ, FUENTES_CRUZ, [_oracion()]))[1]
        m_bien = feats_atrib(_caso(RESP_BIEN, FUENTES_BIEN, [_oracion()]))[1]
        self.assertLess(m_cruz, m_bien)
        self.assertLess(m_cruz, 0.5)


class TestDesacuerdo(unittest.TestCase):
    def test_d2_alto_respaldada_sin_respaldo(self):
        # Montejo-like: el juez dice "respaldada" con confianza y la NLI no respalda.
        caso = _caso("La agrupacion intervino en tres incendios.",
                     ["La agrupacion cuenta con 46 miembros."],
                     [_oracion(veredicto="respaldada", confianza=0.9,
                               respaldo=0.1, contradiccion=0.05)])
        _, _, _, d2max, _ = feats_atrib(caso)
        self.assertGreater(d2max, 0.5)
        self.assertAlmostEqual(d2max, 0.9 * 0.9, places=4)

    def test_d1_negativo_juez_confiado_nli_vacia(self):
        caso = _caso("x con 3 cosas.", ["Hay 3 cosas."],
                     [_oracion(veredicto="respaldada", confianza=0.9,
                               respaldo=0.1, contradiccion=0.05)])
        _, _, d1min, _, _ = feats_atrib(caso)
        self.assertLess(d1min, 0.0)

    def test_q_convencion_no_respaldada(self):
        # q_LLM = kappa con veredicto malo (misma convencion que reglas.py).
        caso = _caso("x con 3 cosas.", ["Hay 3 cosas."],
                     [_oracion(veredicto="no_respaldada", confianza=0.8,
                               respaldo=0.3, contradiccion=0.2)])
        _, _, _, d2max, d3max = feats_atrib(caso)
        self.assertAlmostEqual(d2max, 0.2 * 0.7, places=4)
        self.assertAlmostEqual(d3max, 0.2 * 0.2, places=4)

    def test_veredicto_null_defaults_finitos(self):
        caso = _caso("x con 3 cosas.", ["Hay 3 cosas."],
                     [_oracion(veredicto=None, confianza=None,
                               respaldo=0.8, contradiccion=0.1)])
        feats = feats_atrib(caso)
        self.assertEqual(len(feats), 5)
        for v in feats:
            self.assertTrue(math.isfinite(v), v)
        # q = 0.5 neutral: d2 = 0.5 * (1 - 0.8), d3 = 0.5 * 0.1.
        self.assertAlmostEqual(feats[3], 0.5 * 0.2, places=4)
        self.assertAlmostEqual(feats[4], 0.5 * 0.1, places=4)


class TestDefensivo(unittest.TestCase):
    def test_sin_cifras_vacuidad(self):
        caso = _caso("Penalisa gano el primer premio.", ["Penalisa gano."],
                     [_oracion()])
        f_attr, margen, _, _, _ = feats_atrib(caso)
        self.assertEqual((f_attr, margen), (1.0, 1.0))

    def test_ilegible_no_lanza(self):
        for malo in ({}, None, "texto", {"respuesta": None},
                     {"fuentes": None, "senales": None}):
            feats = feats_atrib(malo)
            self.assertEqual(len(feats), 5)
            for v in feats:
                self.assertTrue(math.isfinite(v), (malo, v))

    def test_nan_no_contamina(self):
        caso = _caso("Hay 3 cosas.", ["Hay 3 cosas."],
                     [_oracion(respaldo=float("nan"),
                               contradiccion=float("inf"))])
        for v in feats_atrib(caso):
            self.assertTrue(math.isfinite(v), v)

    def test_peor_trozo_manda(self):
        # Dos oraciones: el maximo de d2/d3 y el minimo de d1 mandan.
        caso = _caso("Hay 3 cosas y 5 mas.", ["Hay 3 cosas y 5 mas."],
                     [_oracion(respaldo=0.9, contradiccion=0.05),
                      _oracion(veredicto="respaldada", confianza=0.9,
                                respaldo=0.1, contradiccion=0.05)])
        _, _, d1min, d2max, _ = feats_atrib(caso)
        self.assertAlmostEqual(d2max, 0.81, places=4)
        self.assertLess(d1min, 0.0)


if __name__ == "__main__":
    unittest.main()
