"""Tests de marcadores diferenciales + flip de polaridad (PASO 4, B.6 + B.12)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from marcadores import feats_marc


def _caso(respuesta, fuentes, fuente_citada=0):
    """Caso minimo: respuesta + fuentes + veredicto llm con fuente."""
    llm = {"veredicto": "respaldada", "confianza": 0.9}
    if fuente_citada is not None:
        llm["fuente"] = fuente_citada
    oracion = {"texto": respuesta, "llm": llm}
    return {"respuesta": respuesta, "fuentes": list(fuentes),
            "senales": {"oraciones": [oracion]}}


class TestCausal(unittest.TestCase):
    def test_causal_montejo_gracias_a(self):
        # N08::r3 (alucinacion): el "gracias a" lo pone R, no la fuente.
        fuentes = ["La agrupacion de Montejo cuenta con 46 miembros. "
                   "Dispone de dos vehiculos y un presupuesto de 28000 euros."]
        respuesta = ("La agrupacion intervino en tres incendios gracias "
                     "a su vehiculo con deposito de 600 litros.")
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[0], 1)

    def test_causal_en_fuente_no_marca(self):
        fuentes = ["Intervino gracias a su vehiculo con deposito."]
        respuesta = "Intervino gracias a su vehiculo con deposito."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[0], 0)

    def test_sin_causal_da_cero(self):
        fuentes = ["Cuenta con 46 miembros y dos vehiculos."]
        respuesta = "Cuenta con 46 miembros y dos vehiculos."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[0], 0)


class TestAlcance(unittest.TestCase):
    def test_todos_vs_varias(self):
        fuentes = ["Al concurso se presentaron varias queserias de la comarca."]
        respuesta = ("Al concurso se presentaron todas las queserias "
                     "de la comarca.")
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[1], 1)

    def test_alcance_en_fuente_no_marca(self):
        fuentes = ["Participaron todas las queserias de la comarca."]
        respuesta = "Participaron todas las queserias de la comarca."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[1], 0)

    def test_sin_cuantificador_da_cero(self):
        fuentes = ["Participaron varias queserias de la comarca."]
        respuesta = "Participaron varias queserias de la comarca."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[1], 0)


class TestHedge(unittest.TestCase):
    def test_hedge_perdido(self):
        fuentes = ["El programa podria beneficiar a 200 familias de la comarca."]
        respuesta = "El programa beneficia a 200 familias de la comarca."
        self.assertGreaterEqual(feats_marc(_caso(respuesta, fuentes))[2], 1)

    def test_hedge_conservado_no_cuenta(self):
        fuentes = ["El programa podria beneficiar a 200 familias."]
        respuesta = "El programa podria beneficiar a 200 familias."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[2], 0)

    def test_varios_matices_suman(self):
        fuentes = ["Solo podrian participar unas 30 queserias de la comarca."]
        respuesta = "Pueden participar 30 queserias de la comarca."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[2], 3)

    def test_matiz_de_frase_no_retomada_no_cuenta(self):
        # El "solo" habla de otro tema: su ventana no comparte nada con R.
        fuentes = ["Solo el jurado conoce el fallo. Gano Penalisa el primer premio."]
        respuesta = "Gano Penalisa el primer premio."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[2], 0)


class TestFlip(unittest.TestCase):
    def test_aumento_cayo_mismo_objeto(self):
        fuentes = ["La produccion cayo en 2025 por la sequia."]
        respuesta = "La produccion aumento en 2025 por la sequia."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[3], 1)

    def test_distinto_objeto_no_flip(self):
        fuentes = ["El consumo cayo en 2025 por la sequia."]
        respuesta = "La produccion aumento en 2025 por la sequia."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[3], 0)

    def test_mismo_signo_no_flip(self):
        fuentes = ["La produccion crecio en 2025 por la sequia."]
        respuesta = "La produccion aumento en 2025 por la sequia."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[3], 0)

    def test_sin_verbos_da_cero(self):
        fuentes = ["La produccion de 2025 por la sequia."]
        respuesta = "La produccion de 2025 por la sequia."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes))[3], 0)


class TestFuenteCitada(unittest.TestCase):
    def test_citada_manda_sobre_otras(self):
        # El marcador falta en la citada (0) aunque este en la otra (1).
        fuentes = ["Cuenta con 46 miembros.",
                   "Intervino gracias a su vehiculo."]
        respuesta = "Intervino con 46 miembros gracias a su vehiculo."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes, 0))[0], 1)
        self.assertEqual(feats_marc(_caso(respuesta, fuentes, 1))[0], 0)

    def test_sin_llm_usa_todas(self):
        fuentes = ["Cuenta con 46 miembros.",
                   "Intervino gracias a su vehiculo."]
        respuesta = "Intervino con 46 miembros gracias a su vehiculo."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes, None))[0], 0)


class TestSinMarcadores(unittest.TestCase):
    def test_todo_cero(self):
        fuentes = ["Penalisa obtuvo el primer premio en curados."]
        respuesta = "Penalisa obtuvo el primer premio en curados."
        self.assertEqual(feats_marc(_caso(respuesta, fuentes)), (0, 0, 0, 0))

    def test_ilegible_no_lanza(self):
        self.assertEqual(feats_marc({}), (0, 0, 0, 0))
        self.assertEqual(feats_marc(None), (0, 0, 0, 0))
        self.assertEqual(feats_marc({"respuesta": None}), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
