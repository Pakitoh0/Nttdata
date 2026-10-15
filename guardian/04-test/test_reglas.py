"""Tests del baseline de solo-reglas (PASO 1). Solo libreria estandar (unittest)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from reglas import agregar_respuesta, prob_juez_llm, prob_no_respaldo_oracion


def _oracion(respaldo, contradiccion, veredicto, confianza):
    return {
        "respaldo_todas": respaldo,
        "contradiccion_todas": contradiccion,
        "llm": {"veredicto": veredicto, "confianza": confianza},
    }


class TestJuezLlm(unittest.TestCase):
    def test_veredicto_null_da_un_medio(self):
        # FUNDAMENTO B.4: sin veredicto no hay evidencia -> 0.5 neutral.
        self.assertEqual(prob_juez_llm(None, 0.9), 0.5)
        self.assertEqual(prob_juez_llm(None, None), 0.5)
        # Oracion neutra en NLI + juez null -> q = 0.5 exacto.
        self.assertAlmostEqual(
            prob_no_respaldo_oracion(_oracion(0.5, 0.5, None, None)), 0.5
        )

    def test_mapeo_veredicto_confianza(self):
        self.assertAlmostEqual(prob_juez_llm("respaldada", 0.9), 0.1)
        self.assertAlmostEqual(prob_juez_llm("no_respaldada", 0.8), 0.8)
        self.assertAlmostEqual(prob_juez_llm("contradicha", 0.7), 0.7)


class TestAgregacion(unittest.TestCase):
    def test_una_falsa_basta_no_promedio(self):
        # Una oracion mala + dos buenas: el promedio simple (< 0.5)
        # diria "fiel"; el maximo debe imponer sospecha (>= 0.5).
        mala = _oracion(0.01, 0.9, "contradicha", 0.9)
        buena = _oracion(0.95, 0.02, "respaldada", 0.95)
        qs = [prob_no_respaldo_oracion(o) for o in (mala, buena, buena)]
        self.assertLess(sum(qs) / len(qs), 0.5)
        self.assertGreaterEqual(agregar_respuesta(qs), 0.5)


class TestRobustez(unittest.TestCase):
    def test_clipping_cero_uno(self):
        # Valores fuera de escala no pueden romper el rango [0, 1].
        rara = _oracion(999.0, -5.0, "respaldada", 999.0)
        q = prob_no_respaldo_oracion(rara)
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)
        self.assertGreaterEqual(agregar_respuesta([q, 2.0, -3.0]), 0.0)
        self.assertLessEqual(agregar_respuesta([q, 2.0, -3.0]), 1.0)

    def test_caso_ilegible_da_fallback(self):
        from mi_bot import Guardian

        g = Guardian()
        # Sin 'senales' o con caso vacio: 0.55 (coste asimetrico, B.9).
        self.assertEqual(g.score({}), 0.55)
        self.assertEqual(g.score({"senales": {}}), 0.55)
        self.assertEqual(g.score({"senales": {"oraciones": []}}), 0.55)

    def test_mi_bot_acepta_formato_sin_etiqueta(self):
        # Como generar_predicciones._caso_para_bot: sin etiqueta ni tipo.
        from mi_bot import Guardian

        caso = {
            "pregunta": "p",
            "fuentes": ["f"],
            "respuesta": "r",
            "senales": {"oraciones": [_oracion(0.9, 0.05, "respaldada", 0.9)]},
        }
        s = Guardian().score(caso)
        self.assertIsInstance(s, float)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
