"""Tests del veto de test (RNF-04). Solo libreria estandar (unittest)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-build"))

from carga import (
    VetoTestError,
    cargar_test,
    cargar_train,
    preparar_entrenamiento,
)

GUARDIAN = os.path.join(os.path.dirname(__file__), "..")
TRAIN = os.path.join(GUARDIAN, "datos", "train.jsonl")
TEST = os.path.join(GUARDIAN, "datos", "test_publico.jsonl")


def escribir(casos):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for c in casos:
        f.write(json.dumps(c) + "\n")
    f.close()
    return f.name


class TestVeto(unittest.TestCase):
    def test_train_real_carga(self):
        casos = cargar_train(TRAIN)
        self.assertEqual(len(casos), 1707)
        self.assertTrue(all("etiqueta" in c for c in casos))

    def test_test_real_carga(self):
        casos = cargar_test(TEST)
        self.assertEqual(len(casos), 440)
        self.assertFalse(any("etiqueta" in c for c in casos))

    def test_train_rechaza_fichero_test(self):
        with self.assertRaises(VetoTestError):
            cargar_train(TEST)

    def test_test_rechaza_fichero_train(self):
        with self.assertRaises(VetoTestError):
            cargar_test(TRAIN)

    def test_entrenar_con_test_dispara_error(self):
        with self.assertRaises(VetoTestError):
            preparar_entrenamiento([TEST])

    def test_entrenar_con_train_pasa(self):
        casos = preparar_entrenamiento([TRAIN])
        self.assertEqual(len(casos), 1707)

    def test_caso_train_sin_etiqueta_dispara_error(self):
        ruta = escribir([{"id": "x1", "respuesta": "r"}])
        self.addCleanup(os.unlink, ruta)
        with self.assertRaises(VetoTestError):
            cargar_train(ruta)

    def test_caso_test_con_etiqueta_dispara_error(self):
        ruta = escribir([{"id": "x1", "respuesta": "r", "etiqueta": 1}])
        self.addCleanup(os.unlink, ruta)
        with self.assertRaises(VetoTestError):
            cargar_test(ruta)


if __name__ == "__main__":
    unittest.main()
