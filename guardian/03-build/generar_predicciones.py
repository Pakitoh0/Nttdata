"""Genera el fichero de predicciones que entregáis al organizador. Solo Python
estándar: no instala ni ejecuta ningún modelo.

  python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl --salida predicciones.csv

Tras el giro (15:30), pasad los dos ficheros a la vez (un solo CSV con todo):

  python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida predicciones.csv

Carga vuestra clase `Guardian` y puntúa cada caso. Produce un CSV `id,score`
(una línea por caso) listo para entregar.
"""
import argparse
import importlib.util
import json


def _cargar_bot(ruta):
    spec = importlib.util.spec_from_file_location("bot_equipo", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Guardian


def cargar_casos(rutas):
    return [json.loads(l) for r in rutas for l in open(r, encoding="utf-8") if l.strip()]


def _caso_para_bot(c):
    # El bot ve lo mismo que en el train, pero sin etiqueta ni tipo.
    return {k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales") if k in c}


def generar(bot_cls, casos):
    g = bot_cls()
    return [(c["id"], float(g.score(_caso_para_bot(c)))) for c in casos]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="ejemplo_bot.py")
    ap.add_argument("--test", required=True, nargs="+",
                    help="test_publico.jsonl (y giro_publico.jsonl tras las 15:30)")
    ap.add_argument("--salida", default="predicciones.csv")
    args = ap.parse_args(argv)

    casos = cargar_casos(args.test)
    filas = generar(_cargar_bot(args.bot), casos)
    with open(args.salida, "w", encoding="utf-8") as f:
        f.write("id,score\n")
        for cid, score in filas:
            f.write(f"{cid},{score:.6f}\n")
    print(f"{len(filas)} predicciones -> {args.salida}")


if __name__ == "__main__":
    main()
