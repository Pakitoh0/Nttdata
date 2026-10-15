"""Guardian final (PASO 7 reentrenado limpio, config baseline + alineacion).

Senales usadas (solo `senales.oraciones`, nunca etiqueta ni tipo):
- Por oracion: `respaldo_todas`/`contradiccion_todas` (NLI conjunto) y
  `llm.veredicto`/`confianza` (juez LLM) -> q_i en escala logit
  (media geometrica de apuestas: evita sobrecontar evidencias
  correlacionadas); `respaldo_por_fuente` + `respaldo_todas` ->
  alineacion (cobertura, dispersion, margen_min, f_dos_peores).
- Agregacion respuesta: maximo + noisy-OR + media dos peores / 3
  (una sola afirmacion falsa basta, REQ-04) + reescalado B.2
  s = 4p/(1+3p): el corte Bayes p > 0,2 del coste 4:1 cae en 0,5.
- Logistica L2 (C = 1,0): z = b + w.((x-mu)/sigma), p = sigmoide(z),
  score = reescalado(p). Corte operativo 0,5 (score >= 0,5 = alucina).

Podado y por que (ablacion paso 6, OOF en H, gate IC95 > 0 en ambas):
- cifras: FUERA (dAUC -0,000, no aporta); marcadores: FUERA (dAUC
  -0,008, resta); Platt: FUERA (a 0,5 fijo da -0,566, exigiría
  re-derivar el umbral); peso de clase: FUERA (una sola via B.2).
  Modulos conservados como historial en 03-build, fuera del camino.

Desconfianza del juez/NLI: veredicto null o ilegible -> 0,5 neutral;
ante conflicto NLI/juez prevalece la media en logit (ninguno manda
solo); claves ausentes -> 0,0/0,5 defensivos, caso ilegible -> 0,55.
"""

import math

# Parametros finales EMBEBIDOS (reentreno config COLUMNAS_FINAL sobre
# TODO train no-H, 1076 casos; C = 1,0 por CV interna GroupKFold x3;
# columnas: cobertura, dispersion, margen_min, f_dos_peores, score_base).
_MU = [0.511988042935998, 1.2100371747211895, 0.19346840148698877,
       0.4829888475836429, 0.6103148786996748]
_SIGMA = [0.47067425336840607, 0.7048237925493791, 0.3526277009395813,
          0.4100596484697471, 0.4290932927022826]
_W = [1.0159930590537853, -0.6258563882720143, -1.4297281893979186,
      -0.4224026367058514, 3.2570450394786623]
_B = -0.38919133061890954


class Guardian:
    """Detector autocontenido: solo stdlib `math`, sin ficheros externos."""

    FALLBACK = 0.55
    EPS = 1e-4
    TAU = 0.5
    _VEREDICTO_MALO = ("no_respaldada", "contradicha")

    def score(self, caso):
        """Score [0, 1] de un caso (probabilidad de alucinacion)."""
        try:
            if not isinstance(caso, dict):
                return self.FALLBACK
            feats = self._vector_final(caso)
            if feats is None:
                return self.FALLBACK
            return self._puntuar_vector(feats)
        except (TypeError, ValueError, AttributeError):
            return self.FALLBACK

    @classmethod
    def _puntuar_vector(cls, x):
        try:
            z = _B
            for i, v in enumerate(x or []):
                try:
                    m = float(_MU[i])
                    s = float(_SIGMA[i])
                    w = float(_W[i])
                except (TypeError, ValueError, IndexError):
                    continue
                try:
                    if not math.isfinite(m):
                        m = 0.0
                    if not math.isfinite(s) or s == 0.0:
                        s = 1.0
                    if not math.isfinite(w):
                        w = 0.0
                    vv = float(v)
                    if not math.isfinite(vv):
                        vv = 0.0
                    z += w * ((vv - m) / s)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
            if not math.isfinite(z):
                return cls.FALLBACK
            if z >= 0.0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                e = math.exp(z)
                p = e / (1.0 + e)
            s = 4.0 * p / (1.0 + 3.0 * p)
            return min(1.0, max(0.0, s))
        except (TypeError, ValueError, AttributeError, OverflowError):
            return cls.FALLBACK

    @classmethod
    def _vector_final(cls, caso):
        try:
            senales = caso.get("senales", {}) or {}
            if not isinstance(senales, dict):
                return None
            oraciones = senales.get("oraciones", []) or []
            if not isinstance(oraciones, list) or not oraciones:
                return None
            qs = [cls._q_oracion(o) for o in oraciones]
            base = cls._agregar(qs)
            alin = cls._feats_alin(oraciones)
            return [alin[0], alin[1], alin[2], alin[3], base]
        except (TypeError, ValueError, AttributeError):
            return None

    @classmethod
    def _recortar(cls, p):
        try:
            return min(1.0 - cls.EPS, max(cls.EPS, float(p)))
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def _logit(cls, p):
        q = cls._recortar(p)
        return math.log(q / (1.0 - q))

    @staticmethod
    def _sigmoide(z):
        try:
            if z >= 0.0:
                return 1.0 / (1.0 + math.exp(-z))
            e = math.exp(z)
            return e / (1.0 + e)
        except (TypeError, ValueError, OverflowError):
            return 0.5

    @classmethod
    def _q_juez(cls, veredicto, confianza):
        try:
            if veredicto is None:
                return 0.5
            if veredicto in cls._VEREDICTO_MALO:
                return cls._recortar(confianza)
            if veredicto == "respaldada":
                return cls._recortar(1.0 - float(confianza))
            return 0.5
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def _q_oracion(cls, oracion):
        try:
            datos = oracion if isinstance(oracion, dict) else {}
            try:
                respaldo = float(datos.get("respaldo_todas", 0.5))
            except (TypeError, ValueError):
                respaldo = 0.5
            try:
                contradiccion = float(datos.get("contradiccion_todas", 0.5))
            except (TypeError, ValueError):
                contradiccion = 0.5
            llm = datos.get("llm", {}) or {}
            if not isinstance(llm, dict):
                llm = {}
            q_juez = cls._q_juez(llm.get("veredicto"), llm.get("confianza"))
            q_nli = max(1.0 - respaldo, contradiccion)
            q = cls._sigmoide((cls._logit(q_nli) + cls._logit(q_juez)) / 2.0)
            return min(1.0, max(0.0, q))
        except (TypeError, ValueError, AttributeError):
            return 0.5

    @classmethod
    def _agregar(cls, qs):
        try:
            valores = []
            for q in qs or []:
                try:
                    valores.append(min(1.0, max(0.0, float(q))))
                except (TypeError, ValueError):
                    continue
            if not valores:
                return cls.FALLBACK
            peor = max(valores)
            resto = 1.0
            for q in valores:
                resto *= 1.0 - q
            ruidoso = 1.0 - resto
            ordenadas = sorted(valores, reverse=True)
            if len(ordenadas) >= 2:
                dos_peores = (ordenadas[0] + ordenadas[1]) / 2.0
            else:
                dos_peores = ordenadas[0]
            combinado = (peor + ruidoso + dos_peores) / 3.0
            reescalado = 4.0 * combinado / (1.0 + 3.0 * combinado)
            return min(1.0, max(0.0, reescalado))
        except (TypeError, ValueError, AttributeError):
            return cls.FALLBACK

    @classmethod
    def _a_cero_uno(cls, x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        try:
            if not math.isfinite(v):
                return 0.0
        except (TypeError, ValueError):
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @classmethod
    def _alin_oracion(cls, oracion):
        try:
            datos = oracion if isinstance(oracion, dict) else {}
            try:
                todas = cls._a_cero_uno(datos.get("respaldo_todas", 0.0))
            except (TypeError, ValueError, AttributeError):
                todas = 0.0
            try:
                brutas = datos.get("respaldo_por_fuente", []) or []
                if isinstance(brutas, (str, bytes)):
                    brutas = []
                valores = [cls._a_cero_uno(v) for v in list(brutas)]
            except (TypeError, ValueError, AttributeError):
                valores = []
            if not valores:
                return [todas]
            return [v if v >= todas else todas for v in valores]
        except (TypeError, ValueError, AttributeError):
            return [0.0]

    @classmethod
    def _feats_alin(cls, oraciones):
        try:
            mejores = []
            usados = []
            segundas = []
            for oracion in oraciones:
                try:
                    vals = cls._alin_oracion(oracion)
                except (TypeError, ValueError, AttributeError):
                    vals = [0.0]
                if not vals:
                    vals = [0.0]
                try:
                    mejor = max(vals)
                except (TypeError, ValueError):
                    mejor = 0.0
                try:
                    orden = sorted(vals, reverse=True)
                    segunda = orden[1] if len(orden) >= 2 else 0.0
                except (TypeError, ValueError, IndexError):
                    segunda = 0.0
                try:
                    arg = vals.index(mejor)
                except (TypeError, ValueError):
                    arg = 0
                mejores.append(mejor)
                segundas.append(segunda)
                usados.append(arg)
            n = len(mejores)
            cobertura = sum(1 for m in mejores if m >= cls.TAU) / float(n)
            if not math.isfinite(cobertura):
                cobertura = 0.0
            dispersion = len(set(usados))
            margenes = [max(0.0, b - s) for b, s in zip(mejores, segundas)]
            margen_min = min(margenes) if margenes else 0.0
            if not math.isfinite(margen_min):
                margen_min = 0.0
            ordenadas = sorted(mejores)
            if len(ordenadas) >= 2:
                f_dos = (ordenadas[0] + ordenadas[1]) / 2.0
            else:
                f_dos = ordenadas[0]
            if not math.isfinite(f_dos):
                f_dos = 0.0
            return cobertura, dispersion, margen_min, f_dos
        except (TypeError, ValueError, AttributeError):
            return 0.0, 0, 0.0, 0.0
