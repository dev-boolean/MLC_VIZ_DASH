"""Países de la plataforma: identidad, fuente y definición de cada serie.

Solo aparecen los nueve países en los que ALEE se entrenó y se evaluó.
"""
from collections import OrderedDict

_OCDE = "OCDE, Infra-annual labour statistics"

COUNTRIES = OrderedDict([
    ("colombia", dict(name="Colombia", short="COL", iso=170, lat=4.71, lon=-74.07, source="DANE (GEIH), vía Banco de la República",
                      definition="Total nacional, sin ajuste estacional")),
    ("eeuu", dict(name="Estados Unidos", short="USA", iso=840, lat=38.90, lon=-77.04, source="BLS, vía FRED (UNRATENSA)",
                  definition="Total, 16 años o más, sin ajuste estacional")),
    ("espana", dict(name="España", short="ESP", iso=724, lat=40.42, lon=-3.70, source="Eurostat (une_rt_m)",
                    definition="Total, % de la población activa, sin ajuste estacional")),
    ("mexico", dict(name="México", short="MEX", iso=484, lat=19.43, lon=-99.13, source=f"{_OCDE} (INEGI, ENE y ENOE)",
                    definition="Total, 15 años o más, sin ajuste estacional")),
    ("canada", dict(name="Canadá", short="CAN", iso=124, lat=45.42, lon=-75.70, source=_OCDE,
                    definition="Total, 15 años o más, sin ajuste estacional")),
    ("australia", dict(name="Australia", short="AUS", iso=36, lat=-35.28, lon=149.13, source=_OCDE,
                       definition="Total, 15 años o más, sin ajuste estacional")),
    ("japon", dict(name="Japón", short="JPN", iso=392, lat=35.68, lon=139.69, source=_OCDE,
                   definition="Total, 15 años o más, sin ajuste estacional")),
    ("alemania", dict(name="Alemania", short="DEU", iso=276, lat=52.52, lon=13.40, source=_OCDE,
                      definition="Total, 15 años o más, sin ajuste estacional")),
    ("corea", dict(name="Corea del Sur", short="KOR", iso=410, lat=37.57, lon=126.98, source=_OCDE,
                   definition="Total, 15 años o más, sin ajuste estacional")),
])
DEFAULT = "colombia"

MODEL_LABELS = {
    "ALEE": "ALEE (modelo original)", "KNN": "KNN", "Ridge": "Ridge", "Lasso": "Lasso", "BayesianRidge": "Regresión bayesiana",
    "DecisionTree": "Árbol de decisión", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVR_RBF": "SVR (kernel RBF)",
    "LightGBM": "LightGBM",
}
EXPERT_LABELS = {"persistencia": "Persistencia", "nucleo": "Núcleo del cambio", "estacional": "Cambio estacional"}

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MON = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def name(c):
    return COUNTRIES[c]["name"]


def month_long(d):
    return f"{MONTHS[d.month - 1]} de {d.year}"


def month_short(d):
    return f"{MON[d.month - 1]} {d.year}"


def num(x, dec=2, sign=False):
    """Número con coma decimal, como se escribe en español."""
    s = f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"
    if float(s.replace("+", "")) == 0:
        s = s.replace("-", "").replace("+", "")
    return s.replace("-", "−").replace(".", ",")
