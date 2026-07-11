"""Configuración central y rutas del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

RANDOM_STATE = 42
TARGET_COLUMN = "es_fraude"
MODEL_COMPONENTS = 23
CLASSIFICATION_THRESHOLD = 0.50

RAW_COLUMN_MAP = {
    "trans_date_trans_time": "fecha_hora_transaccion",
    "cc_num": "numero_tarjeta",
    "merchant": "comercio",
    "category": "categoria_comercio",
    "amt": "monto",
    "first": "nombre",
    "last": "apellido",
    "gender": "genero",
    "street": "direccion",
    "city": "ciudad",
    "state": "estado",
    "zip": "codigo_postal",
    "lat": "latitud_cliente",
    "long": "longitud_cliente",
    "city_pop": "poblacion_ciudad",
    "job": "ocupacion",
    "dob": "fecha_nacimiento",
    "trans_num": "id_transaccion",
    "unix_time": "timestamp_unix",
    "merch_lat": "latitud_comercio",
    "merch_long": "longitud_comercio",
    "is_fraud": TARGET_COLUMN,
}

NUMERIC_FEATURES = [
    "log_monto",
    "poblacion_ciudad",
    "edad_cliente",
    "distancia_km",
    "es_fin_de_semana",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
]

LOW_CARDINALITY_FEATURES = [
    "genero",
    "categoria_comercio",
    "dia_semana_transaccion",
    "periodo_dia",
    "estado",
]

HIGH_CARDINALITY_FEATURES = [
    "ciudad",
    "comercio",
    "ocupacion",
    "codigo_postal",
]

MODEL_INPUT_FEATURES = (
    NUMERIC_FEATURES + LOW_CARDINALITY_FEATURES + HIGH_CARDINALITY_FEATURES
)

FINAL_CLEAN_COLUMNS = [
    "fecha_hora_transaccion",
    "comercio",
    "categoria_comercio",
    "monto",
    "genero",
    "ciudad",
    "estado",
    "codigo_postal",
    "latitud_cliente",
    "longitud_cliente",
    "poblacion_ciudad",
    "ocupacion",
    "fecha_nacimiento",
    "id_transaccion",
    "timestamp_unix",
    "latitud_comercio",
    "longitud_comercio",
    "hora_transaccion",
    "dia_semana_transaccion",
    "mes_transaccion",
    "edad_cliente",
    "distancia_km",
    "tarjeta_ultimos4",
    TARGET_COLUMN,
    "type",
]


def find_project_root(start: str | Path | None = None) -> Path:
    """Busca la raíz usando ``pyproject.toml`` como marcador.

    Si no se encuentra el marcador, devuelve el directorio actual o su padre
    cuando la ejecución se inició dentro de ``notebooks``.
    """
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    return current.parent if current.name.lower() == "notebooks" else current


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DELIVERY_DIR = PROJECT_ROOT / "delivery" / "week14_final_integrated_delivery"


def existing_features(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    """Devuelve las variables candidatas presentes, conservando el orden."""
    available = set(columns)
    return [name for name in candidates if name in available]
