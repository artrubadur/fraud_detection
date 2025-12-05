import json
import os
from pathlib import Path

import joblib
import optuna
from catboost import CatBoost

from .data_preprocessor import DataPreprocessor


class DataManager:
    def __init__(self, base_path="./artifacts"):
        self.base_path = Path(base_path)

    def save_preprocessor(
        self, preprocessor: DataPreprocessor, name: str = "preprocessor"
    ):
        path = self.base_path / "preprocessing" / f"{name}.joblib"
        os.makedirs(path.parent, exist_ok=True)
        joblib.dump(preprocessor, path)

    def load_preprocessor(self, name: str = "preprocessor") -> DataPreprocessor:
        path = self.base_path / "preprocessing" / f"{name}.joblib"
        return joblib.load(path)

    def get_study_storage(self) -> str:
        path = self.base_path / "optimization"
        os.makedirs(path, exist_ok=True)
        return f"sqlite:///{path}/study.db"

    def load_study(self, name: str):
        storage = self.get_study_storage()
        return optuna.load_study(study_name=name, storage=storage)

    def save_params(self, params: dict, name: str = "best_params"):
        path = self.base_path / "models" / f"{name}.json"
        os.makedirs(path.parent, exist_ok=True)
        self._save(json.dumps(params, indent=4), path)

    def load_params(self, name: str = "best_params") -> dict:
        path = self.base_path / "models" / f"{name}.json"
        return json.loads(self._load(path))

    def save_model(self, model: CatBoost, name: str = "model"):
        path = self.base_path / "models" / f"{name}.cbm"
        os.makedirs(path.parent, exist_ok=True)
        model.save_model(path)

    def load_model(self, model: CatBoost, name: str = "model"):
        path = self.base_path / "models" / f"{name}.cbm"
        model.load_model(path)

    def save_threshold(self, threshold: float, name: str = "threshold"):
        path = self.base_path / "models" / f"{name}.txt"
        os.makedirs(path.parent, exist_ok=True)
        self._save(str(threshold), path)

    def load_threshold(self, name: str = "threshold"):
        path = self.base_path / "models" / f"{name}.txt"
        return float(self._load(path))

    def _save(self, data: str, path: Path):
        with open(path, "w") as f:
            f.write(data)

    def _load(self, path: Path):
        with open(path, "r") as f:
            return f.read()
