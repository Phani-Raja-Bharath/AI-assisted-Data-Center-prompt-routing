"""Versioned model bundles on local disk, optionally backed by Neon/Postgres.

Only application-controlled artifacts belong here: joblib uses pickle internally.
"""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import tempfile

import joblib
import sklearn

from energy_model import model_input_names

DEFAULT_TRAINING_PERIOD = ("2021-01-01", "2024-12-31")  # historical weather window
SCHEMA_VERSION = 3  # 3: one-hot cooling inputs, cooling sampled across all weather


def training_config(mode, fingerprint, feature_names, cooling_types, period=DEFAULT_TRAINING_PERIOD):
    """Training identity: models trained on a different weather period are a different bundle."""
    return {"schema_version": SCHEMA_VERSION, "mode": mode,
            "fingerprint": fingerprint, "features": list(feature_names),
            "model_inputs": model_input_names(tuple(feature_names), tuple(cooling_types)),
            "cooling_types": list(cooling_types), "sklearn_version": sklearn.__version__,
            "use_real_weather": True, "period": [str(period[0]), str(period[1])], "n_samples": 5000}


def config_key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_bundle(bundle, config):
    if bundle.get("training_config") != config:
        raise ValueError("Model bundle is incompatible with the current training configuration.")
    if bundle.get("feature_names") != config["features"]:
        raise ValueError("Model feature schema does not match.")
    scaler = bundle.get("scalers", {}).get("main")
    if getattr(scaler, "n_features_in_", None) != len(config.get("model_inputs", config["features"])):
        raise ValueError("Model scaler has the wrong feature count.")
    if bundle.get("best_model_name") not in bundle.get("models", {}):
        raise ValueError("Model bundle has no usable best model.")
    if not bundle.get("metrics"):
        raise ValueError("Model bundle is missing evaluation metrics.")
    return bundle


class ModelStore:
    def __init__(self, directory, database_url=None):
        self.directory = Path(directory)
        self.database_url = database_url

    def _connect(self):
        import psycopg
        if not self.database_url:
            raise ValueError("A model database URL is required for remote storage.")
        # DATABASE_URL is Neon/PgBouncer's pooled connection. Keep timeout state
        # transaction-local and disable automatic named prepared statements.
        conn = psycopg.connect(self.database_url, connect_timeout=10, prepare_threshold=None)
        conn.execute("SET LOCAL statement_timeout = '15s'")
        return conn

    @staticmethod
    def _ensure_table(conn):
        migration = Path(__file__).resolve().parent / "migrations" / "001_model_bundles.sql"
        conn.execute(migration.read_text(encoding="utf-8"))

    def load(self, config):
        key = config_key(config)
        local_path = self.directory / f"models_{key}.joblib"
        if local_path.exists():
            try:
                return validate_bundle(joblib.load(local_path), config)
            except Exception:
                # A damaged cache is a miss, so a valid remote copy can recover it.
                pass
        if self.database_url:
            with self._connect() as conn:
                self._ensure_table(conn)
                row = conn.execute(
                    "SELECT bundle FROM routing_model_bundles WHERE config_key = %s",
                    (key,),
                ).fetchone()
            if row:
                bundle = validate_bundle(joblib.load(BytesIO(bytes(row[0]))), config)
                self._save_local(bundle, local_path)
                self._save_local(bundle, self.directory / "ai_models_bundle.joblib")
                return bundle
        return None

    def _save_local(self, bundle, path):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".joblib")
        os.close(handle)
        try:
            joblib.dump(bundle, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, bundle, config):
        validate_bundle(bundle, config)
        key = config_key(config)
        self._save_local(bundle, self.directory / f"models_{key}.joblib")
        self._save_local(bundle, self.directory / "ai_models_bundle.joblib")
        if self.database_url:
            buffer = BytesIO()
            joblib.dump(bundle, buffer)
            with self._connect() as conn:
                self._ensure_table(conn)
                conn.execute("""INSERT INTO routing_model_bundles (config_key, metadata, bundle)
                    VALUES (%s, %s::jsonb, %s)
                    ON CONFLICT (config_key) DO UPDATE SET
                    metadata = EXCLUDED.metadata, bundle = EXCLUDED.bundle, updated_at = now()
                    """, (key, json.dumps(config), buffer.getvalue()))


def bundle_from_suite(suite, config):
    return {"models": suite.models, "scalers": suite.scalers, "metrics": suite.metrics,
            "best_model_name": suite.best_model_name, "feature_names": suite.feature_names,
            "training_source": suite.training_source, "training_samples": suite.training_samples,
            "training_config": config, "revision": os.urandom(16).hex(),
            "trained_at": datetime.now(timezone.utc).isoformat()}


def restore_suite(suite, bundle):
    for name in ("models", "scalers", "metrics", "best_model_name", "feature_names",
                 "training_source", "training_samples"):
        setattr(suite, name, bundle[name])
    suite.best_model = suite.models[suite.best_model_name]
    suite.used_real_weather = suite.training_source == "real"
    suite.is_trained = True
    return suite


def get_or_train_bundle(config, store, memory, train, force=False):
    """Reuse session/disk/remote state; train only on a miss or explicit request."""
    key = config_key(config)
    notices = []
    if not force and key in memory:
        return validate_bundle(memory[key], config), "session", notices
    if not force:
        try:
            bundle = store.load(config)
            if bundle is not None:
                memory[key] = bundle
                return bundle, "saved bundle", notices
        except Exception:
            notices.append("Saved model storage could not be read; training a fresh model.")
    bundle = train()
    validate_bundle(bundle, config)
    memory[key] = bundle
    try:
        store.save(bundle, config)
    except Exception:
        notices.append("Model is reusable in this session, but persistence did not fully succeed. "
                       "Check local write access and the optional database configuration.")
    return bundle, "new training", notices
