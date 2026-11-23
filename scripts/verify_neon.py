"""Exercise the application's real Postgres model persistence, without leaking URLs."""
import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from dotenv import dotenv_values
import numpy as np
import psycopg

from app_config import get_database_url
from energy_model import COOLING_TYPES, FEATURE_NAMES
from model_store import (ModelStore, bundle_from_suite, config_key, get_or_train_bundle,
                         restore_suite, training_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, help="Use an explicit ignored environment file.")
    parser.add_argument("--migrate", action="store_true", help="Apply the idempotent schema over a direct connection.")
    args = parser.parse_args()
    if args.env_file:
        values = dotenv_values(args.env_file, interpolate=False)
        url = values.get("DATABASE_URL")
    else:
        values = {}
        for name in (".env", ".env.local"):
            if (root / name).is_file():
                values.update(dotenv_values(root / name, interpolate=False))
        values.update({k: v for k, v in os.environ.items() if k.startswith("DATABASE_URL")})
        url = get_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is missing. Run neon deploy first.")
    if args.migrate:
        direct = values.get("DATABASE_URL_UNPOOLED")
        if not direct:
            raise RuntimeError("DATABASE_URL_UNPOOLED is required for migrations.")
        with psycopg.connect(direct, connect_timeout=15) as conn:
            ModelStore._ensure_table(conn)
        print("PASS: versioned schema migration applied using the direct connection.")

    with psycopg.connect(url, connect_timeout=15) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
        assert conn.pgconn.ssl_in_use, "Database connection is not using TLS."
    print("PASS: pooled Postgres SELECT 1 over TLS.")

    # Train the app's actual MLR and restore it from the remote database into a
    # different empty directory, so a local cache cannot hide a broken connection.
    import ai_assisted_datacenter_allocation as app

    cfg = training_config("mlr", f"neon-e2e-{uuid.uuid4().hex}", FEATURE_NAMES, COOLING_TYPES)
    key = config_key(cfg)
    try:
        with tempfile.TemporaryDirectory(prefix="neon-e2e-") as temporary:
            first = ModelStore(Path(temporary) / "writer", url)
            second = ModelStore(Path(temporary) / "fresh-reader", url)
            trained = []

            def train():
                suite = app.AIModelSuite()
                with redirect_stdout(io.StringIO()):
                    suite.train_all("mlr", use_real_weather=False, n_samples=300)
                trained.append(suite)
                return bundle_from_suite(suite, cfg)

            saved, source, notices = get_or_train_bundle(cfg, first, {}, train)
            assert source == "new training" and not notices, "Remote model save did not succeed."
            print("PASS: application trained a model and persisted its bundle to Postgres.")

            def forbidden_training():
                raise AssertionError("Fresh app session unexpectedly retrained.")

            loaded, source, notices = get_or_train_bundle(cfg, second, {}, forbidden_training)
            assert source == "saved bundle" and not notices
            features = [21.8, 36, 5.25, 978, 906, COOLING_TYPES.index("air_economizer")]
            restored = restore_suite(app.AIModelSuite(), loaded)
            expected, actual = trained[0].predict(features), restored.predict(features)
            np.testing.assert_allclose(actual, expected)
            assert loaded["revision"] == saved["revision"]
            print("PASS: fresh app session loaded the remote bundle without training; predictions match.")
    finally:
        # Delete only this invocation's uniquely identified test record.
        with psycopg.connect(url, connect_timeout=15) as conn:
            conn.execute("DELETE FROM routing_model_bundles WHERE config_key = %s", (key,))
    print("PASS: isolated verification record removed; application tables preserved.")
    print("NEON END-TO-END VERIFICATION PASSED.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        message = re.sub(r"postgres(?:ql)?://\S+", "[REDACTED DATABASE URL]", str(exc))
        print(f"NEON VERIFICATION FAILED: {type(exc).__name__}: {message}", file=sys.stderr)
        sys.exit(1)
