-- Idempotent model storage schema. Run with DATABASE_URL_UNPOOLED for migrations.
CREATE TABLE IF NOT EXISTS routing_model_bundles (
    config_key TEXT PRIMARY KEY,
    metadata JSONB NOT NULL,
    bundle BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
