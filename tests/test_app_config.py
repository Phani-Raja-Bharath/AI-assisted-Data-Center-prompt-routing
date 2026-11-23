from app_config import get_database_url


def test_neon_generated_file_and_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MODEL_DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text('DATABASE_URL="postgresql://dotenv"\n')
    assert get_database_url(project_dir=tmp_path) == "postgresql://dotenv"
    (tmp_path / ".env.local").write_text('DATABASE_URL="postgresql://local"\n')
    assert get_database_url(project_dir=tmp_path) == "postgresql://local"
    assert get_database_url({"MODEL_DATABASE_URL": "postgresql://secret"}, tmp_path) == "postgresql://secret"
    monkeypatch.setenv("DATABASE_URL", "postgresql://environment")
    assert get_database_url(project_dir=tmp_path) == "postgresql://environment"
    monkeypatch.setenv("MODEL_DATABASE_URL", "postgresql://explicit")
    assert get_database_url(project_dir=tmp_path) == "postgresql://explicit"


def test_empty_files_keep_local_mode_and_secrets_are_not_expanded(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MODEL_DATABASE_URL", raising=False)
    assert get_database_url(project_dir=tmp_path) is None
    (tmp_path / ".env.local").write_text('# comment\nDATABASE_URL=""\n')
    assert get_database_url(project_dir=tmp_path) is None
    (tmp_path / ".env.local").write_text('DATABASE_URL="postgresql://user:p${word}@example/db"\n')
    assert get_database_url(project_dir=tmp_path) == "postgresql://user:p${word}@example/db"
