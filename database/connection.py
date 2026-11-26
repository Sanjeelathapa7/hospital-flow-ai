
# HospitalFlow AI
# SQLite Database Connection

import sqlite3

from pathlib import Path
from contextlib import contextmanager

from core.config import DATABASE_PATH


SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schema.sql"
)


def get_connection(db_path=DATABASE_PATH):

    db_path = Path(db_path)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        str(db_path),
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return connection


@contextmanager
def database_transaction(
    db_path=DATABASE_PATH
):

    connection = get_connection(db_path)

    try:

        connection.execute(
            "BEGIN IMMEDIATE"
        )

        yield connection

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


def initialise_database(
    db_path=DATABASE_PATH
):

    connection = get_connection(db_path)

    try:

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        schema = SCHEMA_PATH.read_text(
            encoding="utf-8"
        )

        connection.executescript(schema)

        connection.commit()

    finally:

        connection.close()

    return Path(db_path)


def get_table_names(
    db_path=DATABASE_PATH
):

    connection = get_connection(db_path)

    try:

        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        return [
            row["name"]
            for row in rows
        ]

    finally:

        connection.close()
