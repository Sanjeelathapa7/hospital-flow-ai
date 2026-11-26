
# Database Initialisation

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

from database.connection import (
    initialise_database,
    get_table_names,
    get_connection
)


def main():

    print("=" * 60)
    print("HOSPITALFLOW AI DATABASE")
    print("=" * 60)

    db_path = initialise_database()

    print("\nDatabase created at:")

    print(db_path)

    tables = get_table_names(db_path)

    print("\nDATABASE TABLES")

    for index, table in enumerate(
        tables, start=1
    ):

        print(f"{index}. {table}")

    connection = get_connection(db_path)

    try:

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        foreign_keys = connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]

        print(
            "\nDatabase integrity:",
            integrity
        )

        print(
            "Foreign keys enabled:",
            bool(foreign_keys)
        )

    finally:

        connection.close()

    print("\nTotal tables:", len(tables))

    print(
        "\nDATABASE INITIALISATION COMPLETED"
    )


if __name__ == "__main__":

    main()
