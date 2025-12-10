# create_tables.py
from app.db import Base, engine
from app import models_db  # noqa: F401  # needed so models register with Base


def main():
    print("Creating tables in the database...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


if __name__ == "__main__":
    main()
