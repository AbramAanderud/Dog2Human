from app.db import Base, engine
from app import models_db  


def main():
    print("Creating tables in the database...")
    print("Using DB:", engine.url)
    Base.metadata.create_all(bind=engine)
    print("Done.")
 

if __name__ == "__main__":
    main()
