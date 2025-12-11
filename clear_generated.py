from app.db import SessionLocal
from app.models_db import GeneratedImage, DogImage

def main():
    db = SessionLocal()
    try:
        deleted_generated = db.query(GeneratedImage).delete()
        deleted_dogs = db.query(DogImage).delete()

        db.commit()

        print(f"Deleted {deleted_generated} rows from generated_images.")
        print(f"Deleted {deleted_dogs} rows from dog_images.")
    except Exception as e:
        db.rollback()
        print("Error while clearing tables:", e)
    finally:
        db.close()

if __name__ == "__main__":
    main()
