from fastapi import FastAPI, File, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image

from sqlalchemy.orm import Session

from .database import Base, engine, SessionLocal
from .db_models import Generation
from src.models import Dog2HumanNet  # re-use your generator


app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

# Static & templates
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
uploads_dir = static_dir / "uploads"
generated_dir = static_dir / "generated"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

uploads_dir.mkdir(parents=True, exist_ok=True)
generated_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load model (GAN-trained generator)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Dog2HumanNet().to(device)

# TODO: point this to your best GAN checkpoint
ckpt_path = Path("checkpoints_gan/gan_epoch_20.pt")
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
else:
    print("WARNING: GAN checkpoint not found, app will not generate meaningful images.")


transform_input = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def tensor_to_pil(tensor):
    tensor = (tensor * 0.5) + 0.5     # [-1,1] -> [0,1]
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor.squeeze(0).cpu())


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save uploaded dog image
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_filename = f"dog_{timestamp}_{file.filename}"
    input_path = uploads_dir / input_filename

    with input_path.open("wb") as f:
        f.write(await file.read())

    # Load & preprocess
    img = Image.open(input_path).convert("RGB")
    img_t = transform_input(img).unsqueeze(0).to(device)

    # Run generator
    with torch.no_grad():
        fake = G(img_t)

    # Save output image
    output_filename = f"human_{timestamp}.png"
    output_path = generated_dir / output_filename
    out_img = tensor_to_pil(fake)
    out_img.save(output_path)

    # Store in DB
    gen = Generation(
        input_path=str(input_path.relative_to(static_dir.parent)),   # store relative
        output_path=str(output_path.relative_to(static_dir.parent)),
    )
    db.add(gen)
    db.commit()
    db.refresh(gen)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "generated_image_url": f"/static/generated/{output_filename}",
        },
    )


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request, db: Session = Depends(get_db)):
    items = db.query(Generation).order_by(Generation.created_at.desc()).all()
    return templates.TemplateResponse(
        "gallery.html",
        {"request": request, "items": items},
    )
