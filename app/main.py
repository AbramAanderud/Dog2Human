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
from src.models import UNetDog2Human 
from .db import engine, get_db
from .auth import get_current_user
from .models_db import User, DogImage, GeneratedImage


app = FastAPI()

Base.metadata.create_all(bind=engine)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetDog2Human().to(device)

ckpt_path = Path("checkpoints_gan/gan_epoch_20.pt")
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"Loaded GAN checkpoint from {ckpt_path}")
else:
    print("WARNING: GAN checkpoint not found, app will not generate meaningful images.")

transform_input = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def tensor_to_pil(tensor):
    tensor = (tensor * 0.5) + 0.5    
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor.squeeze(0).cpu())

# routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Landing page
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    # Main
    return templates.TemplateResponse(
        "app.html",
        {"request": request},
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user), 
):
    """
    1. Save uploaded dog image to static/uploads
    2. Run GAN to generate human image
    3. Insert DogImage + GeneratedImage rows tied to this user
    4. Render page with the new generated image
    """
    # Save dog image
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_filename = f"dog_{timestamp}_{file.filename}"
    input_path = uploads_dir / input_filename

    contents = await file.read()
    with input_path.open("wb") as f:
        f.write(contents)

    img = Image.open(input_path).convert("RGB")
    img_t = transform_input(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = G(img_t)

    output_filename = f"human_{timestamp}.png"
    output_path = generated_dir / output_filename
    out_img = tensor_to_pil(fake)
    out_img.save(output_path)

    dog_rel_path = (input_path.relative_to(static_dir)).as_posix()      # e.g. "uploads/dog_..."
    gen_rel_path = (output_path.relative_to(static_dir)).as_posix()     # e.g. "generated/human_..."

    dog_row = DogImage(
        user_id=current_user.id,
        file_path=dog_rel_path,
    )
    db.add(dog_row)
    db.flush()  

    gen_row = GeneratedImage(
        user_id=current_user.id,
        dog_image_id=dog_row.id,
        file_path=gen_rel_path,
        model_version="gan_epoch_15",  
    )
    db.add(gen_row)
    db.commit()
    db.refresh(gen_row)

    return templates.TemplateResponse(
        "app.html",
        {
            "request": request,
            "generated_image_url": f"/static/{gen_rel_path}",
        },
    )


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  
):
    items = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.user_id == current_user.id)
        .order_by(GeneratedImage.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "items": items,
        },
    )