from fastapi import FastAPI, File, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
import requests


import torch
from torchvision import transforms
from PIL import Image

from sqlalchemy.orm import Session

from .db import get_db
from .models_db import User, DogImage, GeneratedImage
from .auth import get_current_user
from .auth import router as auth_router
from src.models import UNetDog2Human 


app = FastAPI()
app.include_router(auth_router)


BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
uploads_dir = static_dir / "uploads"
generated_dir = static_dir / "generated"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

uploads_dir.mkdir(parents=True, exist_ok=True)
generated_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetDog2Human().to(device)

CKPT_DIR = Path("checkpoints_gan")
CKPT_DIR.mkdir(exist_ok=True)

ckpt_path = CKPT_DIR / "gan_epoch_20.pt"

CKPT_URL = "https://github.com/AbramAanderud/Dog2Human/releases/download/v1-gan-checkpoint/gan_epoch_20.pt"


if not ckpt_path.exists():
    print("Checkpoint not found locally. Downloading...")
    resp = requests.get(CKPT_URL, stream=True)
    resp.raise_for_status()
    with ckpt_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Checkpoint download complete.")

try:
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"Loaded GAN checkpoint from {ckpt_path}")
except Exception as e:
    print(f"WARNING: Failed to load checkpoint: {e}")
    print("App will run but image quality will be bad.")




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


@app.post("/generate")
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
    4. Return JSON with URLs so the frontend can update the page
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

    dog_rel_path = (input_path.relative_to(static_dir)).as_posix()     
    gen_rel_path = (output_path.relative_to(static_dir)).as_posix()    

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
        model_version="gan_epoch_20",
    )
    db.add(gen_row)
    db.commit()
    db.refresh(gen_row)

    return {
        "uploaded_image_url": f"/static/{dog_rel_path}",
        "generated_image_url": f"/static/{gen_rel_path}",
    }



@app.get("/gallery", response_class=HTMLResponse)
async def gallery(
    request: Request,
    db: Session = Depends(get_db),
):
    items = (
        db.query(GeneratedImage)
        .order_by(GeneratedImage.created_at.desc())
        .limit(50)
        .all()
    )

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "items": items,
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})