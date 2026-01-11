import torch
import os
import numpy as np
from PIL import Image, ImageFilter
from IA.sharp_model import SharpSRModel
import psycopg2
import psycopg2.extras
from psycopg2.extras import Json
import base64
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
import json

baseDir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)
modelPath = os.path.join(baseDir, "IA", "model.pth")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def loadModel():
    global _model
    if _model is None:
        _model = SharpSRModel().to(_device)
        _model.load_state_dict(
            torch.load(modelPath, map_location=_device)
        )
        _model.eval()
    return _model

def runModel(pil_image: Image.Image) -> Image.Image:
    model = loadModel()
    img = pil_image.convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        out = model(tensor)[0]
    
    out = out.clamp(0, 1).cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    return Image.fromarray((out * 255).astype(np.uint8))

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)



DBHOST = os.getenv("dbHost")
DBNAME = os.getenv("dbName")
DBUSER = os.getenv("dbUser")
DBTABLE = os.getenv("dbTable2", "users")
DBPORT = os.getenv("dbPort")
DBPASSWORD = os.getenv("dbPassword")

_conn = None
_cur = None

def get_db_connection():
    global _conn, _cur
    if _conn is None or _conn.closed:
        print(f"Connecting to PostgreSQL with user: {DBUSER}")
        _conn = psycopg2.connect(
            host=DBHOST,
            database=DBNAME,
            user=DBUSER,
            password=DBPASSWORD,
            port=DBPORT
        )
        _cur = _conn.cursor()
    return _conn, _cur

def stockImgDb(pil_image: Image.Image, uuid: str, is_high_res: bool = True) -> bool:
    try:
        conn, cur = get_db_connection()
        
        buf = BytesIO()
        pil_image.save(buf, format="WEBP", quality=90)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        img_url = f"data:image/webp;base64,{img_base64}"
        
        column = "newhigh" if is_high_res else "newlow"
        
        cur.execute(f"SELECT {column} FROM {DBTABLE} WHERE uuid = %s", (uuid,))
        result = cur.fetchone()
        
        if result is None:
            print(f"User {uuid} not found in database")
            return False
        
        current_array = result[0] if result[0] else []
        current_array.append(img_url)
        
        cur.execute(f"""
            UPDATE {DBTABLE}
            SET {column} = %s
            WHERE uuid = %s
        """, (Json(current_array), uuid))
        
        conn.commit()
        print(f"Image successfully added to {column} for user {uuid}")
        print(f"Total images in {column}: {len(current_array)}")
        return True
        
    except Exception as e:
        if _conn:
            _conn.rollback()
        print(f"Error storing image: {e}")
        import traceback
        traceback.print_exc()
        return False

def closeDbConnection():
    global _conn, _cur
    if _cur:
        _cur.close()
    if _conn:
        _conn.close()

def degradeImg(file):
    width, height = file.size
    small = file.resize((width // 4, height // 4), Image.NEAREST)
    degraded = small.resize((width, height), Image.NEAREST)    
    degraded = degraded.filter(ImageFilter.GaussianBlur(radius=2))
    degraded = degraded.convert('P', palette=Image.ADAPTIVE, colors=16)
    degraded = degraded.convert('RGB')
    return degraded