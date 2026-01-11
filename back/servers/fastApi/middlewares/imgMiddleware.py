from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import io
import magic

async def checkImg(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    contents = await file.read()

    mime = magic.from_buffer(contents, mime=True)
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fake image")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted image")

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(400, "File too large")


    image = Image.open(io.BytesIO(contents))

    image = image.resize((256, 256))
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=90)
    buf.seek(0)

    return {
        "filename": file.filename,
        "content_type": "image/webp",
        "file": buf
    }
