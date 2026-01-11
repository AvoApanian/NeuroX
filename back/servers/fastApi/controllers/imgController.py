# servers/fastApi/controllers/imgController.py
from PIL import Image
from fastapi import UploadFile, HTTPException
from io import BytesIO
import base64
from fastapi.concurrency import run_in_threadpool
from servers.fastApi.service.imgService import runModel, stockImgDb,degradeImg

async def controller_low(file: UploadFile, uuid: str):
    try:
        # Ouvrir l'image
        pil_image = Image.open(file.file).convert("RGB")
        
        # Exécuter le modèle d'amélioration
        result_image = await run_in_threadpool(runModel, pil_image)
        
        # Stocker dans la base de données
        success = await run_in_threadpool(stockImgDb, result_image, uuid, is_high_res=False)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store image in database")
        
        # Retourner l'image en base64 pour affichage
        buf = BytesIO()
        result_image.save(buf, format="WEBP", quality=90)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        return {
            "image": f"data:image/webp;base64,{img_str}",
            "stored": True,
            "message": "Image successfully processed and stored"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def controller_hiht(file: UploadFile, uuid: str):
    try:
        pil_image = Image.open(file.file).convert("RGB")

        res_image = await run_in_threadpool(degradeImg, pil_image)

        # stockage DB
        success = await run_in_threadpool(
            stockImgDb,
            res_image,
            uuid,
            True
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store image in database"
            )

        buf = BytesIO()
        res_image.save(buf, format="WEBP", quality=90)
        buf.seek(0)

        img_str = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "image": f"data:image/webp;base64,{img_str}",
            "stored": True,
            "message": "Image successfully processed and stored"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

