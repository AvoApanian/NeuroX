# servers/fastApi/routers/imgRouter.py
from fastapi import APIRouter, UploadFile, File, Depends
from servers.fastApi.controllers.imgController import controller_low,controller_hiht
from servers.fastApi.middlewares.authMiddleware import verifyToken  

router = APIRouter()

@router.post("/low")
async def upload_low(
    image: UploadFile = File(...),
    uuid: str = Depends(verifyToken)
):
    return await controller_low(image, uuid)


@router.post("/high")
async def upload_hiht(
    image: UploadFile = File(...),
    uuid: str = Depends(verifyToken)
):
    return await controller_hiht(image, uuid)   