import jwt
import os
from fastapi import HTTPException, Header
from dotenv import load_dotenv
from pathlib import Path

# Charger le .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

print(f"Loading .env from: {ENV_PATH}")
print(f".env exists: {ENV_PATH.exists()}")

load_dotenv(ENV_PATH)

# Essayer les deux noms possibles
TOKEN_SECRET = os.getenv("accesSecret") or os.getenv("accessSecret")

print(f"=== ENV LOADING DEBUG ===")
print(f"accesSecret: {os.getenv('accesSecret')}")
print(f"accessSecret: {os.getenv('accessSecret')}")
print(f"TOKEN_SECRET final: {TOKEN_SECRET[:20] if TOKEN_SECRET else 'NOT LOADED'}")
print(f"=========================")

if not TOKEN_SECRET:
    raise RuntimeError("TOKEN_SECRET not found in .env file! Check your .env configuration.")

def verifyToken(authorization: str = Header(None)) -> str:
    """
    Vérifie le token JWT et retourne l'UUID de l'utilisateur
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="No Authorization header")
    
    try:
        # Extraire le token du header "Bearer <token>"
        parts = authorization.split(" ")
        
        if len(parts) != 2 or parts[0] != "Bearer":
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        token = parts[1]
        
        # Décoder le token JWT
        payload = jwt.decode(token, TOKEN_SECRET, algorithms=["HS256"])
        
        # Extraire l'UUID
        uuid = payload.get("id")
        if not uuid:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        print(f"Token verified for user: {uuid}")
        return uuid
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")