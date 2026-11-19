"""Entrypoint local : lance uvicorn sur l'app FastAPI (api.app).

Pour le developpement local :
    python app.py

En production (HF Spaces) : le Dockerfile execute directement uvicorn.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, log_level="info")
