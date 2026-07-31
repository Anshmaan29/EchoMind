from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.middleware.exception_handler import custom_exception_handler
from app.api.middleware.logging_middleware import RequestLoggingMiddleware
from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import EchoMindException
from app.core.logging import logger, setup_logging
from app.database.session import init_db
from app.schemas.common import HealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Lifespan Context Manager managing startup and shutdown tasks."""
    setup_logging()
    logger.info("Initializing EchoMind AI Memory Operating System Backend...")
    
    # Initialize database tables
    await init_db()
    
    yield
    logger.info("Shutting down EchoMind Backend Engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="EchoMind - AI-powered Digital Memory Operating System Backend API",
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Exception Handlers
app.add_exception_handler(EchoMindException, custom_exception_handler)
app.add_exception_handler(Exception, custom_exception_handler)

# Custom Middlewares
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root level health endpoint requirement: GET /health -> {"status":"ok"}
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def root_health_check() -> HealthResponse:
    """Direct root health endpoint returning status: ok."""
    return HealthResponse(status="ok")

# Register API v1 routers
app.include_router(api_v1_router)
app.include_router(api_v1_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
