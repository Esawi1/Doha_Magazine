"""Main FastAPI application for Doha Magazine AI Assistant"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.routers import chat

# Create FastAPI app
app = FastAPI(
    title="Doha Magazine AI Assistant",
    description="AI-powered search and chat assistant for Doha Magazine content",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))




    # Fallback if no frontend is built
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doha Magazine AI Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🎯 Doha Magazine AI Assistant API</h1>
        <p>Welcome to the Doha Magazine AI Assistant backend.</p>
        
        <h2>📚 API Documentation</h2>
        <p>Interactive API docs: <a href="/docs">/docs</a></p>
        <p>Alternative docs: <a href="/redoc">/redoc</a></p>
        
        <h2>🔌 Available Endpoints</h2>
        
        <div class="endpoint">
            <strong>POST /api/chat</strong><br>
            Chat with AI assistant using RAG
        </div>
        
        <div class="endpoint">
            <strong>POST /api/feedback</strong><br>
            Submit feedback on responses
        </div>
        
        <div class="endpoint">
            <strong>GET /api/conversation/{session_id}</strong><br>
            Get conversation history
        </div>
        
        <div class="endpoint">
            <strong>GET /api/feedback/statistics/{session_id}</strong><br>
            Get feedback statistics for a session
        </div>
        
        
        
        <h2>🚀 Quick Start</h2>
        <ol>
            <li>Configure your <code>.env</code> file with Azure credentials</li>
            <li>Start chatting: <code>POST /api/chat</code> with a message</li>
        </ol>
    </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "doha-magazine-ai"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

