"""Chat API routes"""
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
import os
import io
from app.models import ChatRequest, ChatResponse, FeedbackRequest, Message, TTSRequest
from app.services.chat import generate_chat_response
from app.services.storage import get_storage_service
from typing import Dict

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI assistant using RAG.
    
    - Retrieves relevant content from Doha Magazine
    - Generates contextual answer with citations
    - Stores chat history
    """
    try:
        answer, sources, session_id, message_id = generate_chat_response(message=request.message, session_id=request.session_id)
        
        return ChatResponse(answer=answer, sources=sources, session_id=session_id, message_id=message_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a chat response.
    
    - Updates feedback directly in the message
    """
    try:
        print(f"[API] Received feedback request: session={request.session_id}, message={request.message_id}, rating={request.rating}")
        
        storage = get_storage_service()
        
        if not storage._messages_container:
            print("[API] ERROR: Messages container not available")
            raise HTTPException(status_code=500, detail="Storage not available")
        
        # Use save_feedback method which updates both message and statistics
        print(f"[API] Calling save_feedback with session_id={request.session_id}, message_id={request.message_id}, rating={request.rating}")
        feedback_id = storage.save_feedback(session_id=request.session_id,message_id=request.message_id,rating=request.rating)
        
        print(f"[API] save_feedback returned: {feedback_id}")
        
        if feedback_id:
            print(f"[API] SUCCESS: Feedback updated successfully with ID: {feedback_id}")
            return {"status": "success", "message": "Feedback updated successfully", "feedback_id": feedback_id}
        else:
            print(f"[API] ERROR: Failed to update feedback - save_feedback returned empty string")
            raise HTTPException(status_code=500, detail="Failed to update feedback - message not found or update failed")
            
    except Exception as e:
        print(f"[API] EXCEPTION: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@router.get("/chat/history")
async def get_conversation_history(session_id: str = Query(..., description="Session ID to retrieve history for")):
    """
    Get conversation history for a specific session.
    
    Returns list of messages in chronological order.
    """
    storage = get_storage_service()
    
    try:
        messages = list[Message](storage.get_session_messages(session_id, limit=50))
        
        # Convert to simple format for frontend
        history = []
        for msg in messages:
            history.append({
                "id": msg.id,
                "role": msg.role,
                "text": msg.text,
                "feedback": msg.feedback,
                "sources": [{"title": s.title, "url": s.url} for s in msg.sources] if msg.sources else []
            })
        
        return history
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation history: {str(e)}")


@router.get("/feedback/statistics/{session_id}")
async def get_feedback_statistics(session_id: str) -> Dict:
    """
    Get feedback statistics for a chat session.
    
    Returns:
        Dictionary with feedback statistics including:
        - positive: Number of positive feedback
        - negative: Number of negative feedback  
        - null: Number of no feedback
        - total_assistant_messages: Total assistant messages
        - positive_ratio: Ratio of positive feedback
        - negative_ratio: Ratio of negative feedback
        - overall_feedback: Overall sentiment (positive/negative/neutral)
    """
    try:
        storage = get_storage_service()
        
        if not storage._messages_container:
            raise HTTPException(status_code=500, detail="Storage not available")
        
        # Get feedback statistics
        statistics = storage.get_feedback_statistics(session_id)
        
        if not statistics:
            # If no statistics exist, calculate them
            statistics = storage.update_feedback_statistics(session_id)
        
        return {
            "session_id": session_id,
            "statistics": statistics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback statistics: {str(e)}")


@router.post("/feedback/statistics/{session_id}/update")
async def force_update_feedback_statistics(session_id: str) -> Dict:
    """
    Force update feedback statistics for a session.
    
    This is useful for existing sessions that may not have statistics calculated yet.
    
    Returns:
        Dictionary with updated feedback statistics
    """
    try:
        storage = get_storage_service()
        
        if not storage._messages_container:
            raise HTTPException(status_code=500, detail="Storage not available")
        
        # Force update statistics
        statistics = storage.force_update_statistics(session_id)
        
        return {
            "session_id": session_id,
            "statistics": statistics,
            "message": "Statistics updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update feedback statistics: {str(e)}")


@router.post("/feedback/statistics/{session_id}/recalculate")
async def recalculate_feedback_statistics(session_id: str) -> Dict:
    """
    Recalculate feedback statistics from scratch for a session.
    
    This ensures all feedbacks are counted correctly and fixes any discrepancies.
    
    Returns:
        Dictionary with recalculated feedback statistics
    """
    try:
        storage = get_storage_service()
        
        if not storage._messages_container:
            raise HTTPException(status_code=500, detail="Storage not available")
        
        # Recalculate statistics from scratch
        statistics = storage.recalculate_all_statistics(session_id)
        
        return {
            "session_id": session_id,
            "statistics": statistics,
            "message": "Statistics recalculated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recalculate feedback statistics: {str(e)}")


# ========== Speech-to-Text (local) ==========

def _load_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT unavailable: {str(e)}")
    # Use base model for faster processing (smaller and faster than medium)
    # Options: tiny, base, small, medium, large (tiny/base fastest, large most accurate)
    device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_size = os.environ.get("WHISPER_MODEL", "base")  # Changed default from medium to base
    return WhisperModel(model_size, device=device, compute_type=compute_type)


_whisper_model = None


@router.post("/stt")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe uploaded WAV audio to text using local faster-whisper (Arabic).

    Expects WAV format audio files recorded directly in the browser.
    """
    import tempfile
    global _whisper_model
    tmp_path = None
    try:
        # Save uploaded WAV file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # Lazy-load model once
        if _whisper_model is None:
            _whisper_model = _load_whisper_model()

        # Transcribe audio with speed optimizations
        # beam_size=1 uses greedy decoding (faster than default beam_size=5)
        # vad_filter=True helps skip silence quickly
        segments, info = _whisper_model.transcribe(
            tmp_path, 
            language="ar",
            beam_size=1,  # Faster decoding (greedy instead of beam search)
            vad_filter=True,  # Skip silence for faster processing
            vad_parameters=dict[str, int](min_silence_duration_ms=500)  # Skip short silence
        )
        text = "".join(seg.text for seg in segments).strip() or ""
        
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ========== Text-to-Speech (backend) ==========

# Cache for Arabic voice (lazy-loaded)
# Note: Cache is cleared on server restart, ensuring fresh voice lookup
_arabic_voice_cache = None

async def _get_arabic_voice():
    """Get Arabic voice - dynamically finds and caches an Arabic voice from edge-tts."""
    global _arabic_voice_cache
    
    # Return cached voice if available
    if _arabic_voice_cache is not None:
        return _arabic_voice_cache
    
    try:
        import edge_tts
        # List all available voices
        voices = await edge_tts.list_voices()
        
        # Find Arabic voices (prefer Saudi Arabia, then any Arabic locale)
        arabic_voices = [
            v for v in voices 
            if 'ar' in v.get('Locale', '').lower()
        ]
        
        if not arabic_voices:
            # Fallback to hardcoded voice if no Arabic voices found
            _arabic_voice_cache = "ar-SA-ZariyahNeural"
            return _arabic_voice_cache
        
        # Prefer Saudi Arabic female voice, otherwise use first available Arabic voice
        preferred_voice = None
        for v in arabic_voices:
            locale = v.get('Locale', '')
            gender = v.get('Gender', '')
            short_name = v.get('ShortName', '') or v.get('Name', '')  # Use ShortName if available, fallback to Name
            
            # Prefer ar-SA female voice
            if locale == 'ar-SA' and gender == 'Female':
                preferred_voice = short_name
                break
        
        # If no preferred voice found, use first Arabic voice
        if preferred_voice is None:
            preferred_voice = arabic_voices[0].get('ShortName', '') or arabic_voices[0].get('Name', '')
        
        # Cache the voice
        _arabic_voice_cache = preferred_voice
        return _arabic_voice_cache
        
    except Exception as e:
        # Fallback to hardcoded voice on error
        _arabic_voice_cache = "ar-SA-ZariyahNeural"
        return _arabic_voice_cache

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using edge-tts (Arabic).
    
    Returns audio as MP3 stream.
    """
    try:
        import edge_tts
        text = request.text.strip()
        
        # Validate text
        if not text or len(text) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Get Arabic voice (async lookup with caching)
        arabic_voice = await _get_arabic_voice()
        
        # Generate audio
        communicate = edge_tts.Communicate(text, arabic_voice)
        audio_data = b""
        chunk_count = 0
        audio_chunk_count = 0
        
        async for chunk in communicate.stream():
            chunk_count += 1
            if chunk.get("type") == "audio":
                audio_chunk_count += 1
                if "data" in chunk:
                    audio_data += chunk["data"]
        
        # Validate that we received audio data
        if len(audio_data) == 0:
            error_detail = f"No audio was received. Voice: {arabic_voice}, Chunks: {chunk_count}, Audio chunks: {audio_chunk_count}"
            raise HTTPException(status_code=500, detail=error_detail)
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="TTS unavailable: edge-tts not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

