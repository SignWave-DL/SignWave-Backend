from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import traceback

from services.whisper_service  import transcribe_bytes

from services.gloss_service import text_to_gloss

router = APIRouter()

@router.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()

    audio_buffer = bytearray()
    language = "en"

       
        # --- RECEIVE LOOP ---
    try:
            while True:
                msg = await ws.receive()

                # 1. Text Control Messages
                if msg.get("text") is not None:
                    if msg["text"].strip().lower() == "end":
                        break
                    continue

                # 2. Binary Audio Chunks
                if msg.get("bytes") is not None:
                    audio_buffer.extend(msg["bytes"])

    except WebSocketDisconnect:
            # Normal client disconnect
            return
        
    except RuntimeError as e:
            # Starlette/FastAPI raises RuntimeError if 'receive' is called on a closed connection
            if "disconnect" in str(e).lower():
                return
            # If it is a different runtime error, print it and try to notify client
            print(f"RuntimeError: {e}")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
            return

    except Exception as e:
        # Handle unexpected errors in the loop
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
            return

        # --- PROCESSING PHASE ---
        # If we broke out of the loop (received "end"), we proceed here.
        
    if not audio_buffer:
            try:
                await ws.send_json({"type": "error", "message": "No audio received"})
            except:
                pass
            return

    try:
            

            transcript = transcribe_bytes(bytes(audio_buffer), language=language)



            # 3) Text -> Gloss (NLP)
            gloss = text_to_gloss(transcript)


            # 6) Return result to client
            await ws.send_json({
                "type": "result",
                "transcript": transcript,
                "gloss": gloss,
            })

    except Exception as e:
            # Catch processing errors (Whisper, DB, etc.)
            print(f"Error during processing: {e}")
            traceback.print_exc() # Helpful for server logs
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                # If the client is gone, we just suppress the error
                pass