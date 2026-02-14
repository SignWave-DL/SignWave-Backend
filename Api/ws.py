from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import traceback

from services.whisper_service  import transcribe_bytes
from services.ctc_service import transcribe_bytes_ctc

from services.gloss_service import text_to_gloss

router = APIRouter()

@router.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    print(f"🔌 WebSocket connection accepted")

    audio_buffer = bytearray()
    language = "en"
    
    model_type = ws.query_params.get("model", "whisper")
    print(f"🤖 Model selected: {model_type.upper()}")

       
        # --- RECEIVE LOOP ---
    try:
            while True:
                msg = await ws.receive()

                # 1. Text Control Messages
                if msg.get("text") is not None:
                    if msg["text"].strip().lower() == "end":
                        print(f"🛑 Received 'end' signal. Total audio collected: {len(audio_buffer)} bytes")
                        break
                    continue

                # 2. Binary Audio Chunks
                if msg.get("bytes") is not None:
                    chunk_size = len(msg["bytes"])
                    audio_buffer.extend(msg["bytes"])
                    print(f"📦 Received audio chunk: {chunk_size} bytes (total: {len(audio_buffer)} bytes)")

    except WebSocketDisconnect:
            # Normal client disconnect
            print(f"⚠️ Client disconnected. Audio buffer size: {len(audio_buffer)} bytes")
            return
        
    except RuntimeError as e:
            # Starlette/FastAPI raises RuntimeError if 'receive' is called on a closed connection
            if "disconnect" in str(e).lower():
                print(f"⚠️ Runtime disconnect detected")
                return
            # If it is a different runtime error, print it and try to notify client
            print(f"❌ RuntimeError: {e}")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
            return

    except Exception as e:
        # Handle unexpected errors in the loop
            print(f"❌ Unexpected error in receive loop: {e}")
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
            return

        # --- PROCESSING PHASE ---
        # If we broke out of the loop (received "end"), we proceed here.
        
    print(f"📊 Processing phase: buffer size = {len(audio_buffer)} bytes")
    
    if not audio_buffer:
            print(f"⚠️ No audio data received!")
            try:
                await ws.send_json({"type": "error", "message": "No audio received"})
            except:
                pass
            return

    try:
            

            if model_type == "ctc":
                print(f"\n🎤 CTC Model - Starting transcription...")
                import time
                start_time = time.time()
                transcript, confidence = transcribe_bytes_ctc(bytes(audio_buffer), language=language)
                elapsed = time.time() - start_time
                print(f"✅ CTC Transcription completed in {elapsed:.2f}s")
                print(f"📊 Confidence Score: {confidence:.2%}")
                print(f"📝 Transcript: {transcript}")
            else:
                print(f"\n🎤 Whisper Model - Starting transcription...")
                import time
                start_time = time.time()
                transcript, confidence = transcribe_bytes(bytes(audio_buffer), language=language)
                elapsed = time.time() - start_time
                print(f"✅ Whisper Transcription completed in {elapsed:.2f}s")
                print(f"📊 Confidence Score: {confidence:.2%}")
                print(f"📝 Transcript: {transcript}")



            # 3) Text -> Gloss (NLP)
            gloss = text_to_gloss(transcript)
            print(f"🔤 Gloss tokens: {gloss}")


            # 6) Return result to client
            await ws.send_json({
                "type": "result",
                "transcript": transcript,
                "confidence": confidence,
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