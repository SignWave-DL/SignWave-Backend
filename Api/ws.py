from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.session_repos import create_session, mark_session_done
from db.utterance_repo import create_utterance
from services.storage_service import save_audio_bytes, save_result_json
from services.whisper_service import transcribe
from services.gloss_service import text_to_gloss

router = APIRouter()

@router.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()

    audio_buffer = bytearray()
    language = "en"

    async with AsyncSessionLocal() as db:  # create db session for this websocket
        session = await create_session(db, language=language)
        await ws.send_json({"type": "session", "session_id": str(session.id)})

        try:
            while True:
                msg = await ws.receive()

                # control
                if msg.get("text") is not None:
                    if msg["text"].strip().lower() == "end":
                        break
                    continue

                # binary audio chunk
                if msg.get("bytes") is not None:
                    audio_buffer.extend(msg["bytes"])

        except WebSocketDisconnect:
            return
        except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})
            return

        if not audio_buffer:
            await ws.send_json({"type": "error", "message": "No audio received"})
            return

        # 1) store audio on disk
        audio_path = save_audio_bytes(bytes(audio_buffer), ext="webm")

    try:
            # 2) whisper -> text
            transcript = transcribe(audio_path, language=language)

            # 3) text -> gloss (NLP)
            gloss = text_to_gloss(transcript)

            # 4) store json on disk
            json_path = save_result_json(str(session.id), transcript, gloss)

            # 5) store in Postgres
            utterance = await create_utterance(
                db=db,
                session_id=session.id,
                audio_path=audio_path,
                json_path=json_path,
                transcript=transcript,
                gloss=gloss,
            )

            await mark_session_done(db, session.id)

            # 6) return
            await ws.send_json({
                "type": "result",
                "session_id": str(session.id),
                "utterance_id": str(utterance.id),
                "transcript": transcript,
                "gloss": gloss,
                "audio_path": audio_path,
                "json_path": json_path,
            })

    except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})
