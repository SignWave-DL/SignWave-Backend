from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import time

app = FastAPI()

OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    # create a new file for this session
    filename = OUTPUT_DIR / f"audio_{int(time.time())}.webm"
    f = open(filename, "wb")

    try:
        while True:
            message = await websocket.receive()

            # Binary audio chunk comes as bytes
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                f.write(chunk)

            # Optional text messages (like START/STOP)
            elif "text" in message and message["text"] is not None:
                txt = message["text"]
                if txt == "STOP":
                    break

    except WebSocketDisconnect:
        pass
    finally:
        f.close()
        await websocket.close()
        print(f"Saved: {filename}")
