import asyncio
import copy
import json
import logging
import queue
import threading
import uuid
from typing import Optional

import sounddevice as sd
import websockets

from thirdparty.volcengine.protocols import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

import contextlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StreamingPCMPlayer:
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        dtype: str = "int16",
        device: Optional[str | int] = None,
        blocksize: int = 0,
        latency: str = "low",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.blocksize = blocksize
        self.latency = latency

        self._queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._error: Optional[BaseException] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)

        if self._error is not None:
            raise RuntimeError(f"Failed to start the player: {self._error}") from self._error

        self._running = True

    def _run(self) -> None:
        try:
            with sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                blocksize=self.blocksize,
                latency=self.latency,
            ) as stream:
                self._started.set()

                while True:
                    chunk = self._queue.get()
                    try:
                        if chunk is None:
                            return
                        stream.write(chunk)
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            self._started.set()

    def write(self, data: bytes) -> None:
        if not data:
            return
        if self._error is not None:
            raise RuntimeError(f"Failed to write to the player: {self._error}") from self._error
        if not self._running:
            self.start()
        self._queue.put(bytes(data))

    def drain(self) -> None:
        """等待队列里的音频都播完。"""
        if self._error is not None:
            raise RuntimeError(f"Failed to drain the player: {self._error}") from self._error
        if self._running:
            self._queue.join()

    def stop(self) -> None:
        if not self._running:
            return
        self._queue.put(None)
        self._queue.join()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._running = False


class BidirectionalTTSClient:
    # Reference: https://www.volcengine.com/docs/6561/1329505?lang=zh#%E4%B8%8B%E8%BD%BD%E4%BB%A3%E7%A0%81%E7%A4%BA%E4%BE%8B

    def __init__(
        self,
        *,
        appid: str,
        access_token: str,
        voice_type: str,
        resource_id: str,
        endpoint: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        encoding: str = "pcm",
        sample_rate: int = 24000,
        enable_timestamp: bool = True,
        disable_markdown_filter: bool = False,
        playback_device: Optional[str | int] = None,
        playback_channels: int = 1,
        playback_dtype: str = "int16",
        max_ws_message_size: int = 10 * 1024 * 1024,
    ) -> None:
        if encoding != "pcm":
            raise ValueError("One should use encoding='pcm' for playing live")

        self.appid = appid
        self.access_token = access_token
        self.voice_type = voice_type
        self.resource_id = resource_id
        self.endpoint = endpoint
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.enable_timestamp = enable_timestamp
        self.disable_markdown_filter = disable_markdown_filter
        self.max_ws_message_size = max_ws_message_size

        self._uid = str(uuid.uuid4())
        self._ws = None
        self._speak_lock = asyncio.Lock()

        self._player = StreamingPCMPlayer(
            sample_rate=sample_rate,
            channels=playback_channels,
            dtype=playback_dtype,
            device=playback_device,
        )

    async def __aenter__(self) -> "BidirectionalTTSClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _build_headers(self) -> dict:
        return {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def _build_base_request(self) -> dict:
        return {
            "user": {
                "uid": self._uid,
            },
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": self.voice_type,
                "audio_params": {
                    "format": self.encoding,
                    "sample_rate": self.sample_rate,
                    "enable_timestamp": self.enable_timestamp,
                },
                "additions": json.dumps(
                    {
                        "disable_markdown_filter": self.disable_markdown_filter,
                    }
                ),
            },
        }

    @staticmethod
    def _iter_text_chunks(text: str, chunk_size: int = 8):
        # Currently simulating text input stream
        if chunk_size <= 0:
            yield text
            return

        punctuations = set("，。！？；：,.!?;:\n")
        buf = []

        for ch in text:
            buf.append(ch)
            if len(buf) >= chunk_size or ch in punctuations:
                yield "".join(buf)
                buf.clear()

        if buf:
            yield "".join(buf)

    async def connect(self) -> None:
        if self._ws is not None:
            return

        headers = self._build_headers()
        safe_headers = dict(headers)
        safe_headers["X-Api-Access-Key"] = "***REDACTED***"

        logger.debug("Connecting to %s with headers: %s", self.endpoint, safe_headers)

        self._ws = await websockets.connect(
            self.endpoint,
            additional_headers=headers,
            max_size=self.max_ws_message_size,
        )

        logger.debug(
            "Connected to WebSocket server, Logid: %s",
            self._ws.response.headers.get("x-tt-logid"),
        )

        await start_connection(self._ws)
        await wait_for_event(
            self._ws,
            MsgType.FullServerResponse,
            EventType.ConnectionStarted,
        )

        await asyncio.to_thread(self._player.start)

    async def close(self) -> None:
        ws = self._ws
        self._ws = None

        try:
            if ws is not None:
                try:
                    await finish_connection(ws)
                    await wait_for_event(
                        ws,
                        MsgType.FullServerResponse,
                        EventType.ConnectionFinished,
                    )
                finally:
                    await ws.close()
                    logger.debug("Connection closed")
        finally:
            await asyncio.to_thread(self._player.stop)

    async def speak(
        self,
        text: str,
        *,
        chunk_size: int = 8,
        chunk_delay: float = 0.01,
    ) -> None:
        if not text or not text.strip():
            return

        await self.connect()

        async with self._speak_lock:
            session_id = str(uuid.uuid4())
            base_request = self._build_base_request()

            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession

            await start_session(
                self._ws,
                json.dumps(start_session_request, ensure_ascii=False).encode("utf-8"),
                session_id,
            )
            await wait_for_event(
                self._ws,
                MsgType.FullServerResponse,
                EventType.SessionStarted,
            )

            async def send_text_chunks():
                try:
                    for chunk in self._iter_text_chunks(text, chunk_size=chunk_size):
                        req = copy.deepcopy(base_request)
                        req["event"] = EventType.TaskRequest
                        req["req_params"]["text"] = chunk

                        await task_request(
                            self._ws,
                            json.dumps(req, ensure_ascii=False).encode("utf-8"),
                            session_id,
                        )

                        if chunk_delay > 0:
                            await asyncio.sleep(chunk_delay)
                finally:
                    await finish_session(self._ws, session_id)

            send_task = asyncio.create_task(send_text_chunks())

            audio_bytes = 0
            try:
                while True:
                    msg = await receive_message(self._ws)

                    if msg.type == MsgType.AudioOnlyServer:
                        if msg.payload:
                            self._player.write(msg.payload)
                            audio_bytes += len(msg.payload)

                    elif msg.type == MsgType.FullServerResponse:
                        if msg.event == EventType.SessionFinished:
                            break

                        if msg.event in (
                            EventType.TTSSentenceStart,
                            EventType.TTSSentenceEnd,
                        ):
                            logger.debug("TTS event=%s payload=%s", msg.event, msg.payload)
                        else:
                            logger.debug("Server event=%s payload=%s", msg.event, msg.payload)

                    else:
                        raise RuntimeError(f"Unexpected message from server: {msg}")

                await send_task

                if audio_bytes == 0:
                    raise RuntimeError("No audio data received")

                await asyncio.to_thread(self._player.drain)

            finally:
                if not send_task.done():
                    send_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await send_task
