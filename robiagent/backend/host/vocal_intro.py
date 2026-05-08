import os
import asyncio
from thirdparty.volcengine.tts import BidirectionalTTSClient


class VocalIntro(object):
    # TODO: avoid hard-coding
    RESOURCE_ID = "seed-tts-2.0"
    VOICE_TYPE = "zh_female_vv_uranus_bigtts"
    SAMPLE_RATE = 24000

    def __init__(self, task_config):
        self.task_config = task_config

    def play(self):
        async def play_text():
            client = BidirectionalTTSClient(
                appid=os.environ["ARK_TTS_APPID"],
                access_token=os.environ["ARK_TTS_ACCESS_TOKEN"],
                resource_id=self.RESOURCE_ID,
                voice_type=self.VOICE_TYPE,
                sample_rate=self.SAMPLE_RATE,
                encoding="pcm",
            )

            async with client:
                await client.speak(
                    self.task_config.text,
                    chunk_size=self.task_config.chunk_size,
                    chunk_delay=self.task_config.chunk_delay,
                )

        asyncio.run(play_text())  # async to sync
        result = {
            "err_code": 0,
            "detail": f"Played text: {self.task_config.text}"
        }
        return result
