import asyncio
import json
import logging
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
import imageio_ffmpeg


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_BASE_URL = os.getenv("SPEECHMATICS_BASE_URL", "https://eu1.asr.api.speechmatics.com/v2")
SPEECHMATICS_LANGUAGE = os.getenv("SPEECHMATICS_LANGUAGE", "ru")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
POLL_TIMEOUT_SECONDS = float(os.getenv("POLL_TIMEOUT_SECONDS", "90.0"))


logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing. Set it in the environment or .env file.")

if not SPEECHMATICS_API_KEY:
    raise RuntimeError("SPEECHMATICS_API_KEY is missing. Set it in the environment or .env file.")


bot = Bot(BOT_TOKEN, parse_mode=ParseMode.MARKDOWN)
dp = Dispatcher()
bot_username_cache: str | None = None
pending_requests: dict[str, dict[str, int | str]] = {}


async def get_bot_username() -> str:
    global bot_username_cache
    if bot_username_cache:
        return bot_username_cache
    me = await bot.get_me()
    bot_username_cache = me.username or "voice_to_text_bot"
    return bot_username_cache


async def download_voice_file(file_id: str, dest: Path) -> None:
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, destination=dest)


def resolve_ffmpeg_bin() -> str:
    if FFMPEG_BIN and FFMPEG_BIN != "auto":
        return FFMPEG_BIN
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_to_wav(src: Path, dest: Path) -> None:
    ffmpeg_bin = resolve_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dest),
    ]
    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({completed.returncode}): {completed.stderr.decode(errors='ignore')}"
        )


async def create_job(session: ClientSession, wav_path: Path) -> str:
    url = f"{SPEECHMATICS_BASE_URL.rstrip('/')}/jobs"
    config = {
        "type": "transcription",
        "transcription_config": {
            "language": SPEECHMATICS_LANGUAGE,
        },
    }
    data = {
        "config": json.dumps(config),
    }
    form = aiohttp.FormData()
    form.add_field("config", data["config"])
    form.add_field(
        "data_file",
        wav_path.open("rb"),
        filename=wav_path.name,
        content_type="audio/wav",
    )
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}

    async with session.post(url, data=form, headers=headers) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"Speechmatics job create failed ({resp.status}): {text}")
        payload = await resp.json()
    job_id = payload.get("id") or payload.get("job", {}).get("id")
    if not job_id:
        raise RuntimeError(f"Cannot read job id from response: {payload}")
    return job_id


async def wait_for_job_done(session: ClientSession, job_id: str) -> str:
    url = f"{SPEECHMATICS_BASE_URL.rstrip('/')}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}
    elapsed = 0.0

    while elapsed <= POLL_TIMEOUT_SECONDS:
        async with session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"Speechmatics job status failed ({resp.status}): {text}")
            payload = await resp.json()
        status = payload.get("job", {}).get("status") or payload.get("status")
        if status in {"done", "rejected", "expired", "deleted"}:
            return status
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    raise TimeoutError(f"Speechmatics job {job_id} did not finish in {POLL_TIMEOUT_SECONDS}s")


async def fetch_transcript(session: ClientSession, job_id: str) -> str:
    url = f"{SPEECHMATICS_BASE_URL.rstrip('/')}/jobs/{job_id}/transcript"
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}
    params = {"format": "txt"}

    async with session.get(url, headers=headers, params=params) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"Speechmatics transcript failed ({resp.status}): {text}")
        return (await resp.text()).strip()


def build_start_keyboard(bot_username: str) -> InlineKeyboardMarkup:
    invite_link = f"https://t.me/{bot_username}?startgroup=true"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Добавить бота в группу", url=invite_link)],
        ]
    )


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    bot_username = await get_bot_username()
    keyboard = build_start_keyboard(bot_username)
    await message.answer(
        "Привет, это бот который голосовые сообщения переводит в текстовый вариант.\n"
        "Просто перешли любое голосовое сообщение, например от друга, и я его расшифрую!\n"
        "Меня также можно добавить в группы!",
        reply_markup=keyboard,
    )


@dp.message(F.voice)
async def handle_voice(message: Message) -> None:
    if message.chat.type in {"group", "supergroup"}:
        status_message = await message.reply("Генерирую расшифровку...")
        await transcribe_and_send(
            file_id=message.voice.file_id,
            chat_id=message.chat.id,
            status_message=status_message,
            mode="full",
        )
        return

    request_id = secrets.token_hex(6)
    pending_requests[request_id] = {"file_id": message.voice.file_id, "user_id": message.from_user.id}
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Полная расшифровка", callback_data=f"tr:full:{request_id}")],
            [InlineKeyboardButton(text="Summary AI", callback_data=f"tr:summary:{request_id}")],
        ]
    )
    await message.reply("Выберите вид:", reply_markup=keyboard)


def summarize_text(text: str, max_sentences: int = 2, max_chars: int = 800) -> str:
    """Very lightweight extractive summary: keep the first few sentences within limits."""

    cleaned = text.strip()
    if not cleaned:
        return "Текст не распознан."

    sentences: list[str] = []
    current = []
    for ch in cleaned:
        current.append(ch)
        if ch in {".", "!", "?"}:
            sentences.append("".join(current).strip())
            current = []
        if len(sentences) >= max_sentences:
            break
    if not sentences and current:
        sentences.append("".join(current).strip())

    summary = " ".join(sentences) if sentences else cleaned
    summary = summary[:max_chars]
    return summary or "Текст не распознан."


async def transcribe_and_send(file_id: str, chat_id: int, status_message: Message, mode: str) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            ogg_path = tmp_dir_path / "voice.ogg"
            wav_path = tmp_dir_path / "voice.wav"

            await download_voice_file(file_id, ogg_path)
            convert_to_wav(ogg_path, wav_path)

            timeout = ClientTimeout(total=POLL_TIMEOUT_SECONDS + 30)
            async with ClientSession(timeout=timeout) as session:
                job_id = await create_job(session, wav_path)
                status = await wait_for_job_done(session, job_id)
                if status != "done":
                    raise RuntimeError(f"Job finished with status {status}")
                transcript = await fetch_transcript(session, job_id)

        transcript = transcript or "Текст не распознан."
        if mode == "summary":
            transcript = summarize_text(transcript)

        safe_text = transcript.replace("`", "\\`")
        await status_message.edit_text(f"```\n{safe_text}\n```")
    except Exception as exc:
        logger.exception("Failed to transcribe voice: %s", exc)
        await status_message.edit_text("Не удалось расшифровать сообщение. Попробуй еще раз позже.")


@dp.callback_query(F.data.startswith("tr:"))
async def handle_choice(callback: CallbackQuery) -> None:
    await callback.answer()

    try:
        _, mode, request_id = callback.data.split(":", 2)
    except ValueError:
        await callback.message.edit_text("Не удалось обработать выбор. Отправь голосовое еще раз.")
        return

    payload = pending_requests.pop(request_id, None)
    if not payload:
        await callback.message.edit_text("Запрос устарел. Отправь голосовое еще раз.")
        return

    expected_user = payload.get("user_id")
    if expected_user and callback.from_user and callback.from_user.id != expected_user:
        await callback.message.answer("Это голосовое принадлежит другому пользователю. Отправь свое сообщение.")
        return

    status_message = await callback.message.answer("Генерирую расшифровку...")
    await transcribe_and_send(
        file_id=str(payload["file_id"]),
        chat_id=callback.message.chat.id,
        status_message=status_message,
        mode=mode,
    )


async def main() -> None:
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
