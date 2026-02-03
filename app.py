import asyncio
import logging
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

from openai import AsyncOpenAI
from openai import OpenAIError
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
import imageio_ffmpeg


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
OPENAI_TRANSCRIBE_LANGUAGE = os.getenv("OPENAI_TRANSCRIBE_LANGUAGE", "ru").strip()
OPENAI_ORG = os.getenv("OPENAI_ORG")
openai_client: AsyncOpenAI | None = None
SUMMARY_AI_PROMPT = (
    "Кратко перескажи суть голосового сообщения без ограничений по количеству предложений. "
    "Убери повторы, эмоции и разговорную воду. Пиши ясно и по делу."
)


logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing. Set it in the environment or .env file.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or .env file.")


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


def get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)
    return openai_client


async def transcribe_audio(wav_path: Path, prompt: str | None = None) -> str:
    request: dict[str, str] = {
        "model": OPENAI_TRANSCRIBE_MODEL,
    }
    if OPENAI_TRANSCRIBE_LANGUAGE:
        request["language"] = OPENAI_TRANSCRIBE_LANGUAGE
    if prompt:
        request["prompt"] = prompt

    with wav_path.open("rb") as audio_file:
        transcription = await get_openai_client().audio.transcriptions.create(
            file=audio_file,
            **request,
        )
    text = getattr(transcription, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(transcription, str):
        return transcription.strip()
    raise RuntimeError("Unexpected transcription response format from OpenAI API")


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


async def summarize_text(text: str) -> tuple[str, bool]:

    cleaned = (text or "").strip()
    if not cleaned:
        return "Текст не распознан.", False

    try:
        user_content = f"{SUMMARY_AI_PROMPT}\n\nРасшифровка:\n{cleaned}"

        completion = await get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Ты — инструмент для сжатия информации и извлечения сути."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_completion_tokens=200,
        )
        choice = (completion.choices[0].message.content or "").strip()
        return (choice or "Текст не распознан."), True
    except OpenAIError as exc:  # pragma: no cover - network dependent
        logger.warning("OpenAI summary failed, fallback to local: %s", exc)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Unexpected OpenAI error, fallback to local: %s", exc)

    sentences: list[str] = []
    current = []
    for ch in cleaned:
        current.append(ch)
        if ch in {".", "!", "?"}:
            sentences.append("".join(current).strip())
            current = []
        if len(sentences) >= 2:
            break
    if not sentences and current:
        sentences.append("".join(current).strip())
    summary = " ".join(sentences) if sentences else cleaned
    return (summary[:800] or "Текст не распознан."), False


async def transcribe_and_send(file_id: str, status_message: Message, mode: str) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            ogg_path = tmp_dir_path / "voice.ogg"
            wav_path = tmp_dir_path / "voice.wav"

            await download_voice_file(file_id, ogg_path)
            convert_to_wav(ogg_path, wav_path)
            transcribe_prompt = SUMMARY_AI_PROMPT if mode == "summary" else None
            transcript = await transcribe_audio(wav_path, prompt=transcribe_prompt)

        transcript = transcript or "Текст не распознан."
        if mode == "summary":
            summary, used_openai = await summarize_text(transcript)
            safe = (
                summary.replace("`", "\\`")
                .replace("*", "\\*")
                .replace("_", "\\_")
            )
            suffix = "" if used_openai else "\n_(локальный пересказ, OpenAI недоступен)_"
            await status_message.edit_text(f"🤖Summary AI:\n**{safe}**{suffix}")
        else:
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

    status_message = await callback.message.edit_text("Генерирую расшифровку...")
    await transcribe_and_send(
        file_id=str(payload["file_id"]),
        status_message=status_message,
        mode=mode,
    )


async def main() -> None:
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
