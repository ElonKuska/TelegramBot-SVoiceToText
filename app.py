import asyncio
import json
import logging
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from openai import AsyncOpenAI
from openai import OpenAIError
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
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
PASSWORD = os.getenv("PASSWORD", os.getenv("password", "")).strip()
WORK_GROUP_RAW = os.getenv("WORK_GROUP", os.getenv("work_group", "true")).strip().lower()
WORK_GROUP = WORK_GROUP_RAW in {"1", "true", "yes", "on"}
PASSWORD_ENABLED = bool(PASSWORD)
PASSWORD_MAX_ATTEMPTS = 5
PASSWORD_LOCK_SECONDS = 24 * 60 * 60
AUTH_DB_PATH = Path(os.getenv("AUTH_DB_PATH", "auth_users.json"))
openai_client: AsyncOpenAI | None = None
OPENAI_PROMPT = (
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
password_prompt_messages: dict[int, int] = {}
auth_db_lock = asyncio.Lock()


def load_auth_db() -> tuple[set[int], dict[int, dict[str, float]]]:
    if not AUTH_DB_PATH.exists():
        return set(), {}
    try:
        payload = json.loads(AUTH_DB_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read auth db, starting fresh: %s", exc)
        return set(), {}

    raw_users = payload.get("authorized_users", [])
    raw_attempts = payload.get("failed_attempts", {})
    authorized_users: set[int] = set()
    failed_attempts: dict[int, dict[str, float]] = {}

    if isinstance(raw_users, list):
        for value in raw_users:
            try:
                authorized_users.add(int(value))
            except (TypeError, ValueError):
                continue

    if isinstance(raw_attempts, dict):
        for key, value in raw_attempts.items():
            try:
                user_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            count = int(value.get("count", 0))
            locked_until = float(value.get("locked_until", 0))
            failed_attempts[user_id] = {"count": float(count), "locked_until": locked_until}

    return authorized_users, failed_attempts


authorized_users, failed_attempts = load_auth_db()


def save_auth_db() -> None:
    payload = {
        "authorized_users": sorted(authorized_users),
        "failed_attempts": {
            str(user_id): {
                "count": int(data.get("count", 0)),
                "locked_until": float(data.get("locked_until", 0)),
            }
            for user_id, data in failed_attempts.items()
        },
    }
    AUTH_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remaining_attempts_from_count(count: int) -> int:
    return max(0, PASSWORD_MAX_ATTEMPTS - count)


async def get_user_status(user_id: int) -> tuple[bool, bool, int]:
    async with auth_db_lock:
        if user_id in authorized_users:
            return True, False, PASSWORD_MAX_ATTEMPTS

        entry = failed_attempts.get(user_id)
        if not entry:
            return False, False, PASSWORD_MAX_ATTEMPTS

        now_ts = time.time()
        locked_until = float(entry.get("locked_until", 0))
        if locked_until > now_ts:
            return False, True, 0

        if locked_until and locked_until <= now_ts:
            failed_attempts.pop(user_id, None)
            save_auth_db()
            return False, False, PASSWORD_MAX_ATTEMPTS

        count = int(entry.get("count", 0))
        return False, False, remaining_attempts_from_count(count)


async def register_failed_password(user_id: int) -> tuple[int, bool]:
    async with auth_db_lock:
        now_ts = time.time()
        entry = failed_attempts.get(user_id, {"count": 0.0, "locked_until": 0.0})

        locked_until = float(entry.get("locked_until", 0))
        if locked_until and locked_until <= now_ts:
            entry = {"count": 0.0, "locked_until": 0.0}

        count = int(entry.get("count", 0)) + 1
        locked = count >= PASSWORD_MAX_ATTEMPTS
        next_locked_until = now_ts + PASSWORD_LOCK_SECONDS if locked else 0.0
        failed_attempts[user_id] = {"count": float(count), "locked_until": next_locked_until}
        save_auth_db()
        return remaining_attempts_from_count(count), locked


async def mark_user_authorized(user_id: int) -> None:
    async with auth_db_lock:
        authorized_users.add(user_id)
        failed_attempts.pop(user_id, None)
        save_auth_db()


async def delete_message_safely(chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except TelegramBadRequest:
        return


async def send_welcome_message(chat_id: int) -> None:
    bot_username = await get_bot_username()
    keyboard = build_start_keyboard(bot_username)
    await bot.send_message(
        chat_id,
        "Привет, это бот который голосовые сообщения переводит в текстовый вариант.\n"
        "Просто перешли любое голосовое сообщение, например от друга, и я его расшифрую!\n"
        "Меня также можно добавить в группы!",
        reply_markup=keyboard,
    )


async def show_password_prompt(chat_id: int, user_id: int, text: str) -> None:
    previous_prompt_id = password_prompt_messages.pop(user_id, None)
    if previous_prompt_id:
        await delete_message_safely(chat_id=chat_id, message_id=previous_prompt_id)
    prompt = await bot.send_message(chat_id, text)
    password_prompt_messages[user_id] = prompt.message_id


async def ensure_user_access(message: Message) -> bool:
    if message.chat.type in {"group", "supergroup"} and not WORK_GROUP:
        return False

    if not PASSWORD_ENABLED:
        return True

    user = message.from_user
    if not user:
        return False

    user_id = user.id
    is_authorized, is_locked, remaining = await get_user_status(user_id)
    if is_authorized:
        return True
    if is_locked:
        return False

    # Пароль спрашиваем в личке, в группах для неавторизованных пользователей просто молчим.
    if message.chat.type != "private":
        return False

    text = (message.text or "").strip()
    has_prompt = user_id in password_prompt_messages

    if not has_prompt or text.startswith("/"):
        await show_password_prompt(
            chat_id=message.chat.id,
            user_id=user_id,
            text=f"Введите пароль (осталось {remaining} попыток):",
        )
        return False

    await delete_message_safely(chat_id=message.chat.id, message_id=message.message_id)
    previous_prompt_id = password_prompt_messages.pop(user_id, None)
    if previous_prompt_id:
        await delete_message_safely(chat_id=message.chat.id, message_id=previous_prompt_id)

    if text == PASSWORD:
        await mark_user_authorized(user_id)
        await bot.send_message(message.chat.id, "Пароль верный.")
        await send_welcome_message(message.chat.id)
        return False

    remaining_after_try, locked = await register_failed_password(user_id)
    if locked:
        return False
    await show_password_prompt(
        chat_id=message.chat.id,
        user_id=user_id,
        text=f"Пароль неверный (осталось {remaining_after_try} попыток):",
    )
    return False


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
    if not await ensure_user_access(message):
        return
    await send_welcome_message(message.chat.id)


@dp.message(F.voice)
async def handle_voice(message: Message) -> None:
    if not await ensure_user_access(message):
        return

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
        user_content = f"{OPENAI_PROMPT}\n\nРасшифровка:\n{cleaned}"

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
    except OpenAIError as exc:
        logger.warning("OpenAI summary failed, fallback to local: %s", exc)
    except Exception as exc:
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
            transcribe_prompt = OPENAI_PROMPT if mode == "summary" else None
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
    if callback.message and callback.message.chat.type in {"group", "supergroup"} and not WORK_GROUP:
        return

    user = callback.from_user
    if not user:
        return

    if PASSWORD_ENABLED:
        is_authorized, is_locked, _ = await get_user_status(user.id)
        if not is_authorized:
            if not is_locked and callback.message and callback.message.chat.type == "private":
                await callback.answer("Сначала введите пароль.", show_alert=True)
            return

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


@dp.message()
async def handle_other_messages(message: Message) -> None:
    if not await ensure_user_access(message):
        return
    if message.chat.type == "private":
        await message.reply("Отправь голосовое сообщение.")


async def main() -> None:
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
