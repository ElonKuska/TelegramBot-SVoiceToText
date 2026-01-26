# SVoiceToText Telegram Bot (Speechmatics)

Бот на aiogram, который расшифровывает голосовые сообщения через Speechmatics Batch API и отвечает расшифровкой в том же чате.

## Требования
- Python 3.10+
- ffmpeg в PATH (или укажите `FFMPEG_BIN`)
- Токен бота Telegram (`BOT_TOKEN`)
- Customer API token Speechmatics (`SPEECHMATICS_API_KEY`)

## Установка
```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Настройка окружения
1. Скопируйте шаблон:
   ```bash
   cp .env.example .env
   ```
2. Заполните в `.env`:
   - `BOT_TOKEN` — токен из @BotFather.
   - `SPEECHMATICS_API_KEY` — Customer API token.
   - При необходимости измените:
     - `SPEECHMATICS_LANGUAGE` (по умолчанию `ru`).
     - `SPEECHMATICS_BASE_URL` (например, `https://eu1.asr.api.speechmatics.com/v2`).
     - `FFMPEG_BIN`, интервалы поллинга.

## Запуск
```bash
python app.py
```
Бот стартует в режиме polling.

## Как пользоваться
- Команда `/start` отправляет приветствие и кнопку «Добавить бота в группу».
- Перешлите или запишите voice‑сообщение в личку или группу:
  - Бот ответит «Генерирую расшифровку…».
  - После готовности отредактирует сообщение и вернет текст в моноширинном блоке.

## Параметры окружения
- `BOT_TOKEN` — токен Telegram.
- `SPEECHMATICS_API_KEY` — ключ Speechmatics.
- `SPEECHMATICS_BASE_URL` — базовый URL API (v2).
- `SPEECHMATICS_LANGUAGE` — язык распознавания (ISO, напр. `ru`, `en`).
- `FFMPEG_BIN` — путь к ffmpeg; установите `auto`, чтобы использовать встроенный бинарник из `imageio-ffmpeg`.
- `POLL_INTERVAL_SECONDS` — частота опроса статуса.
- `POLL_TIMEOUT_SECONDS` — максимальное время ожидания завершения job.
- `LOG_LEVEL` — уровень логов (`INFO`, `DEBUG` и т.д.).

## Примечания
Этот реадме.мд написан фулл вайбкодом и код тоже
- Если системный ffmpeg недоступен, можно задать `FFMPEG_BIN=auto` — будет использован встроенный бинарник из `imageio-ffmpeg`.

