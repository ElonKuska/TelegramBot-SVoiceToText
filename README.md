# SVoiceToText Telegram Bot t.me/SVoiceToTextBot (OpenAI Speech-to-Text)

Бот на aiogram, который расшифровывает голосовые сообщения через OpenAI Audio Transcriptions API и отвечает расшифровкой в том же чате.

## Требования
- Python 3.10+
- ffmpeg в PATH (или укажите `FFMPEG_BIN`)
- Токен бота Telegram (`BOT_TOKEN`)
- OpenAI API key (`OPENAI_API_KEY`)

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
   - `OPENAI_API_KEY` — API key OpenAI.
   - `PASSWORD` — пароль доступа (если пусто, вход без пароля).
   - `WORK_GROUP` — `true/false`, отвечать ли в группах.
   - При необходимости измените:
     - `OPENAI_TRANSCRIBE_MODEL` (по умолчанию `gpt-4o-transcribe`).
     - `OPENAI_TRANSCRIBE_LANGUAGE` (по умолчанию `ru`).
     - `FFMPEG_BIN`.

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
- `OPENAI_API_KEY` — ключ OpenAI (обязателен).
- `PASSWORD` — пароль первого входа; если пустой, парольная система отключена.
- `WORK_GROUP` — если `false`, бот полностью игнорирует группы.
- `OPENAI_TRANSCRIBE_MODEL` — модель распознавания (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe` и т.д.).
- `OPENAI_TRANSCRIBE_LANGUAGE` — язык распознавания (ISO, напр. `ru`, `en`).
- `FFMPEG_BIN` — путь к ffmpeg; установите `auto`, чтобы использовать встроенный бинарник из `imageio-ffmpeg`.
- `LOG_LEVEL` — уровень логов (`INFO`, `DEBUG` и т.д.).
- `OPENAI_MODEL` — модель OpenAI для суммаризации (по умолчанию `gpt-4o-mini`).
- `OPENAI_ORG` — ID организации для OpenAI (нужен, если ключ привязан к нескольким организациям).
- `AUTH_DB_PATH` — путь к JSON-базе авторизованных пользователей (по умолчанию `auth_users.json`).

Для режима `Summary AI` бот добавляет внутренний промпт на этапе ASR и затем делает отдельный AI‑пересказ.

Если `PASSWORD` задан: при первом сообщении в личке бот просит пароль. После 5 неверных попыток пользователь блокируется на 24 часа (бот игнорирует сообщения). Успешные входы сохраняются в JSON-базе.

## Примечания

- Этот реадме.мд написан нейронкой (код тоже вообщето)

- `FFMPEG_BIN=auto` — будет использован встроенный бинарник из `imageio-ffmpeg`.
