import os
import io
import re
import time
import asyncio
import logging
from xml.sax.saxutils import escape

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import errors as genai_errors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- CONFIGURATION (set these as Environment Variables on Render) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# "gemini-flash-latest" always points to the current Flash model, so it won't
# expire like a hardcoded version. Override with GEMINI_MODEL if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Tried in order if the main model is overloaded (503) or unavailable (404).
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",")
    if m.strip()
]
RETRY_CODES = {429, 500, 502, 503, 504}
RETRY_DELAYS = [2, 5, 10]    # seconds to wait between retries on the same model

# Optional allow-list. Edit the list, or set ALLOWED_USER_IDS="123,456" on Render.
# Leave empty to allow anyone (not recommended: strangers use your API quota).
ALLOWED_USER_IDS = [
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()
]

COOLDOWN_SECONDS = 15        # minimum gap between requests per user
GEMINI_TIMEOUT_SECONDS = 90  # per-attempt timeout

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_last_request: dict[int, float] = {}

SYSTEM_INSTRUCTION = (
    "You are an expert academic document writer. Generate a well-written document "
    "based on the prompt. Put a clear title on the very first line, then the body. "
    "Use simple markdown: '#' headings, '-' bullets and **bold** only."
)


# ---------- PDF helpers ----------
def _inline(text: str) -> str:
    """Escape XML-special characters, then convert **bold** / *italic* to ReportLab tags."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    return text.replace("*", "")  # drop any leftover stray asterisks


def generate_pdf_bytes(title: str, body_text: str) -> io.BytesIO:
    """Generates a styled PDF in memory without creating temporary files."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40, title=title)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=15)
    h_style = ParagraphStyle("DocHeading", parent=styles["Heading2"], fontSize=14, leading=18,
                             spaceBefore=8, spaceAfter=6)
    body_style = ParagraphStyle("DocBody", parent=styles["Normal"], fontSize=11, leading=15, spaceAfter=8)
    bullet_style = ParagraphStyle("DocBullet", parent=body_style, leftIndent=18, bulletIndent=6, spaceAfter=4)

    story = [Paragraph(_inline(title), title_style), Spacer(1, 6)]
    para_buf: list[str] = []

    def flush_paragraph():
        if para_buf:
            story.append(Paragraph("<br/>".join(_inline(l) for l in para_buf), body_style))
            para_buf.clear()

    for raw in body_text.splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
        elif re.fullmatch(r"[-*_]{3,}", line):  # horizontal rule
            flush_paragraph()
        elif re.match(r"#{1,6}\s+", line):
            flush_paragraph()
            story.append(Paragraph(_inline(re.sub(r"^#{1,6}\s+", "", line)), h_style))
        elif re.match(r"[-*•]\s+", line):
            flush_paragraph()
            story.append(Paragraph(_inline(re.sub(r"^[-*•]\s+", "", line)), bullet_style, bulletText="•"))
        else:
            para_buf.append(line)
    flush_paragraph()

    doc.build(story)
    buffer.seek(0)
    return buffer


def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_")[:40]
    return f"{name or 'Generated_Document'}.pdf"


# ---------- Gemini helper ----------
async def generate_with_retry(prompt: str, status_msg=None) -> str:
    """Calls Gemini with retries + model fallback to survive 503 'high demand' errors."""
    models = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error: Exception | None = None

    for model in models:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = await asyncio.wait_for(
                    gemini_client.aio.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={"system_instruction": SYSTEM_INSTRUCTION},
                    ),
                    timeout=GEMINI_TIMEOUT_SECONDS,
                )
                return (response.text or "").strip()
            except genai_errors.APIError as e:
                last_error = e
                code = getattr(e, "code", None)
                if code in RETRY_CODES and attempt < len(RETRY_DELAYS):
                    logger.warning("%s returned %s, retry %d", model, code, attempt + 1)
                    if status_msg is not None:
                        try:
                            await status_msg.edit_text("⏳ Gemini is busy, retrying...")
                        except Exception:
                            pass  # e.g. "message not modified"
                    await asyncio.sleep(RETRY_DELAYS[attempt])
                    continue
                logger.warning("%s failed with %s, trying next model", model, code)
                break  # 404 / 400 / retries exhausted -> next model
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning("%s timed out, trying next model", model)
                break

    raise last_error if last_error else RuntimeError("No Gemini model available")


# ---------- Telegram handlers ----------
def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Sorry, this is a private student bot.")
        return

    await update.message.reply_text(
        "🎓 <b>Student AI Document Bot</b>\n\n"
        "Send me a prompt describing the document you need.\n"
        "<i>Example: Write an assignment report on Renewable Energy Sources with 3 key sections</i>",
        parse_mode=ParseMode.HTML,
    )


async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not update.message.text:
        return

    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("🚫 Access denied.")
        return

    # Simple per-user rate limit
    now = time.monotonic()
    wait = COOLDOWN_SECONDS - (now - _last_request.get(user_id, 0))
    if wait > 0:
        await update.message.reply_text(f"⏱ Please wait {int(wait) + 1}s before your next request.")
        return
    _last_request[user_id] = now

    user_prompt = update.message.text
    status_msg = await update.message.reply_text("⏳ Generating document with Gemini...")

    try:
        doc_content = await generate_with_retry(user_prompt, status_msg)
        if not doc_content:
            await status_msg.edit_text("⚠️ Gemini returned an empty response. Try rephrasing your prompt.")
            return

        lines = doc_content.split("\n")
        doc_title = re.sub(r"^#+\s*", "", lines[0]).replace("*", "").strip() or "Generated Document"
        body = "\n".join(lines[1:]).strip()  # avoid repeating the title inside the body

        await status_msg.edit_text("📄 Draft complete! Converting to PDF...")

        pdf_buffer = await asyncio.to_thread(generate_pdf_bytes, doc_title, body)

        caption = f"✅ Document ready for: '{user_prompt[:40]}{'...' if len(user_prompt) > 40 else ''}'"
        await update.message.reply_document(
            document=pdf_buffer,
            filename=safe_filename(doc_title),
            caption=caption,
        )
        await status_msg.delete()

    except (asyncio.TimeoutError, genai_errors.ServerError):
        logger.exception("Gemini unavailable after retries")
        await status_msg.edit_text("⚠️ Gemini is overloaded right now. Please try again in a minute.")
    except Exception:
        logger.exception("Document generation failed")  # full traceback in Render logs
        await status_msg.edit_text("⚠️ Something went wrong generating the document. Please try again.")


def main():
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY environment variable.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.UpdateType.MESSAGE, handle_prompt
    ))

    logger.info("Bot started using model %s (fallbacks: %s)", GEMINI_MODEL, FALLBACK_MODELS)
    app.run_polling()


if __name__ == "__main__":
    main()
