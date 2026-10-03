import os
import io
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- CONFIGURATION (Use Environment Variables or hardcode for testing) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY")

# OPTIONAL: Add Telegram User IDs of you and your friends to block strangers.
# Send a message to @userinfobot on Telegram to get your numeric ID.
ALLOWED_USER_IDS = []  # Example: [123456789, 987654321]. Leave empty [] to allow anyone.

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def generate_pdf_bytes(title: str, body_text: str) -> io.BytesIO:
    """Generates a styled PDF in memory without creating temporary files."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=15)
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=11, leading=15, spaceAfter=10)
    
    story = [Paragraph(title, title_style), Spacer(1, 10)]
    
    # Strip markdown headers/bolding for PDF rendering
    clean_text = body_text.replace('#', '').replace('*', '')
    for para in clean_text.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 8))
            
    doc.build(story)
    buffer.seek(0)
    return buffer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("🚫 Sorry, this is a private student bot.")
        return

    await update.message.reply_text(
        "🎓 **Student AI Document Bot**\n\n"
        "Send me a prompt describing the document you need.\n"
        "*(Example: 'Write an assignment report on Renewable Energy Sources with 3 key sections')*"
    )

async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("🚫 Access denied.")
        return

    user_prompt = update.message.text
    status_msg = await update.message.reply_text("⏳ Generating document with Gemini...")

    try:
        # Prompt engineering optimized for document structuring
        system_instruction = (
            "You are an expert academic document writer. Generate a well-written document "
            "based on the prompt. Place a clear title on the very first line."
        )

        response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt_text,
)
        
        doc_content = response.text
        lines = doc_content.strip().split('\n')
        doc_title = lines[0].replace('#', '').strip() if lines else "Generated_Document"
        
        await status_msg.edit_text("📄 Draft complete! Converting to PDF...")
        
        # Build PDF and send directly
        pdf_buffer = generate_pdf_bytes(doc_title, doc_content)
        
        await update.message.reply_document(
            document=pdf_buffer,
            filename=f"{doc_title.replace(' ', '_')[:30]}.pdf",
            caption=f"✅ Document ready for: '{user_prompt[:40]}...'"
        )
        await status_msg.delete()

    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text("⚠️ Something went wrong generating the document. Please try again.")

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt))
    
    print("Bot started! Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
