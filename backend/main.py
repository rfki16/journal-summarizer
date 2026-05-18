from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

# Load API key dari file .env
load_dotenv()

app = FastAPI()

# CORS: izinkan frontend (GitHub Pages, dll) mengakses backend ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti "*" dengan URL GitHub Pages kamu saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Ekstrak teks dari file PDF menggunakan PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()


def summarize_with_groq(text: str) -> str:
    """Kirim teks ke Groq API dan dapatkan ringkasan."""

    # Batasi panjang teks agar tidak melebihi batas token Groq
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Teks dipotong karena terlalu panjang]"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten akademik yang ahli merangkum jurnal ilmiah. "
                    "Buat ringkasan yang mencakup: "
                    "1) Tujuan penelitian, "
                    "2) Metode yang digunakan, "
                    "3) Hasil utama, "
                    "4) Kesimpulan. "
                    "Gunakan bahasa yang jelas dan mudah dipahami."
                ),
            },
            {
                "role": "user",
                "content": f"Tolong rangkum jurnal berikut ini:\n\n{text}",
            },
        ],
        temperature=0.5,  # 0 = deterministik, 1 = kreatif; 0.5 cocok untuk ringkasan
        max_tokens=1024,
    )

    return response.choices[0].message.content


@app.get("/")
def root():
    """Endpoint health check — untuk memastikan server berjalan."""
    return {"status": "Journal Summarizer API is running!"}


@app.post("/summarize")
async def summarize(
    text: str = Form(None),        # Teks langsung dari input pengguna
    file: UploadFile = File(None), # File PDF yang diupload
):
    """
    Endpoint utama: terima teks atau PDF, kembalikan ringkasan.
    Minimal salah satu (text atau file) harus diisi.
    """

    # Validasi: pastikan ada input
    if not text and not file:
        return {"error": "Harap masukkan teks atau upload file PDF."}

    # Jika ada file PDF, ekstrak teksnya
    if file:
        if not file.filename.endswith(".pdf"):
            return {"error": "File harus berformat PDF."}
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)

        if not extracted_text:
            return {"error": "Gagal mengekstrak teks dari PDF. Pastikan PDF bukan hasil scan gambar."}

        input_text = extracted_text

    else:
        # Gunakan teks langsung dari form
        input_text = text.strip()
        if not input_text:
            return {"error": "Teks tidak boleh kosong."}

    # Kirim ke Groq dan kembalikan hasilnya
    summary = summarize_with_groq(input_text)
    return {"summary": summary}
