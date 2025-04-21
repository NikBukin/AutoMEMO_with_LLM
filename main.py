import os
import sys
import whisper
import faiss
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

TRANSCRIPT_FILE = "transcript.txt"
SUMMARY_FILE = "summary.md"

# --- STEP 1: Extract audio from video ---
def extract_audio(video_path, audio_path="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

# --- STEP 2: Transcribe using Whisper ---
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="ru")
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return result["text"]

# --- STEP 3: Split transcript into chunks ---
def split_text(text, max_words=200):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= max_words:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# --- STEP 4: Build vector index ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# --- STEP 5: Search for similar chunks ---
def search_similar_chunks(query, index, chunks, embeddings, k=3):
    query_emb = embed_model.encode([query])
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0]]

# --- STEP 6: Load LLM ---
llm_model = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(llm_model)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# --- STEP 7: Ask question ---
def answer_question(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    response = llm(prompt)[0]['generated_text']
    return response.split("Ответ:")[-1].strip()

# --- STEP 8: Summarize meeting ---
def summarize_transcript(transcript):
    prompt = f"Вот транскрипт встречи:\n{transcript[:3000]}\n\nСделай краткое резюме основных тем и решений."
    response = llm(prompt)[0]['generated_text']
    return response

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nИспользование: python automeeting.py путь_к_видео.mp4\n")
        sys.exit(1)

    video_path = sys.argv[1]
    print("[1/6] Извлекаем аудио из видео...")
    audio_path = extract_audio(video_path)

    print("[2/6] Расшифровываем аудио через Whisper...")
    transcript = transcribe_audio(audio_path)

    print("[3/6] Разбиваем текст и строим индекс...")
    chunks = split_text(transcript)
    index, embeddings, chunk_store = build_index(chunks)

    print("[4/6] Генерируем краткое содержание встречи...")
    summary = summarize_transcript(transcript)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("# 📝 Краткое содержание встречи\n\n" + summary)

    print("\n✅ Резюме сохранено в summary.md")
    print("🗒️ Полная расшифровка сохранена в transcript.txt")

    while True:
        print("\nХотите задать вопрос по встрече? (y/n)")
        if input().lower() != 'y':
            break
        question = input("Введите ваш вопрос: ")
        relevant = search_similar_chunks(question, index, chunk_store, embeddings)
        answer = answer_question(question, relevant)
        print("\n🤖 Ответ:\n", answer)

    print("\nГотово ✨")
