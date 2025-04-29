import os
import sys
import whisper
import faiss
import numpy as np
import multiprocessing
import time
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

TRANSCRIPT_FILE = "transcript.txt"
SUMMARY_FILE = "summary.md"

# --- STEP 1: Extract audio from video ---
def extract_audio(video_path, audio_path="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

# --- STEP 2: Transcribe using Whisper ---
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="ru")
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return result["text"]

# --- STEP 3: Split transcript into chunks ---
def split_text(text, max_words=500):
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
llm_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(llm_model, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

# --- STEP 7: Ask question ---
def answer_question(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    response = llm(prompt)[0]['generated_text']
    return response.split("Ответ:")[-1].strip()

# --- STEP 8: Summarize meeting ---
# --- Разделение текста на части с перекрытием ---
def split_text_with_overlap(text, max_words=1500, overlap_words=300):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    chunk = []
    chunk_len = 0

    for sentence in sentences:
        words = sentence.split()
        if chunk_len + len(words) <= max_words:
            chunk.append(sentence)
            chunk_len += len(words)
        else:
            chunks.append(' '.join(chunk))
            # Начинаем новый чанк с перекрытием
            overlap = []
            overlap_len = 0
            for sent in reversed(chunk):
                sent_words = sent.split()
                overlap_len += len(sent_words)
                overlap.insert(0, sent)
                if overlap_len >= overlap_words:
                    break
            chunk = overlap + [sentence]
            chunk_len = sum(len(s.split()) for s in chunk)

    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def summarize_chunk(chunk):
    prompt = f"Вот часть транскрипта встречи:\n{chunk}\n\nСделай краткое резюме этой части на 5-7 предложений. Используй русский язык и не задавай вопросы"
    response = llm(prompt)[0]['generated_text']
    return response

def summarize_transcript_sequential(transcript, max_words=1500, overlap_words=300):
    chunks = split_text_with_overlap(transcript, max_words=max_words, overlap_words=overlap_words)
    print(f"🔹 Разбито на {len(chunks)} частей для последовательного суммирования...")

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"🔸 Обрабатываем часть {i+1}/{len(chunks)}...")
        summary = summarize_chunk(chunk)
        partial_summaries.append(summary)

    full_prompt = "Вот краткие резюме частей встречи:\n\n" + "\n\n".join(partial_summaries) + \
                  "\n\nНа основе этих резюме сделай полное краткое содержание всей встречи, выдели основные темы и принятые решения. Используй русский язык и не задавай вопросы"
    final_summary = llm(full_prompt)[0]['generated_text']
    return final_summary
    
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
    summary = summarize_transcript_map_reduce_multiprocessing(transcript)
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
