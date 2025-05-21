import tkinter as tk
from tkinter import ttk
import csv
import requests
import pyaudio
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from PIL import Image, ImageTk

# === NLTK Setup ===
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# === SerpAPI Setup ===
API_KEY = "YOUR_SERPAPI_KEY"  # 🔑 Replace with your SerpAPI key
def search_wikipedia(query):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "extract" in data and data["extract"]:
                return data["extract"]
        return None
    except Exception as e:
        return f"Error searching Wikipedia: {e}"


def search_duckduckgo(query):
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("AbstractText"):
                return data["AbstractText"]
            if data.get("Answer"):
                return data["Answer"]
        return None
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"

def get_direct_answer(query):
    try:
        # First try Google via SerpAPI
        params = {
            "q": query,
            "api_key": API_KEY,
            "engine": "google"
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        if "answer_box" in results:
            answer_box = results["answer_box"]
            if "answer" in answer_box:
                return answer_box["answer"]
            elif "snippet" in answer_box:
                return answer_box["snippet"]
            elif "highlighted_words" in answer_box:
                return ", ".join(answer_box["highlighted_words"])

        if "organic_results" in results:
            snippet = results["organic_results"][0].get("snippet")
            if snippet:
                return snippet

        # If Google doesn't work, try Wikipedia
        wiki_answer = search_wikipedia(query)
        if wiki_answer:
            return f"(From Wikipedia)\n{wiki_answer}"

        # If Wikipedia doesn't work, try DuckDuckGo
        ddg_answer = search_duckduckgo(query)
        if ddg_answer:
            return f"(From DuckDuckGo)\n{ddg_answer}"

        return "Sorry, I couldn't find a direct answer online."
    except Exception as e:
        return f"Error: {e}"


# === Load QA Data ===
def load_qa_data(qa_file):
    try:
        with open(qa_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            return zip(*[(row[0].strip().lower(), row[1].strip()) for row in reader if len(row) == 2])
    except Exception as e:
        print(f"Error loading QA data: {e}")
        exit(1)

questions, answers = load_qa_data(r'D:\suraj\angular\python_practise\file_test\pythonProject\QA.csv')
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def get_best_offline_match(query, threshold=0.7):
    query_vector = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    best_index = similarities.argmax()
    best_score = similarities[best_index]
    return (answers[best_index], best_score) if best_score > threshold else (None, 0.0)

def suggest_questions(query, top_n=3, suggestion_threshold=0.1):
    query_vector = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [
        (questions[i], answers[i])
        for i in top_indices if similarities[i] > suggestion_threshold
    ]

def display_message(user_query, text, source, confidence=None, is_suggestion=False):
    chatbox.config(state='normal')
    chatbox.insert(tk.END, "\nUser: ", "user_question")
    chatbox.insert(tk.END, f"{user_query}\n", "user_question")
    chatbox.insert(tk.END, f"{source} Bot:\n", "bot_response")

    if is_suggestion and "Q:" in text and "A:" in text:
        lines = text.splitlines()
        in_answer_block = False
        for line in lines:
            if line.startswith("Q:"):
                chatbox.insert(tk.END, line + "\n", "suggestion_question")
                in_answer_block = False
            elif line.startswith("A:"):
                chatbox.insert(tk.END, line + "\n", "suggestion_answer")
                in_answer_block = True
            elif in_answer_block:
                chatbox.insert(tk.END, line + "\n", "suggestion_answer")
            else:
                chatbox.insert(tk.END, line + "\n", "suggestion")
    else:
        chatbox.insert(tk.END, text)
        if confidence is not None:
            chatbox.insert(tk.END, f" (Confidence: {confidence:.2f})")
        chatbox.insert(tk.END, "\n")

    chatbox.config(state='disabled')
    chatbox.see(tk.END)
    entry.delete(0, tk.END)

def handle_search(query=None):
    query = query or entry.get().lower().strip()
    if not query:
        display_message("", "Please enter a valid query.", source="System")
        return
    if query in ["exit", "quit"]:
        root.destroy()
        return

    use_online_search = check_var.get()

    if use_online_search:
        answer = get_direct_answer(query)
        display_message(query, answer, source="Online (Google Direct)")
    else:
        offline_answer, confidence = get_best_offline_match(query)
        if offline_answer:
            display_message(query, offline_answer, source="Offline", confidence=confidence)
        else:
            suggestions = suggest_questions(query)
            if suggestions:
                suggestion_text = "No exact match found. Here are similar questions and answers:\n"
                for q, a in suggestions:
                    suggestion_text += f"\nQ: {q}\nA: {a}\n"
                display_message(query, suggestion_text, source="System", is_suggestion=True)
            else:
                display_message(query, "No offline result found. Enable online search or try rephrasing.", source="System")

# === Voice Input Handler ===
def start_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        chatbox.config(state='normal')
        chatbox.insert(tk.END, "\nSystem: Listening... Please speak.\n", "bot_response")
        chatbox.config(state='disabled')
        chatbox.see(tk.END)
        root.update()

        try:
            audio = recognizer.listen(source, timeout=50)
            chatbox.config(state='normal')
            chatbox.insert(tk.END, "System: Recognizing...\n", "bot_response")
            chatbox.config(state='disabled')
            chatbox.see(tk.END)
            root.update()

            text = recognizer.recognize_google(audio)
            chatbox.config(state='normal')
            chatbox.insert(tk.END, f"You (via voice): {text}\n", "user_question")
            chatbox.config(state='disabled')
            chatbox.see(tk.END)
            root.update()

            handle_search(query=text)

        except sr.UnknownValueError:
            display_message("Voice Input", "Could not understand audio.", source="System")
        except sr.RequestError as e:
            display_message("Voice Input", f"Speech recognition error: {e}", source="System")
        except sr.WaitTimeoutError:
            display_message("Voice Input", "No speech detected. Try again.", source="System")

# === GUI Setup ===
def main():
    global root, entry, chatbox, check_var

    root = tk.Tk()
    root.title("Kevision System Chatbot")
    root.geometry("1100x800")
    root.configure(bg="#f0f0f0")


    # === Top Frame ===
    top_frame = tk.Frame(root, bg="#f0f0f0")
    top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
    root.grid_rowconfigure(0, weight=0)  # ✅ Prevent top frame from expanding
    # top_frame.pack(fill=tk.X,pady=(10, 5))

    try:
        original_image = Image.open(r"D:\suraj\angular\python_practise\file_test\pythonProject\download.png")

        # Resize the image (e.g., scale it to 100x100 pixels)
        resized_image = original_image.resize((180, 60), Image.Resampling.LANCZOS) #230, 80

        # Convert the PIL image to a Tkinter-compatible image
        logo_image = ImageTk.PhotoImage(resized_image)

        # logo_image = tk.PhotoImage(file=r"D:\suraj\angular\python_practise\file_test\pythonProject\download.png")
        logo_label = tk.Label(top_frame, image=logo_image, bg="#f0f0f0")

        logo_label.image = logo_image
        # logo_label.grid(row=0, column=0, sticky='w', padx=(10, 10))
        logo_label.grid(row=0, column=0, sticky='w')
        # logo_label.pack()
    except Exception as e:
        print("Logo not found or failed to load:", e)

    title_label = tk.Label(top_frame, text="Kevision Chatbot Assistant", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
    title_label.grid(row=0, column=1, sticky='w', padx=(15, 0))  # Align title beside the logo
    # title_label.grid(row=0, column=1, sticky='w')
    # title_label.pack(pady=(5, 10))

    # === Chat Display ===
    chat_frame = tk.Frame(root)
    # chat_frame.grid(row=1, column=0, padx=10, pady=20, sticky="nsew")
    chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    root.grid_rowconfigure(1, weight=1)  # ✅ Make chat area expand

    scrollbar = tk.Scrollbar(chat_frame)
    # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar.grid(row=0, column=1, sticky='ns')
    # chatbox = tk.Text(root, height=20, wrap=tk.WORD, state='disabled', font=("Helvetica", 11))
    chatbox = tk.Text(
        chat_frame,
        height=20,
        wrap=tk.WORD,
        state='disabled',
        font=("Segoe UI", 11),
        bg="#ffffff",
        fg="#222222",
        relief=tk.FLAT,
        yscrollcommand=scrollbar.set
    )
    # chatbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    chatbox.grid(row=0, column=0, sticky="nsew")
    scrollbar.config(command=chatbox.yview)

    # Enable chat_frame to resize properly
    chat_frame.grid_rowconfigure(0, weight=1)
    chat_frame.grid_columnconfigure(0, weight=1)

    chatbox.tag_config("user_question", foreground="blue", font=("Helvetica", 10, "bold"))
    chatbox.tag_config("bot_response", foreground="dark green", font=("Helvetica", 10))
    chatbox.tag_config("suggestion", foreground="orange", font=("Helvetica", 10))
    chatbox.tag_config("suggestion_question", foreground="black", font=("Helvetica", 10, "bold"))
    chatbox.tag_config("suggestion_answer", foreground="green", font=("Helvetica", 10))
    # chatbox.pack(padx=10, pady=20, fill=tk.BOTH, expand=True)
    # chatbox.grid(row=1, column=0, padx=10, pady=20, sticky="nsew")
    # Configure rows and columns to expand properly
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)  # For chatbox
    root.grid_rowconfigure(2, weight=1)  # For bottom_frame

    # === Bottom Controls ===
    # bottom_frame = tk.Frame(root, bg="#f0f0f0")
    # # bottom_frame.pack(side=tk.BOTTOM, pady=20)
    # bottom_frame.grid(row=2, column=0, pady=20, sticky="ew")
    #
    # entry = tk.Entry(bottom_frame, width=60, font=("Helvetica", 11))
    # entry.grid(row=0, column=0, padx=10, pady=10)
    #
    # entry.bind("<Return>", lambda event: handle_search())
    #
    # search_button = ttk.Button(bottom_frame, text="Search", command=handle_search)
    # search_button.grid(row=0, column=1, padx=10, pady=10)
    #
    # voice_button = ttk.Button(bottom_frame, text="Start Voice", command=start_voice_input)
    # voice_button.grid(row=0, column=2, padx=10, pady=10)
    #
    # check_var = tk.BooleanVar()
    # online_check = tk.Checkbutton(bottom_frame, text="Search Online", variable=check_var, bg="#f0f0f0")
    # online_check.grid(row=0, column=3, padx=10, pady=10)
    #
    # quit_button = ttk.Button(bottom_frame, text="Quit", command=root.destroy)
    # quit_button.grid(row=0, column=4, padx=10, pady=10)

    bottom_frame = tk.Frame(root, bg="#f0f0f0")
    bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
    root.grid_rowconfigure(2, weight=0)  # Bottom should not expand vertically

    # === Sleek Entry Widget ===
    entry = tk.Entry(bottom_frame, font=("Helvetica", 11), relief=tk.SOLID, bd=1)
    entry.grid(row=0, column=0, padx=(0, 5), pady=5, ipady=4, sticky="ew")
    entry.bind("<Return>", lambda event: handle_search())

    bottom_frame.grid_columnconfigure(0, weight=1)  # Let entry expand horizontally

    # === Buttons with minimal padding ===
    search_button = ttk.Button(bottom_frame, text="Search", command=handle_search)
    search_button.grid(row=0, column=1, padx=5, pady=5)

    voice_button = ttk.Button(bottom_frame, text="🎙️", width=3, command=start_voice_input)
    voice_button.grid(row=0, column=2, padx=5, pady=5)

    check_var = tk.BooleanVar()
    online_check = tk.Checkbutton(bottom_frame, text="Online", variable=check_var, bg="#f0f0f0")
    online_check.grid(row=0, column=3, padx=5, pady=5)

    quit_button = ttk.Button(bottom_frame, text="Quit", command=root.destroy)
    quit_button.grid(row=0, column=4, padx=5, pady=5)

    root.mainloop()
    entry.focus()

if __name__ == "__main__":
    main()
