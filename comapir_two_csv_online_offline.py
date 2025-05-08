import tkinter as tk
from tkinter import ttk
import csv
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize tokenizer without using punkt (avoids errors)
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# Load questions and answers from CSV
def load_qa_data(question_file, answer_file):
    questions = []
    answers = []
    try:
        with open(question_file, 'r', encoding='utf-8') as qfile:
            qreader = csv.reader(qfile)
            questions = [row[0].strip().lower() for row in qreader if row]
        with open(answer_file, 'r', encoding='utf-8') as afile:
            areader = csv.reader(afile)
            answers = [row[0].strip() for row in areader if row]
        if len(questions) != len(answers):
            raise ValueError("Mismatch in number of questions and answers")
        return questions, answers
    except FileNotFoundError:
        raise FileNotFoundError("CSV files not found. Please check the paths.")
    except Exception as e:
        raise Exception(f"Error loading CSV files: {str(e)}")

# Use your own CSV paths
try:
    questions, answers = load_qa_data(
        'E:/python notes/practise pyton code/test/question.csv',
        'E:/python notes/practise pyton code/test/answer.csv'
    )
except Exception as e:
    print(f"Failed to load CSV files: {str(e)}")
    exit(1)

# Precompute TF-IDF vectors for questions
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def get_best_offline_match(query):
    query_vector = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    if best_score > 0.2:  # Similarity threshold
        return answers[best_idx], best_score
    return None, 0.0

def show_suggestions(query):
    query_vector = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    top_indices = similarities.argsort()[-3:][::-1]  # Top 3 matches
    suggestions = [questions[i] for i in top_indices if similarities[i] > 0.1]
    if suggestions:
        return "Did you mean: " + ", ".join(suggestions) + "?"
    return "No similar questions found."

def search_wikipedia(query):
    try:
        # Search Wikipedia for the best-matching article
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json"
        headers = {"User-Agent": "Chatbot/1.0"}
        response = requests.get(search_url, headers=headers, timeout=5)
        data = response.json()

        if data["query"]["search"]:
            title = data["query"]["search"][0]["title"]
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            response = requests.get(summary_url, headers=headers, timeout=5)
            summary = response.json()
            if "extract" in summary and len(summary["extract"]) > 50:
                return summary["extract"]
        return None
    except Exception as e:
        return None

def search_google(query):
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get DuckDuckGo-style redirect links
        links = soup.find_all("a", href=True)
        real_links = []

        for a in links:
            href = a['href']
            if href.startswith("/l/?uddg="):
                parsed_link = urllib.parse.unquote(href.split("/l/?uddg=")[-1])
                if "wikipedia.org" not in parsed_link:
                    real_links.append(parsed_link)

        if not real_links:
            return "No useful links found in search results."

        first_link = real_links[0]
        page_response = requests.get(first_link, headers=headers, timeout=5)
        page_soup = BeautifulSoup(page_response.text, 'html.parser')

        # Extract the first meaningful paragraph
        for p in page_soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 80:
                return text

        return f"No clear answer found, but here's the source: {first_link}"

    except Exception as e:
        return f"Error during Google search: {str(e)}"

def show_result(text, source, confidence=None):
    chatbox.config(state='normal')
    chatbox.insert(tk.END, f"\nUser: {entry.get()}")
    confidence_text = f" (Confidence: {confidence:.2f})" if confidence else ""
    chatbox.insert(tk.END, f"\n{source} Bot: {text}{confidence_text}\n")
    chatbox.config(state='disabled')
    chatbox.see(tk.END)
    entry.delete(0, tk.END)

def search_answer():
    query = entry.get().lower().strip()
    if not query:
        show_result("Please enter a valid query.", source="System")
        return
    if query in ["exit", "quit"]:
        root.destroy()
        return

    is_online = check_var.get()
    result, confidence = get_best_offline_match(query)

    if result:
        show_result(result, source="Offline", confidence=confidence)
    elif is_online:
        wiki_result = search_wikipedia(query)
        if wiki_result:
            show_result(wiki_result, source="Online (Wikipedia)")
        else:
            google_result = search_google(query)
            show_result(google_result, source="Online (Google)")
    else:
        suggestions = show_suggestions(query)
        show_result(f"No offline result found. Enable online search or try rephrasing.\n{suggestions}", source="System")

# GUI setup
def main():
    global root, entry, chatbox, check_var

    root = tk.Tk()
    root.title("Offline/Online Chatbot")
    root.geometry("700x500")

    # Corrected line: Removed fill and expand from Text constructor
    chatbox = tk.Text(root, height=25, wrap=tk.WORD)
    chatbox.config(state='disabled')  # Start as read-only
    # Moved fill and expand to pack()
    chatbox.pack(pady=(10, 0), padx=10, fill=tk.BOTH, expand=True)

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side=tk.BOTTOM, pady=10)

    entry = tk.Entry(bottom_frame, width=60)
    entry.grid(row=0, column=0, padx=5)
    entry.bind("<Return>", lambda event: search_answer())

    search_button = ttk.Button(bottom_frame, text="Search", command=search_answer)
    search_button.grid(row=0, column=1, padx=5)

    check_var = tk.BooleanVar()
    online_check = tk.Checkbutton(bottom_frame, text="Search Online", variable=check_var)
    online_check.grid(row=0, column=2, padx=5)

    quit_button = ttk.Button(bottom_frame, text="Quit", command=root.destroy)
    quit_button.grid(row=0, column=3, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()