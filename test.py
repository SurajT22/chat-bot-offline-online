why it is comes repetedly i got exact result then why show suggestion
payload
{
  "question": "i want product inspection",
  "online": false
}
response
{
    "answer": "1. Go to Inspection.\n2. Start the machine.\n3. Printed cartons will be automatically inspected based on barcode accuracy, OCR, and print quality.",
    "question": "How do you start production inspection?",
    "confidence": 0.7,
    "source": "semantic",
    "suggestions": [
        {
            "question": "How do you create a new product on the product code server?",
            "answer": "You must define:\n1)Product Code\n2)Company Prefix\n3)Composed GTIN\n4)HRF1 to HRF5 fields (as required)",
            "confidence": 0.46
        },
        {
            "question": "What should I do if my product has a 2D (Data Matrix) barcode for second-level inspection?\n(AGS-TM/TS)",
            "answer": "To configure barcode inspection in second-level processes:\n\n    Open the Inspection panel.\n    Go to Settings → Method (select the appropriate method).\n    Set Barcode_Type to Data_Matrix.\n    Choose the correct Barcode_Size (e.g., 10mm or 8mm).\n    Remaining parameters will auto-adjust based on design logic. (Note: Some manual adjustments may still be needed depending on the product.)\n    Trigger the machine to capture the image; the barcode will then be decoded.",
            "confidence": 0.43
        },
        {
            "question": "How to Inactive Product ?",
            "answer": "Open the Server Settings section.Go to the Product List & Select the Product which you want to inactive. Click on Update Product button. Change Status from Active to Inactive.Click on Save Button. Enter Username & Password for Authentication. Click Ok. After Successfull Authentication Product will be Inactive Successfully.",
            "confidence": 0.42
        }
    ]
}
views.py

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import QA
from .serializers import QASerializer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging
from openai import OpenAI  # correct for openai>=1.0
import requests
# from chat.similarity_utils import get_vectorizer_and_vectors, get_best_match, suggest_similar_questions, \
#     keyword_based_search
from chat.utils import (
    preprocess_text,
    get_tfidf_vectorizer_and_vectors,
    get_sentence_embeddings, get_best_match_tfidf, get_best_match_sentence
)
from chat.similarity_utils import (
    get_vectorizer_and_vectors,
    get_best_match,
    get_best_match_semantic,
    suggest_similar_questions,
    keyword_based_search,
    find_near_exact_match,
    keyword_filter,
find_closest_question
)

from chat.spellcheck_utils import correct_spelling

logger = logging.getLogger(__name__)


class ChatAPIView(APIView):
    @swagger_auto_schema(
        operation_description="Get all Q&A pairs",
        responses={200: QASerializer(many=True)}
    )
    def get(self, request):
        qas = QA.objects.all()
        serializer = QASerializer(qas, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        operation_description="Ask a question and get an AI-generated answer",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['question'],
            properties={
                'question': openapi.Schema(type=openapi.TYPE_STRING, description='User question'),
                'online': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Enable ChatGPT if offline not found')
            },
        ),
        responses={200: "Returns answer from DB or OpenAI"}
    )
    def post(self, request):
        try:
            user_question = request.data.get('question', '').strip()
            online = request.data.get('online', False)

            if not user_question:
                return Response({"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST)

            # First, try to find answer from local DB
            answer_data = self._search_local_database(user_question)
            if answer_data and answer_data.get("answer") and answer_data.get("source") != "none":
                confidence = answer_data.get("confidence", 0)
                suggestions = answer_data.get("suggestions", [])

                CONFIDENCE_THRESHOLD = 0.6

                if confidence >= CONFIDENCE_THRESHOLD:
                    # Good enough answer, return full answer + suggestions if any
                    return Response(answer_data)
                else:
                    # Confidence too low, don't return main answer, just suggestions if any
                    if suggestions:
                        return Response({
                            "answer": "I couldn't find a confident answer, but here are some suggestions:",
                            "suggestions": suggestions
                        })
                # Optionally add to session chat history if you want
                session_history = request.session.get('chat_history', [])
                session_history.append({"question": user_question, "answer": answer_data["answer"]})
                request.session['chat_history'] = session_history[-5:]  # Keep last 5 questions
                return Response(answer_data)

            # If no good local answer and online is enabled, call OpenAI or external APIs
            if online:
                openai_answer = self._get_openai_response(user_question)  # implement your openai call here
                if openai_answer and openai_answer.get("answer"):
                    return Response(openai_answer)

                ddg_answer = self._search_duckduckgo(user_question)
                if ddg_answer and ddg_answer.get("answer"):
                    return Response(ddg_answer)

                wiki_answer = self._search_wikipedia(user_question)
                if wiki_answer and wiki_answer.get("answer"):
                    return Response(wiki_answer)

            # Default fallback message
            return Response({"answer": "Sorry, I couldn't find a good answer."}, status=200)

        except Exception as e:
            logger.error(f"ChatAPIView Error: {str(e)}")
            return Response({"error": "Internal server error", "details": str(e)}, status=500)

 def _build_context(self, history):
        """
        Combine recent questions into a short context string.
        """
        context = ""
        for item in history[-3:]:  # last 3 entries
            context += f"Previous Q: {item['question']} A: {item['answer']} "
        return context
    def _search_local_database(self, user_question):
        try:
            all_qas = QA.objects.filter(question__isnull=False, answer__isnull=False)
            if not all_qas.exists():
                return None

            processed_question = preprocess_text(correct_spelling(user_question))
            questions = [qa.question.strip() for qa in all_qas]
            answers = [qa.answer.strip() for qa in all_qas]

            # Semantic match
            threshold = 0.5 if len(processed_question.split()) <= 3 else 0.75
            semantic_match = get_best_match_semantic(processed_question, questions, answers, threshold=threshold)

            if semantic_match:
                top_question = find_closest_question(semantic_match["question"].strip().lower())
                input_question = find_closest_question(user_question.strip().lower())

                # Add suggestions only if match is weak or different
                if top_question != input_question or semantic_match["confidence"] < 0.7:
                    vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
                    best_index = questions.index(semantic_match["question"])
                    suggestions = suggest_similar_questions(
                        processed_question,
                        questions,
                        answers,
                        vectorizer,
                        question_vectors,
                        exclude_index=best_index
                    )
                    if suggestions:
                        semantic_match["suggestions"] = suggestions[:3]

                return semantic_match

            # TF-IDF fallback
            vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
            tfidf_match = get_best_match(processed_question, questions, answers, vectorizer, question_vectors,
                                         threshold=0.6)
            if tfidf_match:
                best_index = questions.index(tfidf_match["question"])
                suggestions = suggest_similar_questions(
                    processed_question,
                    questions,
                    answers,
                    vectorizer,
                    question_vectors,
                    exclude_index=best_index
                )
                if suggestions and tfidf_match["confidence"] < 0.8:
                    tfidf_match["suggestions"] = suggestions[:10]
                tfidf_match["source"] = "tf-idf"
                return tfidf_match

            # Keyword fallback
            keyword_results = keyword_based_search(processed_question, questions, answers)
            if keyword_results:
                return {
                    "answer": "I found some related information based on keywords.",
                    "source": "keyword",
                    "confidence": 0.4,
                    "suggestions": keyword_results[:10]
                }

            return {
                "answer": "Sorry, I couldn't find a good match in my knowledge base.",
                "source": "none",
                "confidence": 0.0
            }

        except Exception as e:
            logger.error(f"Local DB search error: {str(e)}")
            return {
                "answer": "An error occurred while searching the knowledge base.",
                "source": "error",
                "confidence": 0.0
            }
# similarity_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from chat.spellcheck_utils import correct_spelling
from chat.text_preprocessing import preprocess_text  # we'll create this next
from .models import QA
import torch

import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
all_qa = list(QA.objects.filter(embedding__isnull=False))
all_embeddings = [torch.tensor(qa.embedding) for qa in all_qa]

from sklearn.feature_extraction.text import TfidfVectorizer
from chat.text_preprocessing import preprocess_text  # or inline if needed

def get_vectorizer_and_vectors(questions):
    processed_questions = [preprocess_text(q) for q in questions]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions)
    return vectorizer, question_vectors

def get_best_match(query, questions, answers, vectorizer, question_vectors, threshold=0.6):
    processed_query = preprocess_text(correct_spelling(query))
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score > threshold:
        return {
            "answer": answers[best_index],
            "question": questions[best_index],
            "confidence": round(float(best_score), 2),
            "source": "tf-idf"
        }
    return None
def suggest_similar_questions(query, questions, answers, vectorizer, question_vectors, top_n=3, threshold=0.2, exclude_index=None):
    processed_query = preprocess_text(correct_spelling(query))
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    suggestions = []
    for idx in top_indices:
        if similarities[idx] > threshold and idx != exclude_index:
            suggestions.append({
                "question": questions[idx],
                "answer": answers[idx],
                "confidence": round(float(similarities[idx]), 2)
            })
    return suggestions

def keyword_based_search(query, questions, answers):
    query_words = set(query.lower().split())
    results = []
    for q, a in zip(questions, answers):
        if any(word in q.lower() for word in query_words):
            results.append({
                "question": q,
                "answer": a,
                "source": "keyword"
            })
    return results
def get_best_match_semantic(query, questions, answers, threshold=0.65):
    query_embedding = model.encode(query, convert_to_tensor=True)
    question_embeddings = model.encode(questions, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    best_score = float(similarities.max())
    best_index = int(similarities.argmax())

    if best_score > threshold:
        return {
            "answer": answers[best_index],
            "question": questions[best_index],
            "confidence": round(best_score, 2),
            "source": "semantic"
        }

    return None


import difflib

def find_near_exact_match(user_question, questions, answers, threshold=0.8):
    uq = user_question.strip().lower()
    print(f"[DEBUG] Searching near exact match for: '{uq}'")

    # Substring match
    for idx, q in enumerate(questions):
        q_clean = q.strip().lower()
        if uq in q_clean or q_clean in uq:
            print(f"[DEBUG] Substring match found: '{questions[idx]}'")
            return {
                "answer": answers[idx],
                "question": questions[idx],
                "confidence": 1.0,
                "source": "exact-substring"
            }

    # Fuzzy match
    import difflib
    question_lowers = [q.lower() for q in questions]
    matches = difflib.get_close_matches(uq, question_lowers, n=1, cutoff=threshold)
    print(f"[DEBUG] Fuzzy matches found: {matches}")
    if matches:
        match = matches[0]
        index = question_lowers.index(match)
        print(f"[DEBUG] Fuzzy match chosen: '{questions[index]}'")
        return {
            "answer": answers[index],
            "question": questions[index],
            "confidence": 0.9,
            "source": "fuzzy-exact"
        }

    print("[DEBUG] No near exact match found")
    return None




def keyword_filter(user_question, candidate_question, required_keywords):
    uq = user_question.lower()
    cq = candidate_question.lower()
    for word in required_keywords:
        if word in uq and word not in cq:
            return False
    return True
def find_closest_question(user_input):
    all_qa = list(QA.objects.filter(embedding__isnull=False))
    if not all_qa:
        return None, 0.0

    all_embeddings = [torch.tensor(qa.embedding) for qa in all_qa]

    user_embedding = model.encode(user_input)
    user_tensor = torch.tensor(user_embedding).unsqueeze(0)
    embeddings_tensor = torch.stack(all_embeddings)
    similarities = util.cos_sim(user_tensor, embeddings_tensor)[0]
    best_match_idx = torch.argmax(similarities).item()
    max_sim = similarities[best_match_idx].item()
    if max_sim > 0.5:
        return all_qa[best_match_idx], max_sim
    return None, 0.0
text_processing.py

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from .models import QA

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        # text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        words = text.split()

        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return ' '.join(words)
    except Exception as e:
        logger.error(f"Text preprocessing error: {str(e)}")
        return text

def find_closest_question(user_input):
    user_input = preprocess_text(user_input)
    qa_pairs = QA.objects.all()
    questions = [preprocess_text(qa.question for qa in qa_pairs)]
    return questions
so how to do more intellgently work and very fluently get the answer if it is required then show suggestion answer if not then show error message and working like your chatboat
