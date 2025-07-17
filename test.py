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

    get_best_match_semantic,
    suggest_similar_questions,
    keyword_based_search,
    find_near_exact_match,


)
# get_best_match,keyword_filter,find_closest_question
from chat.spellcheck_utils import correct_spelling
import random

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

            # Search local database
            answer_data = self._search_local_database(user_question)
            if answer_data and answer_data.get("answer") and answer_data.get("source") != "none":
                confidence = answer_data.get("confidence", 0)
                CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for confident answers

                # Format response with a conversational template
                templates = [
                    "Here's what I found: {}",
                    "Based on my knowledge, {}",
                    "The answer is: {}",
                    "Let me help you with that: {}",
                ]
                response = random.choice(templates).format(answer_data["answer"])

                # Only include suggestions if confidence is low
                response_data = {
                    "question": answer_data["question"],
                    "answer": response,
                    "confidence": confidence,
                    "source": answer_data["source"],
                }
                if confidence < CONFIDENCE_THRESHOLD:
                    response_data["suggestions"] = answer_data.get("suggestions", [])

                # Update session history
                session_history = request.session.get('chat_history', [])
                session_history.append({"question": user_question, "answer": response})
                request.session['chat_history'] = session_history[-5:]  # Keep last 5 questions
                return Response(response_data)

            # If online is enabled, you could add external API calls here (skipped since online=False)
            if online:
                return Response({"answer": "Online mode not supported in this configuration."}, status=200)

            # Fallback for no match
            return Response({
                "answer": "Sorry, I couldn't find a relevant answer in my database. Try rephrasing your question.",
                "source": "none",
                "confidence": 0.0
            }, status=200)

        except Exception as e:
            # logger.error(f"ChatAPIView Error: {str(e)}")
            # return Response({"error": "Internal server error", "details": str(e)}, status=500)
            logger.error(f"Local DB search error: {str(e)}")
            return Response({
                "answer": "Sorry, I couldn't find a relevant answer in my database. Try another question.",
                "source": "none",
                "confidence": 0.0
            }, status=200)

    def _search_local_database(self, user_question):
        try:
            all_qas = QA.objects.filter(question__isnull=False, answer__isnull=False)
            if not all_qas.exists():
                return {
                    "answer": "Sorry, I couldn't find a relevant answer in my database. Try rephrasing your question.",
                    "source": "none",
                    "confidence": 0.0
                }

            processed_question = preprocess_text(user_question)
            questions = [qa.question.strip() for qa in all_qas]
            answers = [qa.answer.strip() for qa in all_qas]

            # 1. Try near-exact match first
            exact_match = find_near_exact_match(processed_question, questions, answers, threshold=0.8)
            if exact_match:
                return exact_match

            # 2. Semantic match with higher threshold
            threshold = 0.65 if len(processed_question.split()) <= 3 else 0.75
            semantic_match = get_best_match_semantic(processed_question, questions, answers, threshold=threshold)
            if semantic_match:
                # Only fetch suggestions if confidence is below 0.75
                if semantic_match["confidence"] < 0.75:
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
                    semantic_match["suggestions"] = suggestions[:3]
                return semantic_match

            # 3. TF-IDF fallback
            vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
            tfidf_match = get_best_match_tfidf(processed_question, questions, answers, vectorizer, question_vectors,
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
                if tfidf_match["confidence"] < 0.75:
                    tfidf_match["suggestions"] = suggestions[:3]
                return tfidf_match

            # 4. Keyword fallback
            keyword_results = keyword_based_search(processed_question, questions, answers)
            if keyword_results:
                return {
                    "answer": "I couldn't find an exact match, but here are some related topics.",
                    "source": "keyword",
                    "confidence": 0.4,
                    "suggestions": keyword_results[:5]
                }

            return {
                "answer": "No relevant answer found.",
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

    def _get_openai_response(self, user_question):
        """Use the OpenAI v1+ SDK properly."""
        try:
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                return {"error": "OpenAI API key not set."}

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_question}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "source": "openai"
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {"error": "Failed to fetch answer from ChatGPT", "details": str(e)}

    def _search_duckduckgo(self, query):
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("AbstractText"):
                return {"answer": data["AbstractText"], "source": "duckduckgo"}
            elif data.get("Answer"):
                return {"answer": data["Answer"], "source": "duckduckgo"}
            return None
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return None

    def _search_wikipedia(self, query):
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            response = requests.get(url)
            data = response.json()
            if "extract" in data and data["extract"]:
                return {"answer": data["extract"], "source": "wikipedia"}
            return None
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return None
# similarity_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from chat.spellcheck_utils import correct_spelling
from chat.text_preprocessing import preprocess_text  # we'll create this next
from .models import QA
import torch
import logging

logger = logging.getLogger(__name__)

import numpy as np
from sentence_transformers import SentenceTransformer, util
import difflib
# Initialize Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache embeddings
all_qas = list(QA.objects.filter(embedding__isnull=False))
all_embeddings = [torch.tensor(qa.embedding) for qa in all_qas]
all_questions = [qa.question.strip() for qa in all_qas]
all_answers = [qa.answer.strip() for qa in all_qas]

def get_vectorizer_and_vectors(questions):
    processed_questions = [preprocess_text(q) for q in questions]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions)
    return vectorizer, question_vectors

def get_best_match_tfidf(query, questions, answers, vectorizer, question_vectors, threshold=0.6):
    processed_query = preprocess_text(query)
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

def get_best_match_semantic(query, questions, answers, threshold=0.65):
    try:
        processed_query = preprocess_text(query)
        query_embedding = model.encode(processed_query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, torch.stack(all_embeddings))[0]
        best_index = similarities.argmax().item()
        best_score = similarities[best_index].item()

        if best_score > threshold:
            return {
                "answer": answers[best_index],
                "question": questions[best_index],
                "confidence": round(best_score, 2),
                "source": "semantic"
            }
        return None
    except Exception as e:
        logger.error(f"Semantic match error: {str(e)}")
        return None

def suggest_similar_questions(query, questions, answers, vectorizer, question_vectors, top_n=3, threshold=0.2, exclude_index=None):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    top_indices = similarities.argsort()[-top_n-1:][::-1] # Account for excluding best match

    suggestions = []
    for idx in top_indices:
        if similarities[idx] > threshold and idx != exclude_index:
            suggestions.append({
                "question": questions[idx],
                "answer": answers[idx],
                "confidence": round(float(similarities[idx]), 2)
            })
    return suggestions[:top_n]

def keyword_based_search(query, questions, answers):
    query_words = set(preprocess_text(query).split())
    results = []
    for q, a in zip(questions, answers):
        if any(word in preprocess_text(q).lower() for word in query_words):
            results.append({
                "question": q,
                "answer": a,
                "confidence": 0.4,
                "source": "keyword"
            })
    return results

def find_near_exact_match(user_question, questions, answers, threshold=0.8):
    uq = preprocess_text(user_question).lower()
    logger.debug(f"Searching near exact match for: '{uq}'")

    # Substring match
    for idx, q in enumerate(questions):
        q_clean = preprocess_text(q).lower()
        if uq in q_clean or q_clean in uq:
            logger.debug(f"Substring match found: '{questions[idx]}'")
            return {
                "answer": answers[idx],
                "question": questions[idx],
                "confidence": 1.0,
                "source": "exact-substring"
            }

    # Fuzzy match
    question_lowers = [preprocess_text(q).lower() for q in questions]
    matches = difflib.get_close_matches(uq, question_lowers, n=1, cutoff=threshold)
    if matches:
        match = matches[0]
        index = question_lowers.index(match)
        logger.debug(f"Fuzzy match found: '{questions[index]}'")
        return {
            "answer": answers[index],
            "question": questions[index],
            "confidence": 0.9,
            "source": "fuzzy-exact"
        }

    logger.debug("No near exact match found")
    return None
so why it is happend my payload
{
  "question": "what is AGS-TM?",
  "online": false
}
but reponse i got one of the answer but this is not correct because this type of question is not available but AGS-TM realted diffrent questions available so why not suggestions question answer they give me
