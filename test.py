views.py

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import QA
from .serializers import QASerializer
from .utils import preprocess_text
from .similarity_utils import (
    get_vectorizer_and_vectors,
    get_best_match_semantic,
    get_best_match_tfidf,
    suggest_similar_questions,
    keyword_based_search,
    find_near_exact_match,
)
from .spellcheck_utils import correct_spelling
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sentence_transformers import SentenceTransformer
import logging
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
                'online': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Enable external APIs if offline not found')
            },
        ),
        responses={200: "Returns answer from DB or suggestions"}
    )
    def post(self, request):
        try:
            user_question = request.data.get('question', '').strip()
            online = request.data.get('online', False)

            if not user_question:
                return Response(
                    {"answer": "Please provide a question.", "source": "none", "confidence": 0.0},
                    status=status.HTTP_200_OK
                )

            # Correct spelling in the user's question
            corrected_question = correct_spelling(user_question)
            logger.debug(f"Original question: {user_question}, Corrected question: {corrected_question}")

            # Search local database
            answer_data = self._search_local_database(corrected_question, user_question)
            if answer_data and answer_data.get("answer") and answer_data.get("source") not in ["none", "error"]:
                confidence = answer_data.get("confidence", 0)
                CONFIDENCE_THRESHOLD = 0.95  # Only consider exact matches for high confidence

                # Format response with a conversational template
                templates = [
                    "Here's what I found: {}",
                    "Based on my knowledge, {}",
                    "The answer is: {}",
                    "Let me help you with that: {}",
                ]
                response = random.choice(templates).format(answer_data["answer"])

                response_data = {
                    "question": answer_data["question"],
                    "answer": response,
                    "confidence": confidence,
                    "source": answer_data["source"],
                }
                # Include suggestions if confidence is below threshold or explicitly provided
                if confidence < CONFIDENCE_THRESHOLD or answer_data.get("suggestions"):
                    response_data["suggestions"] = answer_data.get("suggestions", [])

                # Update session history
                session_history = request.session.get('chat_history', [])
                session_history.append({"question": user_question, "answer": response})
                request.session['chat_history'] = session_history[-5:]
                return Response(response_data, status=status.HTTP_200_OK)

            # If online is enabled, skip external APIs (as per your requirement)
            if online:
                return Response(
                    {"answer": "Online mode not supported in this configuration.", "source": "none", "confidence": 0.0},
                    status=status.HTTP_200_OK
                )

            # No match found, return suggestions
            all_qas = QA.objects.filter(question__isnull=False, answer__isnull=False)
            if all_qas.exists():
                processed_question = preprocess_text(corrected_question)
                questions = [qa.question.strip() for qa in all_qas]
                answers = [qa.answer.strip() for qa in all_qas]
                vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
                suggestions = suggest_similar_questions(
                    processed_question,
                    questions,
                    answers,
                    vectorizer,
                    question_vectors
                )
                if suggestions:
                    return Response(
                        {
                            "answer": "I couldn't find an exact match for your question. Did you mean one of these?",
                            "source": "suggestions",
                            "confidence": 0.0,
                            "suggestions": suggestions[:3]
                        },
                        status=status.HTTP_200_OK
                    )

            # No suggestions available
            return Response(
                {
                    "answer": "Sorry, I couldn't find any relevant answers or suggestions. Try rephrasing your question.",
                    "source": "none",
                    "confidence": 0.0
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            logger.error(f"ChatAPIView Error: {str(e)}")
            return Response(
                {
                    "answer": "Sorry, an error occurred while processing your request. Here are some related questions you might find helpful.",
                    "source": "error",
                    "confidence": 0.0,
                    "suggestions": self._get_fallback_suggestions(corrected_question)
                },
                status=status.HTTP_200_OK
            )

    def _search_local_database(self, corrected_question, original_question):
        try:
            all_qas = QA.objects.filter(question__isnull=False, answer__isnull=False)
            if not all_qas.exists():
                logger.warning("No Q&A pairs found in the database")
                return {
                    "answer": "No data available in the database.",
                    "source": "none",
                    "confidence": 0.0
                }

            processed_question = preprocess_text(corrected_question)
            questions = [qa.question.strip() for qa in all_qas]
            answers = [qa.answer.strip() for qa in all_qas]

            # Generate embeddings on-the-fly if missing
            model = SentenceTransformer('all-MiniLM-L6-v2')
            for qa in all_qas:
                if not qa.embedding:
                    qa.embedding = model.encode(preprocess_text(qa.question)).tolist()
                    qa.save()

            # 1. Try exact match (case-insensitive, exact text)
            for idx, q in enumerate(questions):
                if processed_question.lower() == preprocess_text(q).lower():
                    return {
                        "answer": answers[idx],
                        "question": q,
                        "confidence": 1.0,
                        "source": "exact"
                    }

            # 2. Try near-exact match with stricter logic
            exact_match = find_near_exact_match(processed_question, questions, answers, threshold=0.9)
            if exact_match:
                # Always include suggestions for non-exact matches
                vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
                best_index = questions.index(exact_match["question"])
                suggestions = suggest_similar_questions(
                    processed_question,
                    questions,
                    answers,
                    vectorizer,
                    question_vectors,
                    exclude_index=best_index
                )
                exact_match["suggestions"] = suggestions[:3]
                return exact_match

            # 3. Semantic match
            threshold = 0.6 if len(processed_question.split()) <= 3 else 0.7
            semantic_match = get_best_match_semantic(processed_question, questions, answers, threshold=threshold)
            if semantic_match:
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

            # 4. TF-IDF fallback
            vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
            tfidf_match = get_best_match_tfidf(processed_question, questions, answers, vectorizer, question_vectors, threshold=0.4)
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
                tfidf_match["suggestions"] = suggestions[:3]
                return tfidf_match

            # 5. Keyword fallback with suggestions
            keyword_results = keyword_based_search(processed_question, questions, answers)
            if keyword_results:
                return {
                    "answer": "I couldn't find an exact match, but here are some related topics.",
                    "source": "keyword",
                    "confidence": 0.4,
                    "suggestions": keyword_results[:3]
                }

            # No match found
            return {
                "answer": "No exact match found.",
                "source": "none",
                "confidence": 0.0
            }

        except Exception as e:
            logger.error(f"Local DB search error: {str(e)}")
            return {
                "answer": "Error during search.",
                "source": "error",
                "confidence": 0.0
            }

    def _get_fallback_suggestions(self, corrected_question):
        try:
            all_qas = QA.objects.filter(question__isnull=False, answer__isnull=False)
            if not all_qas.exists():
                return []
            processed_question = preprocess_text(corrected_question)
            questions = [qa.question.strip() for qa in all_qas]
            answers = [qa.answer.strip() for qa in all_qas]
            vectorizer, question_vectors = get_vectorizer_and_vectors(questions)
            suggestions = suggest_similar_questions(
                processed_question,
                questions,
                answers,
                vectorizer,
                question_vectors
            )
            return suggestions[:3]
        except Exception as e:
            logger.error(f"Fallback suggestion error: {str(e)}")
            return []


similarity_utils.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from chat.spellcheck_utils import correct_spelling
from chat.text_preprocessing import preprocess_text
from .models import QA
import torch
import logging
import difflib

logger = logging.getLogger(__name__)

# Initialize Sentence-Transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
    model = None

def get_vectorizer_and_vectors(questions):
    try:
        if not questions:
            return None, None
        processed_questions = [preprocess_text(q) for q in questions]
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(processed_questions)
        return vectorizer, question_vectors
    except Exception as e:
        logger.error(f"TF-IDF vectorization error: {str(e)}")
        return None, None

def get_best_match_tfidf(query, questions, answers, vectorizer, question_vectors, threshold=0.4):
    try:
        if vectorizer is None or question_vectors is None or not questions:
            return None
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
    except Exception as e:
        logger.error(f"TF-IDF match error: {str(e)}")
        return None

def get_best_match_semantic(query, questions, answers, threshold=0.6):
    try:
        if not model or not questions:
            logger.warning("Semantic model or questions not available")
            return None
        all_qas = list(QA.objects.filter(question__isnull=False, answer__isnull=False))
        all_embeddings = []
        for qa in all_qas:
            if not qa.embedding:
                qa.embedding = model.encode(preprocess_text(qa.question)).tolist()
                qa.save()
            all_embeddings.append(torch.tensor(qa.embedding))
        
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

def suggest_similar_questions(query, questions, answers, vectorizer, question_vectors, top_n=3, threshold=0.1, exclude_index=None):
    try:
        if not questions or vectorizer is None or question_vectors is None:
            return []
        processed_query = preprocess_text(query)
        query_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, question_vectors)[0]
        top_indices = similarities.argsort()[-top_n-1:][::-1]

        suggestions = []
        for idx in top_indices:
            if similarities[idx] > threshold and (exclude_index is None or idx != exclude_index):
                suggestions.append({
                    "question": questions[idx],
                    "answer": answers[idx],
                    "confidence": round(float(similarities[idx]), 2)
                })
        return suggestions[:top_n]
    except Exception as e:
        logger.error(f"Suggestion generation error: {str(e)}")
        return []

def keyword_based_search(query, questions, answers):
    try:
        if not questions:
            return []
        query_words = set(preprocess_text(query).split())
        results = []
        for q, a in zip(questions, answers):
            processed_q = preprocess_text(q).lower()
            if any(word in processed_q for word in query_words if len(word) > 2):
                results.append({
                    "question": q,
                    "answer": a,
                    "confidence": 0.4,
                    "source": "keyword"
                })
        return results
    except Exception as e:
        logger.error(f"Keyword search error: {str(e)}")
        return []

def find_near_exact_match(user_question, questions, answers, threshold=0.9):
    try:
        if not questions:
            return None
        uq = preprocess_text(user_question).lower()
        logger.debug(f"Searching near exact match for: '{uq}'")

        # Substring match with stricter conditions
        for idx, q in enumerate(questions):
            q_clean = preprocess_text(q).lower()
            # Require significant overlap, not just substring
            if (uq in q_clean or q_clean in uq) and len(uq.split()) >= len(q_clean.split()) * 0.8:
                logger.debug(f"Substring match found: '{questions[idx]}'")
                return {
                    "answer": answers[idx],
                    "question": questions[idx],
                    "confidence": 0.9,  # Lower confidence for substring matches
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
                "confidence": 0.85,  # Lower confidence for fuzzy matches
                "source": "fuzzy-exact"
            }

        logger.debug("No near exact match found")
        return None
    except Exception as e:
        logger.error(f"Near exact match error: {str(e)}")
        return None



spellcheck_utils.py
from spellchecker import SpellChecker
import re
import logging

logger = logging.getLogger(__name__)

def correct_spelling(text):
    try:
        spell = SpellChecker()
        # Preserve specific terms
        custom_words = {'ags-tm', 'recipe', 'ai'}  # Add more as needed
        spell.word_frequency.load_words(custom_words)

        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        corrected_words = []
        for word in words:
            if word.lower() in custom_words or not word.isalpha():
                corrected_words.append(word)
            else:
                corrected = spell.correction(word)
                corrected_words.append(corrected if corrected else word)
        return ' '.join(corrected_words)
    except Exception as e:
        logger.error(f"Spelling correction error: {str(e)}")
        return text

text_preprocessing.py

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)

def preprocess_text(text):
    try:
        text = text.lower()
        # Preserve hyphenated terms like AGS-TM
        text = re.sub(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', r' \1 ', text)
        text = re.sub(r'[^\w\s-]', '', text)  # Remove punctuation except hyphens
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Text preprocessing error: {str(e)}")
        return text


