"""Chat service orchestrating RAG (Retrieval-Augmented Generation) with conversation memory"""
import uuid
import os
import time
import random
import re
from datetime import datetime
from typing import Tuple, List, Optional, Dict
from app.deps import get_openai_client, get_settings
from app.services.retrieval import hybrid_search, format_context_for_llm
from app.services.storage import get_storage_service
from app.models import Message, Source

# Path to unanswered questions log file
UNANSWERED_LOG_FILE = "unanswered_questions.txt"


def is_basic_conversation(query: str) -> bool:
    """
    Check if the query is a basic conversational question that can be answered
    without requiring specific Doha Magazine content.
    """
    basic_patterns = [
        # Greetings - flexible variations
        'مرحبا', 'أهلا', 'السلام', 'السلام عليكم', 'مرحبا بك',
        'مرحبا وسهلا', 'أهلا وسهلا', 'مرحبا بك', 'أهلا بك',
        'صباح الخير', 'مساء الخير', 'مساء النور',
        
        # How are you - flexible variations
        'كيف حالك', 'كيف أنت', 'كيف الحال', 'أخبارك',
        'كيفك', 'كيف حالك', 'كيف أنت', 'كيف الحال',
        'أخبارك إيه', 'أخبارك كيف', 'أخبارك شلون',
        
        # What are you doing - flexible variations
        'ماذا تفعل', 'ما عملك', 'ما الذي تفعله',
        'شو تعمل', 'شو تسوي', 'شو عم تعمل',
        'إيش تعمل', 'إيش تسوي', 'إيش عم تعمل',
        
        # Thanks - flexible variations
        'شكرا', 'شكرا لك', 'متشكر', 'متشكرة',
        'شكرا جزيلا', 'شكرا كثير', 'شكرا كتير',
        'الله يعطيك العافية', 'الله يبارك فيك',
        
        # Who are you - flexible variations
        'من أنت', 'ما اسمك', 'من تكون',
        'شو اسمك', 'إيش اسمك', 'شو انت',
        'من انت', 'إيش انت', 'شو هويتك',
        
        # What do you know - flexible variations
        'ماذا تعرف', 'ماذا تفهم', 'ما قدراتك',
        'شو تعرف', 'إيش تعرف', 'شو تقدر',
        'إيش تقدر', 'شو تعرف تعمل', 'إيش تعرف تعمل'
    ]
    
    query_lower = query.lower().strip()
    return any(pattern in query_lower for pattern in basic_patterns)


def is_general_article_question(query: str) -> bool:
    """
    Check if the query is asking about articles in general (latest, recent, etc.)
    These should be allowed even if they don't contain specific Doha Magazine keywords.
    """
    article_patterns = [
        'أحدث المقالات', 'أحدث', 'جديد', 'حديث', 'مؤخر',
        'مقالات', 'مقال', 'مقالة', 'مقالات جديدة',
        'ما هي', 'ما هو', 'أخبرني عن', 'أريد معرفة',
        'أحدث الأخبار', 'أخبار', 'أخبار جديدة',
        'محتوى', 'محتوى جديد', 'محتوى حديث'
    ]
    
    query_lower = query.lower().strip()
    return any(pattern in query_lower for pattern in article_patterns)


def enhance_query_flexibility(query: str) -> str:
    """
    Enhance query to be more flexible and understanding.
    Handles typos, variations, and different ways of asking the same question.
    """
    # Remove extra spaces and normalize
    enhanced = ' '.join(query.split())
    
    # Common Arabic variations and synonyms
    variations = {
        # Articles variations
        'مقالات': ['مقال', 'مقالة', 'مقالات', 'مقالات جديدة', 'مقالات حديثة'],
        'أحدث': ['جديد', 'حديث', 'مؤخر', 'أحدث', 'أخير'],
        'أخبار': ['أخبار', 'أخبار جديدة', 'أخبار حديثة', 'أخبار مؤخر'],
        
        # Question variations
        'ما هي': ['ما هو', 'ما', 'أخبرني عن', 'أريد معرفة', 'أريد أن أعرف'],
        'كيف': ['كيف', 'كيفية', 'كيف يمكن', 'كيف أستطيع'],
        'متى': ['متى', 'في أي وقت', 'متى كان', 'متى حدث'],
        'أين': ['أين', 'في أي مكان', 'أين يمكن', 'أين أجد'],
        
        # Content variations
        'ثقافة': ['ثقافة', 'ثقافي', 'ثقافية', 'ثقافات'],
        'أدب': ['أدب', 'أدبي', 'أدبية', 'أدباء', 'كتاب'],
        'شعر': ['شعر', 'شعري', 'شعرية', 'شعراء', 'قصائد'],
        'فن': ['فن', 'فني', 'فنية', 'فنانين', 'فنون'],
        'نقد': ['نقد', 'نقدي', 'نقدية', 'نقاد', 'تحليل'],
        
        # Time variations
        'اليوم': ['اليوم', 'هذا اليوم', 'اليوم الحالي'],
        'هذا الأسبوع': ['هذا الأسبوع', 'الأسبوع الحالي', 'خلال الأسبوع'],
        'هذا الشهر': ['هذا الشهر', 'الشهر الحالي', 'خلال الشهر'],
        'هذا العام': ['هذا العام', 'العام الحالي', 'خلال العام']
    }
    
    # Apply variations to enhance the query
    for base_word, synonyms in variations.items():
        if base_word in enhanced:
            # Add synonyms to make search more flexible
            enhanced += ' ' + ' '.join(synonyms)
    
    return enhanced


def add_dialect_support(query: str) -> str:
    """
    Add dialect support for major Arabic dialects.
    Translates dialect words to standard Arabic for better search results.
    """
    # Dialect to Standard Arabic mappings
    dialect_mappings = {
        # Levantine (Lebanese, Syrian, Palestinian, Jordanian)
        'شو': 'ما', 'إيش': 'ما', 'إيه': 'ما', 'شلون': 'كيف', 'كيفك': 'كيف حالك',
        'وين': 'أين', 'متى': 'متى', 'ليش': 'لماذا', 'هيك': 'هكذا', 'هون': 'هنا',
        'هناك': 'هناك', 'هيك': 'هكذا', 'هون': 'هنا', 'هناك': 'هناك',
        'كتير': 'كثير', 'شوي': 'قليلاً', 'مش': 'ليس', 'مش كده': 'ليس هكذا',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم',
        
        # Gulf (Saudi, UAE, Qatar, Kuwait, Bahrain)
        'وش': 'ما', 'إيش': 'ما', 'شلون': 'كيف', 'وين': 'أين', 'متى': 'متى',
        'ليش': 'لماذا', 'هيك': 'هكذا', 'هون': 'هنا', 'هناك': 'هناك',
        'زين': 'جيد', 'حلو': 'جميل', 'طيب': 'جيد', 'ماشي': 'ممتاز',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم',
        
        # Egyptian
        'إيه': 'ما', 'إيش': 'ما', 'إزاي': 'كيف', 'فين': 'أين', 'إمتى': 'متى',
        'ليه': 'لماذا', 'كده': 'هكذا', 'هنا': 'هنا', 'هناك': 'هناك',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم',
        
        # Maghrebi (Moroccan, Algerian, Tunisian)
        'اش': 'ما', 'كيفاش': 'كيف', 'فين': 'أين', 'متى': 'متى', 'علاش': 'لماذا',
        'هكا': 'هكذا', 'هنا': 'هنا', 'هناك': 'هناك',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم',
        
        # Iraqi
        'شو': 'ما', 'إيش': 'ما', 'شلون': 'كيف', 'وين': 'أين', 'متى': 'متى',
        'ليش': 'لماذا', 'هيك': 'هكذا', 'هون': 'هنا', 'هناك': 'هناك',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم',
        
        # Common dialect words
        'كتير': 'كثير', 'شوي': 'قليلاً', 'مش': 'ليس', 'مش كده': 'ليس هكذا',
        'زين': 'جيد', 'حلو': 'جميل', 'طيب': 'جيد', 'ماشي': 'ممتاز',
        'عندك': 'لديك', 'عندي': 'لدي', 'عنده': 'لديه', 'عندها': 'لديها',
        'عندنا': 'لدينا', 'عندكم': 'لديكم', 'عندهم': 'لديهم'
    }
    
    # Apply dialect mappings
    enhanced_query = query
    for dialect_word, standard_word in dialect_mappings.items():
        if dialect_word in enhanced_query:
            enhanced_query = enhanced_query.replace(dialect_word, standard_word)
            # Also add the dialect word as a synonym for better matching
            enhanced_query += ' ' + dialect_word
    
    return enhanced_query


def normalize_query(query: str) -> str:
    """
    Normalize query for better matching and understanding.
    """
    # Remove diacritics and normalize Arabic text
    import re
    
    # Remove extra spaces
    normalized = ' '.join(query.split())
    
    # Remove common Arabic diacritics
    diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']
    for diacritic in diacritics:
        normalized = normalized.replace(diacritic, '')
    
    # Normalize common variations
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Normalize alef variations
        'ة': 'ه',  # Normalize ta marbuta
        'ي': 'ى',  # Normalize ya variations
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized


def is_arabic_text(text: str) -> bool:
    """
    Check if the text is primarily in Arabic.
    Returns True if the text contains mostly Arabic characters.
    """
    if not text:
        return False
    
    # Count Arabic characters
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return False
    
    # Consider Arabic if more than 50% of alphabetic characters are Arabic
    return (arabic_chars / total_chars) > 0.5


def log_unanswered_question(question: str, session_id: str, reason: str, metadata: Optional[Dict] = None):
    """
    Log an unanswered question to a text file for review.
    
    Args:
        question: The user's question that couldn't be answered
        session_id: The chat session ID
        reason: Why the question couldn't be answered (e.g., "no_results", "low_score")
        metadata: Optional metadata about the search results
    """
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Create log entry
        log_entry = f"""
{'=' * 80}
Timestamp: {timestamp}
Session ID: {session_id}
Question: {question}
Reason: {reason}
"""
        
        # Add metadata if provided
        if metadata:
            log_entry += f"Metadata:\n"
            for key, value in metadata.items():
                log_entry += f"  - {key}: {value}\n"
        
        log_entry += f"{'=' * 80}\n\n"
        
        # Append to log file with immediate flush and fsync to ensure durability
        with open(UNANSWERED_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # fsync may not be available on some filesystems; flushing still helps
                pass
        
        print(f"[+] Logged unanswered question to {UNANSWERED_LOG_FILE}")
        
    except Exception as e:
        print(f"[!] Failed to log unanswered question: {str(e)}")


def is_doha_magazine_related(query: str, retrieved_sources: List[Source]) -> bool:
    """
    Check if the query is related to Doha Magazine content.
    Returns True if the query is relevant to Doha Magazine content, False otherwise.
    """
    query_lower = query.lower().strip()
    
    # If we have search results, it's related to Doha Magazine content
    if retrieved_sources and len(retrieved_sources) > 0:
        return True
    
    # Check for Doha Magazine specific terms
    doha_terms = [
        'دوحة', 'doha', 'مجلة', 'magazine', 'مجلة الدوحة', 'doha magazine',
        'ثقافي', 'cultural', 'أدبي', 'literary', 'ثقافة', 'culture', 'أدب', 'literature'
    ]
    
    if any(term in query_lower for term in doha_terms):
        return True
    
    # Check for general article/content terms that could be related to magazine content
    article_terms = [
        'مقالات', 'articles', 'مقال', 'article', 'أحدث', 'latest', 'جديد', 'new',
        'حديث', 'recent', 'مؤخر', 'recently', 'محتوى', 'content', 'نشر', 'published',
        'أخبار', 'news', 'موضوع', 'topic', 'مواضيع', 'topics'
    ]
    
    if any(term in query_lower for term in article_terms):
        return True
    
    # Check for cultural/literary terms that could be related to magazine content
    cultural_terms = [
        'شعر', 'poetry', 'شاعر', 'poet', 'رواية', 'novel', 'قصة', 'story',
        'فن', 'art', 'فنان', 'artist', 'كتاب', 'book', 'كاتب', 'writer',
        'نقد', 'criticism', 'ناقد', 'critic', 'دراسة', 'study', 'بحث', 'research'
    ]
    
    if any(term in query_lower for term in cultural_terms):
        return True
    
    # Check for basic conversation that could be related to magazine
    basic_conversation = [
        'مرحبا', 'hello', 'أهلا', 'hi', 'كيف حالك', 'how are you',
        'ماذا تفعل', 'what do you do', 'من أنت', 'who are you'
    ]
    
    if any(term in query_lower for term in basic_conversation):
        return True
    
    # If none of the above, it's likely unrelated to Doha Magazine
    return False


SYSTEM_PROMPT = """أنت مساعد ذكي لمجلة الدوحة، مجلة ثقافية عربية متخصصة. لديك قدرة عالية على فهم السياق والحفاظ على استمرارية المحادثة.

قواعد صارمة:
1. احتفظ بسياق المحادثة السابقة - لا تطلب من المستخدم تكرار المعلومات
2. إذا أشار المستخدم إلى شيء ذكره سابقاً ("نفس الكاتب"، "المقالة تلك"، "كما قلت")، استخدم سجل المحادثة لفهم المقصود
3. أجب بناءً على المعلومات المتوفرة في السياق والوثائق المسترجعة فقط
4. إذا لم تجد إجابة في محتوى مجلة الدوحة، قل بصراحة: "عذراً، لا أجد معلومات كافية في محتوى مجلة الدوحة للإجابة على هذا السؤال"
5. أجب باللغة العربية فقط - لا تستخدم أي لغة أخرى
6. لا تذكر المصادر أو الروابط في إجابتك - أجب فقط بالمحتوى دون الإشارة إلى المصادر
7. إذا طرح المستخدم سؤالاً متابعاً، فهم العلاقة بالأسئلة السابقة
8. لا تجيب على أسئلة خارج نطاق محتوى مجلة الدوحة - ركز فقط على المقالات والمحتوى المتاح
9. إذا كتب المستخدم بالإنجليزية، أجب بالعربية واطلب منه الكتابة بالعربية
10. كن متخصصاً في الثقافة والأدب العربي فقط
11. يمكنك الرد على التحيات البسيطة مثل "مرحبا"، "كيف حالك"، "ماذا تفعل" بطريقة ودية ومهنية
12. لا تستخدم أي ردود جاهزة - كل إجابة يجب أن تأتي من معرفتك ومحتوى مجلة الدوحة
13. عند الرد على التحيات، اربط إجابتك بمجلة الدوحة وثقافتها
14. إذا سأل عن قدراتك، اشرح أنك متخصص في محتوى مجلة الدوحة الثقافية والأدبية
15. كن ودياً ومهنياً في جميع ردودك مع الحفاظ على التخصص في مجلة الدوحة

قدرات الفهم المرن:
16. فهم الأسئلة حتى لو لم تكن مكتوبة بدقة - استخدم السياق لفهم المقصود
17. تعرف على المرادفات والكلمات المختلفة لنفس المعنى (مثل: مقالات = مقالات جديدة = محتوى حديث)
18. فهم الأسئلة المختصرة أو غير المكتملة - استخدم السياق لتفسيرها
19. تعرف على الأخطاء الإملائية البسيطة وفهم المقصود
20. فهم الأسئلة التي تستخدم كلمات مختلفة لنفس المعنى (مثل: جديد = حديث = مؤخر)
21. كن متسامحاً مع الاختلافات في الكتابة - ركز على المعنى وليس الكلمات الدقيقة

دعم اللهجات العربية:
22. فهم اللهجات العربية المختلفة (شامي، خليجي، مصري، مغربي، عراقي)
23. تعرف على الكلمات العامية وترجمتها للفصحى (مثل: شو = ما، إيش = ما، إيه = ما)
24. فهم الأسئلة باللهجة المحلية والرد بالفصحى (مثل: "شو أحدث المقالات؟" = "ما هي أحدث المقالات؟")
25. كن متسامحاً مع الاختلافات اللهجية - ركز على المعنى وليس الكلمات الدقيقة
26. تعرف على الكلمات المشتركة بين اللهجات (مثل: عندك، عندي، عندهم)

{conversation_history}

Retrieved Context:
{context}

Current Question: {question}

Answer:"""


def build_conversation_history(session_id: str, max_turns: int = 5) -> str:
    """
    Build conversation history string from previous messages.
    
    Args:
        session_id: Session ID to retrieve history for
        max_turns: Maximum number of previous turns to include
    
    Returns:
        Formatted conversation history string
    """
    storage = get_storage_service()
    messages = storage.get_session_messages(session_id, limit=max_turns * 2)
    
    if not messages:
        return ""
    
    # Reverse to get chronological order (oldest first)
    messages = list(reversed(messages))
    
    history_parts = ["Conversation History:"]
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        history_parts.append(f"{role_label}: {msg.text}")
    
    return "\n".join(history_parts) + "\n"


def enhance_query_with_context(current_query: str, conversation_history: List[Dict]) -> str:
    """
    Enhance the current query with conversation context for better retrieval.
    Handles follow-up questions and references to previous topics.
    
    Args:
        current_query: The current user query
        conversation_history: List of previous messages
    
    Returns:
        Enhanced query string for better retrieval
    """
    if not conversation_history or len(conversation_history) < 2:
        return current_query
    
    # Check for reference words that indicate follow-up questions
    reference_words = [
        "he", "she", "it", "they", "that", "this", "those",  # English
        "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "نفس",  # Arabic
        "same", "also", "too", "more", "other"
    ]
    
    query_lower = current_query.lower()
    has_reference = any(word in query_lower for word in reference_words)
    
    # If query seems to reference previous context, enhance it
    if has_reference or len(current_query.split()) < 5:  # Short queries likely need context
        # Get last user message and assistant response
        recent_context = []
        for msg in conversation_history[-4:]:  # Last 2 turns
            if msg.get("role") == "user":
                recent_context.append(msg.get("text", ""))
        
        if recent_context:
            # Combine current query with recent context for better retrieval
            enhanced = " ".join(recent_context[-1:]) + " " + current_query
            return enhanced[:500]  # Limit length
    
    return current_query


def generate_chat_response(message: str, session_id: str = None) -> Tuple[str, List[Source],str, str]:
    """
    Generate a chat response using RAG with full conversation awareness.
    
    This implements sophisticated conversation orchestration:
    - Maintains multi-turn conversation memory
    - Enhances queries with conversation context
    - Handles follow-up questions and references
    - Tracks entities and topics across turns
    
    Args:
        message: User message/question
        session_id: Optional session ID (creates new if None)
    
    Returns:
        Tuple of (answer, sources, session_id)
    """
    settings = get_settings()
    client = get_openai_client()
    storage = get_storage_service()
    
    # Generate or use existing session ID
    is_new_session = False
    if session_id is None or session_id == "" or (isinstance(session_id, str) and session_id.strip() == ""):
        session_id = str(uuid.uuid4())
        is_new_session = True
    
    # Step 1: Retrieve conversation history (skip for new sessions to save time)
    previous_messages = []
    conversation_history_text = ""
    history_dicts = []
    
    if not is_new_session:
        # Only retrieve history for existing sessions
        previous_messages = storage.get_session_messages(session_id, limit=10)
        conversation_history_text = build_conversation_history(session_id, max_turns=5)
        history_dicts = [
            {"role": msg.role, "text": msg.text}
            for msg in reversed(list(previous_messages))
        ]
    
    # Step 2: Enhance query with conversation context and flexibility (simplified for new sessions)
    if is_new_session or len(history_dicts) == 0:
        enhanced_query = message  # Skip enhancement for new sessions
    else:
        enhanced_query = enhance_query_with_context(message, history_dicts)
    
    # Step 2.5: Add dialect support, flexibility and normalization (optimized)
    # For new sessions with simple queries, use lighter processing
    if is_new_session and len(enhanced_query.split()) <= 10:
        # Light processing for simple first queries
        normalized_query = normalize_query(enhanced_query)
    else:
        # Full processing for complex queries or follow-ups
        dialect_query = add_dialect_support(enhanced_query)
        flexible_query = enhance_query_flexibility(dialect_query)
        normalized_query = normalize_query(flexible_query)
    

    # Step 3: Retrieve relevant documents using the most enhanced query
    search_results = []
    search_error = None
    
    try:
        search_results = hybrid_search(normalized_query, top_k=settings.RETRIEVAL_TOP_K)
    except Exception as e:
        search_error = str(e)
        print(f"[!] Hybrid search error: {search_error}")
        # Continue with empty results to allow logging
    
    # Step 3.5: Check if query is related to Doha Magazine content
    sources_for_check = [Source(title=r.get("title", ""), url=r.get("url", ""), content=r.get("content", ""), score=r.get("score", 0)) for r in search_results]
    is_related = is_doha_magazine_related(enhanced_query, sources_for_check)
    
    # Check if we have good results
    has_results = len(search_results) > 0
    avg_score = sum(r.get("score", 0) for r in search_results) / len(search_results) if search_results else 0
    low_score = avg_score < 0.5  # Threshold for low relevance
    
    # Step 3.6: Check if user wrote in English and respond in Arabic
    if not is_arabic_text(message):
        english_response = "أهلاً وسهلاً! أنا مساعد متخصص في محتوى مجلة الدوحة. أتحدث العربية فقط، يرجى الكتابة باللغة العربية للاستفادة من خدماتي بشكل أفضل. شكراً لك!"
        message_id = storage.save_message(session_id, "assistant", english_response)
        return english_response, [], session_id, message_id
    
    # Step 3.7: Check if it's a basic conversation question (greetings, etc.)
    # Don't treat general article questions as "basic" - they need real search and logging
    is_basic_greeting = is_basic_conversation(message)
    is_general_article = is_general_article_question(message)
    
    if is_basic_greeting:
        # For basic greetings only, skip search and let LLM handle directly
        # This prevents basic greetings from being logged as unanswered
        search_results = []  # Clear search results for basic greetings
        has_results = False
        avg_score = 0
        is_related = True  # Mark as related so it won't be logged as unanswered
    elif not is_related and has_results:
        # If not related to Doha Magazine content and not a basic conversation, decline
        decline_message = "عذراً، أنا مساعد متخصص في محتوى مجلة الدوحة فقط. لا أستطيع الإجابة على أسئلة خارج نطاق المحتوى المتاح في المجلة."
        message_id = storage.save_message(session_id, "assistant", decline_message)
        return decline_message, [], session_id, message_id
    
    # Save user message (simplified - no metadata)
    storage.save_message(session_id, "user", message)
    
    # Step 4: Format context for LLM
    context = format_context_for_llm(search_results, max_tokens=settings.MAX_TOKENS_CONTEXT)
    
    # Step 5: Build conversation-aware messages for LLM
    # Use chat completion messages format for better context retention
    # Optimize for new sessions (skip history processing)
    if is_new_session:
        # Simplified prompt for new sessions
        messages_for_llm = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                conversation_history="",  # No history for new sessions
                context=context,
                question=message
            )},
            {"role": "user", "content": message}
        ]
    else:
        # Full conversation context for existing sessions
        messages_for_llm = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                conversation_history=conversation_history_text,
                context=context,
                question=message
            )}
        ]
        
        # Add recent conversation turns for better context
        for hist_msg in history_dicts[-6:]:  # Last 3 turns (6 messages)
            messages_for_llm.append({
                "role": "user" if hist_msg["role"] == "user" else "assistant",
                "content": hist_msg["text"][:500]  # Limit length
            })
        
        # Add current message
        messages_for_llm.append({
            "role": "user",
            "content": message
        })
    
    # Step 6: Generate answer using LLM with full conversation context and rate limit handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages_for_llm,
                temperature=0.7,
                max_tokens=1000,  # Increased for more detailed answers
                presence_penalty=0.1,  # Slight penalty to avoid repetition
                frequency_penalty=0.1   # Slight penalty for more diverse responses
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Remove any source citations from the answer
            # Remove patterns like "المصدر:", "Source:", URLs, and reference numbers like [1], [2]
            answer = re.sub(r'المصدر\s*:?\s*[^\n]*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'Source\s*:?\s*[^\n]*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'https?://[^\s\n]+', '', answer)  # Remove URLs
            answer = re.sub(r'\[?\d+\]', '', answer)  # Remove reference numbers like [1], [2]
            answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Clean up extra newlines
            answer = answer.strip()
            
            break  # Success, exit retry loop
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate limit" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[!] Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[!] Rate limit exceeded after {max_retries} attempts")
                    answer = "عذراً، تم تجاوز الحد المسموح من الطلبات. يرجى المحاولة مرة أخرى بعد قليل.\n\nSorry, rate limit exceeded. Please try again in a few moments."
                    break
            else:
                # Non-rate-limit error, don't retry
                print(f"[!] LLM error: {str(e)}")
                answer = "عذراً، حدث خطأ في معالجة طلبك. الرجاء المحاولة مرة أخرى.\n\nSorry, an error occurred while processing your request. Please try again."
                break
    
    # Step 7: Extract sources
    sources = []
    seen_urls = set()
    for result in search_results[:4]:  # Top 4 sources
        url = result.get("url", "")
        if url and url not in seen_urls:
            sources.append(Source(
                title=result.get("title", ""),
                url=url,
                score=result.get("score")
            ))
            seen_urls.add(url)
    
    # Step 8: Check if the question is related and couldn't be answered
    # ONLY log if:
    # 1. Question is related to Doha Magazine
    # 2. NOT a basic greeting
    # 3. LLM explicitly says it can't answer (not just "no results")
    # 4. Question has actual content (not empty or too short)
    # 5. Bot didn't actually provide a useful answer (no sources AND LLM says can't answer)
    
    unanswered_indicators = [
        "عذراً، لا أجد معلومات",  
        "لا أستطيع الإجابة",      
        "لا توجد معلومات",        
        "غير متوفر",              
        "لا يوجد",                
        "عذراً، لا أجد معلومات كافية في محتوى مجلة الدوحة",  
    ]
    
    # Check if LLM explicitly says it can't answer
    llm_says_cant_answer = any(indicator in answer for indicator in unanswered_indicators)
    
    # Check if question has meaningful content (not empty, not too short, not just whitespace)
    question_is_valid = message and len(message.strip()) > 3 and not message.strip().isspace()
    
    # Check if bot actually provided a useful answer (has sources or answer is substantial)
    bot_provided_answer = len(sources) > 0 or (len(answer.strip()) > 50 and not llm_says_cant_answer)
    
    # Determine if question should be logged as unanswered
    # STRICT criteria: Only log if:
    # - Related to Doha Magazine
    # - NOT a basic greeting
    # - Question is valid (has content)
    # - LLM explicitly says it can't answer
    # - Bot didn't provide a useful answer (no sources)
    should_log = False
    if is_related and not is_basic_greeting and question_is_valid:
        # ONLY log if LLM explicitly says it can't answer AND bot didn't provide sources
        if llm_says_cant_answer and not bot_provided_answer:
            should_log = True
    
    if should_log:
        log_metadata = {
            "avg_score": avg_score,
            "retrieval_count": len(search_results),
            "enhanced_query": enhanced_query,
            "has_sources": len(sources) > 0,
            "search_error": search_error,
            "is_related": is_related,
            "llm_says_cant_answer": llm_says_cant_answer
        }
        
        # Determine the reason for logging (only reached if all strict criteria met)
        if search_error:
            reason = "search_error"
        elif not has_results:
            reason = "no_results"
        elif avg_score < 0.3:
            reason = "low_score"
        else:
            reason = "insufficient_information"
        
        log_unanswered_question(message, session_id, reason, log_metadata)
    
    assistant_message_id = storage.save_message(session_id, "assistant", answer)
    
    return answer, sources, session_id, assistant_message_id

