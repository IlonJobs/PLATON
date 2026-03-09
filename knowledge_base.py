import os
from typing import List, Dict, Optional

# Импорты для Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http import models
from dotenv import load_dotenv
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langsmith import traceable
import pymupdf4llm
import pathlib
import pymupdf
from langchain.schema import Document
from langgraph.graph import StateGraph
import uuid
import json
import pandas as pd


load_dotenv()

# Настройки
COLLECTION_NAME = "telegram_kb"
SECTIONS_COLLECTION = "sections"
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# GigaChat Embeddings принимает не более 514 токенов на один запрос.
# Для русского текста ~1 токен ≈ 1.5-2 символа, поэтому безопасный
# chunk_size ≈ 500 символов (запас ~35% от лимита токенов).
# Используется во всех сплиттерах проекта — меняем в одном месте.
GIGACHAT_EMBED_MAX_TOKENS = 514
CHUNK_SIZE = 500   # символов — безопасно укладывается в лимит токенов
CHUNK_OVERLAP = 80 # ~16% от chunk_size — стандартное перекрытие

class KnowledgeBase:
    def __init__(self):
        # 1. Настройка Embeddings
         
        # от Google
        #self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        #от СБЕРа  
        embedding=GigaChatEmbeddings(
            credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
        )
        self.embeddings = embedding
        
        # 2. Клиент Qdrant
        self.qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_HOST"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )


        vector_size = 1024  # размерность для GigaChatEmbeddings
        #vector_size=768,  # Размерность text-embedding-004
        # Создаем коллекцию, если нет
        if not self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Создана коллекция {COLLECTION_NAME}")

        # Создаем коллекцию, если нет
        if not self.qdrant_client.collection_exists(SECTIONS_COLLECTION):
            self.qdrant_client.create_collection(
                collection_name=SECTIONS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Создана коллекция {SECTIONS_COLLECTION}")

        # Гарантируем наличие всех индексов при каждом запуске.
        #
        # ИСПРАВЛЕНИЕ — три ошибки в прежнем коде:
        # 1. Индексы создавались только внутри блока "if not collection_exists",
        #    т.е. для уже существующей базы они никогда не создавались.
        # 2. Поле было названо "section_name" (верхний уровень), но LangChain
        #    сохраняет все метаданные внутри объекта metadata, поэтому реальный
        #    путь в Qdrant — "metadata.section_name". Qdrant требует индекс
        #    точно по тому пути, который используется в фильтре get_relevants.
        # 3. Аналогично "metadata.section_id" вместо "section_id".
        #
        # create_payload_index идемпотентен: если индекс уже существует —
        # вызов завершится без ошибки, поэтому безопасно вызывать при каждом
        # старте приложения.
        self._ensure_indexes()



        # Интеграция LangChain и Qdrant
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings
        )

        # 3. LLM 
        # Gemini
        
        # self.llm = ChatGoogleGenerativeAI(
        #     model=LLM_MODEL, 
        #     temperature=0.3,
        #     convert_system_message_to_human=True # Иногда нужно для старых версий langchain
        # )
        
        #СБЕР
        
        self.llm = GigaChat(
            credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            verify_ssl_certs=False,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def _ensure_indexes(self):
        """Создаёт все необходимые payload-индексы, если они ещё не существуют.

        Вызывается при каждом старте — это безопасно, потому что
        create_payload_index идемпотентен (повторный вызов не вызывает ошибку).

        Индексы для коллекции telegram_kb:
        ┌─────────────────────────────┬─────────┬──────────────────────────────────┐
        │ Поле                        │ Тип     │ Используется в                   │
        ├─────────────────────────────┼─────────┼──────────────────────────────────┤
        │ metadata.user_id            │ INTEGER │ все запросы — базовый фильтр     │
        │ metadata.section_name       │ KEYWORD │ get_relevants (фильтр по разделу)│
        │ metadata.section_id         │ KEYWORD │ поиск чанков раздела             │
        └─────────────────────────────┴─────────┴──────────────────────────────────┘

        Важно: LangChain при сохранении Document складывает metadata в поле
        payload["metadata"], поэтому путь всегда "metadata.<поле>", а НЕ
        просто "<поле>" на верхнем уровне.

        Индексы для коллекции sections:
        ┌─────────┬─────────┬──────────────────────────────────────────┐
        │ Поле    │ Тип     │ Используется в                           │
        ├─────────┼─────────┼──────────────────────────────────────────┤
        │ user_id │ INTEGER │ _get_user_sections, find_section_for_query│
        └─────────┴─────────┴──────────────────────────────────────────┘

        В коллекции sections поля хранятся на верхнем уровне payload (не через
        LangChain), поэтому путь просто "user_id".
        """
        kb_indexes = [
            ("metadata.user_id",      models.PayloadSchemaType.INTEGER),
            ("metadata.section_name", models.PayloadSchemaType.KEYWORD),  # исправлен путь
            ("metadata.section_id",   models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_schema in kb_indexes:
            self.qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"  ✅ Индекс [{COLLECTION_NAME}] {field_name}")

        sections_indexes = [
            ("user_id", models.PayloadSchemaType.INTEGER),
        ]
        for field_name, field_schema in sections_indexes:
            self.qdrant_client.create_payload_index(
                collection_name=SECTIONS_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"  ✅ Индекс [{SECTIONS_COLLECTION}] {field_name}")

    def add_text(self, text: str, user_id: int, source: str = "message"):
        """Добавляет текст в базу (синхронно).
        
        ИЗМЕНЕНИЕ: Теперь так же, как add_document, определяет раздел через LLM
        и создаёт/переиспользует раздел в коллекции sections.
        Это гарантирует, что у КАЖДОГО чанка в базе есть section_name и section_id,
        независимо от того, как он был добавлен — файлом или командой «Запомни:».
        """
        # --- НОВЫЙ БЛОК: классификация раздела ---
        existing_sections = self._get_user_sections(user_id)
        classification = self._classify_document(text, existing_sections)
        section_name = classification["section_name"]
        section_description = classification["description"]
        keywords = classification.get("keywords", [])

        # Определяем section_id: создаём новый раздел или берём существующий
        section_id = None
        if classification.get("is_new", True):
            similar = self._find_similar_section(section_description, user_id)
            if similar:
                section_id = similar["id"]
                section_name = similar["name"]
                # Раздел существует — пополняем ключевые слова
                self._update_section_keywords(section_id, keywords)
            else:
                section_id = self._create_section(section_name, section_description, keywords, user_id)
        else:
            for section in existing_sections:
                if section["name"] == section_name:
                    section_id = section["id"]
                    break
            if section_id:
                # Раздел существует — пополняем ключевые слова
                self._update_section_keywords(section_id, keywords)
            else:
                section_id = self._create_section(section_name, section_description, keywords, user_id)
        # --- КОНЕЦ БЛОКА ---

        docs = [Document(
            page_content=text,
            metadata={
                "user_id": user_id,
                "source": source,
                # НОВОЕ: метаданные раздела — такие же поля, как в add_document
                "section_id": section_id,
                "section_name": section_name,
                "section_description": section_description,
                "keywords": keywords,
            }
        )]
        splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)
        
    def clear_user_db(self, user_id: int, source: str = "message"):
        """Удалить все точки с конкретным user_id"""
        self.qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        )

    def clean_db(self):
        """Очистить все точки коллекции и пересоздать с полными индексами."""
        collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)

        self.qdrant_client.delete_collection(COLLECTION_NAME)

        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=collection_info.config.params.vectors
        )

        # ИСПРАВЛЕНИЕ: раньше здесь создавался только индекс user_id.
        # Теперь _ensure_indexes создаёт все необходимые индексы для обеих коллекций.
        self._ensure_indexes()

    def add_document0(self, file_path: str, user_id: int):
        """Обрабатывает файл"""
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        else:
            return "Формат не поддерживается"

        for doc in docs:
            doc.metadata["user_id"] = user_id
            doc.metadata["source"] = os.path.basename(file_path)

        splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)
        return f"Добавлено {len(splits)} фрагментов."
    

    def add_document1(self, file_path: str, user_id: int):
        """Обрабатывает файл"""
        
        if file_path.endswith(".pdf"):
            # Конвертируем PDF в Markdown
            md_text = pymupdf4llm.to_markdown(file_path)
            
            # Создаем документ из Markdown текста
            from langchain.schema import Document
            docs = [Document(
                page_content=md_text,
                metadata={
                    "user_id": user_id,
                    "source": os.path.basename(file_path),
                    "format": "pdf"
                }
            )]
            
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            # Добавляем метаданные
            for doc in docs:
                doc.metadata["user_id"] = user_id
                doc.metadata["source"] = os.path.basename(file_path)
                
        else:
            return "Формат не поддерживается"

        # Разбиваем на фрагменты и добавляем в векторное хранилище
        splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)
        return f"Добавлено {len(splits)} фрагментов."
    
    def get_sections_summary(self, user_id: int = None) -> str:
        """Возвращает список всех разделов базы знаний.

        ИЗМЕНЕНИЕ: фильтр по user_id убран — показываются все разделы,
        добавленные любым пользователем. Параметр user_id сохранён
        в сигнатуре для обратной совместимости с вызовом из бота.
        """
        sections = self._get_user_sections()

        if not sections:
            return "В базе знаний пока нет разделов."

        summary = "📚 **Разделы базы знаний:**\n\n"
        for section in sections:
            summary += f"• **{section['name']}**\n"
            summary += f"  _{section['description']}_\n\n"

        return summary

    def _get_user_sections(self, user_id: int = None, limit: int = 50) -> List[Dict]:
        """Получить все разделы базы знаний.

        ИЗМЕНЕНИЕ: фильтр по user_id убран — возвращаются все разделы коллекции.
        Параметры user_id и limit сохранены для обратной совместимости.
        """
        results = self.qdrant_client.scroll(
            collection_name=SECTIONS_COLLECTION,
            limit=limit
        )

        sections = []
        for point in results[0]:
            sections.append({
                "id":          point.id,
                "name":        point.payload.get("section_name"),
                "description": point.payload.get("description"),
                "keywords":    point.payload.get("keywords", [])
            })

        return sections


    def _get_existing_sections(self, user_id: int) -> list:
            """Получает список всех уникальных разделов из Qdrant.

            ИЗМЕНЕНИЕ: фильтр по metadata.user_id удалён — разделы общие для всех.
            Параметр user_id сохранён в сигнатуре для обратной совместимости.
            """
            try:
                results, _ = self.qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    with_payload=["metadata.section"],
                    limit=1000
                )
                sections = {res.payload['metadata']['section'] for res in results if 'metadata' in res.payload and 'section' in res.payload['metadata']}
                return list(sections)
            except Exception as e:
                print(f"Ошибка при получении разделов: {e}")
                return []


    def _determine_section(self, text_preview: str, existing_sections: list) -> str:
        """Использует LLM для определения или создания раздела"""
        sections_list = ", ".join(existing_sections) if existing_sections else "Разделов пока нет"
        
        prompt = [
            SystemMessage(content=(
                "Ты — эксперт по языку Python. Классифицируй текст по разделам.\n"
                f"Текущие разделы: [{sections_list}].\n"
                "Если подходит существующий — назови его. Если нет — создай новый (1-3 слова).\n"
                "В ответе напиши ТОЛЬКО название."
            )),
            HumanMessage(content=f"Текст:\n{text_preview[:1500]}")
        ]
        
        res = self.llm.invoke(prompt)
        return res.content.strip().replace('"', '')

    def add_document(self, file_path: str, user_id: int):
        """Обрабатывает файл через Markdown-трансформацию и умное разбиение"""
        file_name = os.path.basename(file_path)
        md_content = ""
        file_format = ""

        # 1. КОНВЕРТАЦИЯ В MARKDOWN (Markdown-центричность)
        if file_path.endswith(".pdf"):
            md_content = pymupdf4llm.to_markdown(file_path)
            file_format = "pdf"
        
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
            md_content = f"# Таблица: {file_name}\n\n" + df.to_markdown(index=False)
            file_format = "excel"
            
        elif file_path.endswith((".txt", ".md")):
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            file_format = "markdown" if file_path.endswith(".md") else "text"
            
        else:
            return "Формат не поддерживается"

        if not md_content.strip():
            return "Файл пуст"

        # 2. ОПРЕДЕЛЕНИЕ РАЗДЕЛА (Метаданные)
        existing_sections = self._get_user_sections(user_id)
        #existing_sections = self._get_existing_sections(user_id)
        #section = self._determine_section(md_content, existing_sections)
        
        # 3. Классифицируем документ
        classification = self._classify_document(md_content, existing_sections)
        section_name = classification["section_name"]
        section_description = classification["description"]
        keywords = classification.get("keywords", [])
        
        # 5. Проверяем, нужно ли создать новый раздел
        section_id = None

        if classification.get("is_new", True):
            # Дополнительная проверка через векторный поиск
            similar = self._find_similar_section(section_description, user_id)

            if similar:
                print(f"🔍 Найден похожий раздел: '{similar['name']}' (score: {similar['score']:.2f})")
                section_id = similar['id']
                section_name = similar['name']
                # Раздел существует — пополняем его ключевые слова новыми из текущего документа
                self._update_section_keywords(section_id, keywords)
            else:
                # Создаём новый раздел
                section_id = self._create_section(
                    section_name, section_description, keywords, user_id
                )
        else:
            # Ищем существующий раздел по имени
            for section in existing_sections:
                if section['name'] == section_name:
                    section_id = section['id']
                    break

            if section_id:
                # Раздел существует — пополняем его ключевые слова
                self._update_section_keywords(section_id, keywords)
            else:
                # Раздела нет — создаём
                section_id = self._create_section(
                    section_name, section_description, keywords, user_id
                )


        # 4. УМНОЕ РАЗБИЕНИЕ (MarkdownHeaderTextSplitter)
        
        # Определяем заголовки, по которым будем резать
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = md_splitter.split_text(md_content)

        # Дорезаем длинные куски рекурсивным сплиттером
        # chunk_size=CHUNK_SIZE — ограничен лимитом GigaChat Embeddings (514 токенов).
        # Было 1000 символов → вызывало ошибку 413 "Tokens limit exceeded".
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        final_splits = recursive_splitter.split_documents(header_splits)

        # 4. ОБОГАЩЕНИЕ МЕТАДАННЫХ
        for split in final_splits:
            split.metadata.update({
                "user_id": user_id,
                "source": file_name,
                "format": file_format,
                # МЕТАДАННЫЕ РАЗДЕЛА
                "section_id": section_id,
                "section_name": section_name,
                "section_description": section_description,
                "keywords": keywords
            })

        # 5. ЗАГРУЗКА В QDRANT
        self.vector_store.add_documents(final_splits)
        
        return f"Раздел: {section}. Файл разбит на {len(final_splits)} смысловых чанков."

    def _classify_document(self, text: str, existing_sections: List[Dict]) -> Dict:
        """
        Классифицирует документ и определяет раздел.
        Возвращает: {"section_name": str, "description": str, "is_new": bool}
        """
        # Формируем список существующих разделов
        sections_list = "\n".join([
            f"- {s['name']}: {s['description']}" 
            for s in existing_sections
        ]) if existing_sections else "Разделов пока нет."
        
        prompt = f"""Проанализируй следующий текст и определи его тематический раздел.

        СУЩЕСТВУЮЩИЕ РАЗДЕЛЫ:
        {sections_list}

        ТЕКСТ ДЛЯ АНАЛИЗА:
        {text[:2000]}...

        ЗАДАЧА:
        1. Если текст относится к одному из существующих разделов - выбери его название ТОЧНО как указано
        2. Если текст не подходит ни к одному разделу - придумай новое короткое название раздела

        ОТВЕТЬ СТРОГО В ФОРМАТЕ JSON:
        {{
            "section_name": "Название раздела",
            "description": "Краткое описание темы (1-2 предложения)",
            "keywords": ["ключевое слово 1", "ключевое слово 2", "ключевое слово 3"],
            "is_new": true/false
        }}

        Только JSON, без дополнительного текста!!!"""
        
        response = self.llm.invoke(prompt)
        
        # Парсим JSON из ответа
        try:
            # Убираем markdown разметку если есть
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            return result
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            print(f"Ответ LLM: {response.content}")
            # Возвращаем дефолтный раздел
            return {
                "section_name": "Общие материалы",
                "description": "Общие материалы без определенной категории",
                "keywords": [],
                "is_new": True
            }

    def _find_similar_section(self, section_description: str, user_id: int) -> Optional[Dict]:
        """Поиск похожего раздела через векторный поиск.
        
        ИСПРАВЛЕНИЕ: qdrant-client >= 1.7.0 убрал метод search() — используем
        query_points(), который является его актуальной заменой.
        Результаты находятся в атрибуте .points возвращаемого объекта.
        """
        query_vector = self.embeddings.embed_query(section_description)
        
        response = self.qdrant_client.query_points(
            collection_name=SECTIONS_COLLECTION,
            query=query_vector,              # вместо query_vector=
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=1,
            score_threshold=0.85,
            with_payload=True                # явно запрашиваем payload
        )
        results = response.points           # <-- в search() был просто список
        
        if results:
            return {
                "id":          results[0].id,
                "name":        results[0].payload.get("section_name"),
                "description": results[0].payload.get("description"),
                "score":       results[0].score
            }
        
        return None
    
    def find_section_for_query(self, query: str, user_id: int = None) -> Optional[Dict]:
        """Определяет раздел базы знаний для вопроса пользователя.

        Гибридный поиск в два шага:

        ШАГ 1 — Keyword matching (быстро, без embeddings):
            Берём все разделы из коллекции sections и проверяем, встречается ли
            хотя бы одно ключевое слово раздела в тексте вопроса (регистронезависимо).
            Если найдено совпадение — возвращаем раздел сразу, без обращения к Qdrant.

            Приоритет отдаётся разделу с наибольшим количеством совпавших ключевых слов —
            это снижает вероятность ложного срабатывания на короткие общие слова.

        ШАГ 2 — Vector search (точно, но дороже):
            Если ни одно ключевое слово не совпало — векторизуем вопрос и ищем
            ближайший раздел по косинусной схожести (порог 0.70).
            Этот шаг используется как запасной вариант для вопросов, сформулированных
            иначе, чем ключевые слова, но семантически близких к теме раздела.

        Параметр user_id сохранён в сигнатуре для обратной совместимости.
        """
        QUERY_SECTION_THRESHOLD = 0.70
        query_lower = query.lower()

        # --- ШАГ 1: Keyword matching ---
        all_sections = self._get_user_sections()   # все разделы из коллекции sections

        best_keyword_match = None
        best_keyword_count = 0

        for section in all_sections:
            keywords = section.get("keywords", [])
            if not keywords:
                continue

            # Считаем, сколько ключевых слов раздела встретилось в вопросе
            matched_keywords = [kw for kw in keywords if kw.lower() in query_lower]
            count = len(matched_keywords)

            if count > best_keyword_count:
                best_keyword_count = count
                best_keyword_match = section
                print(f"🔑 Keyword hit: раздел '{section['name']}', "
                      f"совпавшие слова: {matched_keywords}")

        if best_keyword_match:
            print(f"✅ Раздел найден по ключевым словам ({best_keyword_count} совп.): "
                  f"'{best_keyword_match['name']}'")
            return best_keyword_match

        # --- ШАГ 2: Vector search (запасной вариант) ---
        print("🔍 Keyword matching не дал результата — переходим к векторному поиску...")

        query_vector = self.embeddings.embed_query(query)

        response = self.qdrant_client.query_points(
            collection_name=SECTIONS_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=QUERY_SECTION_THRESHOLD,
            with_payload=True
        )
        results = response.points

        if not results:
            print(f"🔍 Раздел не определён (нет совпадений выше {QUERY_SECTION_THRESHOLD})")
            return None

        best = results[0]
        matched = {
            "id":       best.id,
            "name":     best.payload.get("section_name"),
            "keywords": best.payload.get("keywords", []),
            "score":    best.score
        }
        print(f"✅ Раздел найден векторным поиском: '{matched['name']}' "
              f"(score: {matched['score']:.2f})")
        return matched


    def _create_section(self, section_name: str, description: str, 
                       keywords: List[str], user_id: int) -> str:
        """Создает новый раздел в базе"""
        section_id = str(uuid.uuid4())
        
        # Векторизуем описание раздела
        vector = self.embeddings.embed_query(f"{section_name}. {description}")
        
        # Добавляем в Qdrant
        self.qdrant_client.upsert(
            collection_name=SECTIONS_COLLECTION,
            points=[
                PointStruct(
                    id=section_id,
                    vector=vector,
                    payload={
                        "section_name": section_name,
                        "description": description,
                        "keywords": keywords,
                        "user_id": user_id,
                        "doc_count": 0
                    }
                )
            ]
        )
        
        print(f"✅ Создан новый раздел: '{section_name}'")
        return section_id

    def _update_section_keywords(self, section_id: str, new_keywords: List[str]):
        """Пополняет список ключевых слов существующего раздела.

        Вызывается каждый раз, когда новый документ или заметка попадают
        в уже существующий раздел. Новые ключевые слова из классификации
        мержатся с текущими: дубликаты отбрасываются (сравнение без учёта
        регистра), порядок сохраняется — сначала старые, потом новые.

        Используем set_payload вместо upsert — это точечное обновление
        конкретного поля payload без перезаписи вектора и остальных полей.
        """
        if not new_keywords:
            return

        # Читаем текущее состояние раздела из Qdrant
        result = self.qdrant_client.retrieve(
            collection_name=SECTIONS_COLLECTION,
            ids=[section_id],
            with_payload=True
        )
        if not result:
            print(f"⚠️  Раздел {section_id} не найден при обновлении ключевых слов")
            return

        existing_keywords: List[str] = result[0].payload.get("keywords", [])

        # Мержим: добавляем только те новые слова, которых ещё нет (без учёта регистра)
        existing_lower = {kw.lower() for kw in existing_keywords}
        added = [kw for kw in new_keywords if kw.lower() not in existing_lower]

        if not added:
            print("🔑 Новых ключевых слов нет — раздел не изменён")
            return

        merged_keywords = existing_keywords + added

        # Точечно обновляем только поле keywords в payload
        self.qdrant_client.set_payload(
            collection_name=SECTIONS_COLLECTION,
            payload={"keywords": merged_keywords},
            points=[section_id]
        )
        print(f"🔑 Раздел обновлён: добавлены слова {added} "
              f"(итого: {len(merged_keywords)} ключевых слов)")



    @traceable(name="get_answer", run_type="tool") # <--- Декорируем функцию
    def get_answer(self, query: str, user_id: int, chat_history: List[Dict]):
        """RAG пайплайн (устаревший метод, оставлен для совместимости).

        ИЗМЕНЕНИЕ: фильтр по metadata.user_id удалён — поиск по всей базе.
        """
        # Поиск без фильтра по user_id — база знаний общая
        search_results = self.vector_store.similarity_search(query, k=6)

        context_text = "\n\n".join([doc.page_content for doc in search_results])
        if not context_text:
            context_text = "Нет информации в базе знаний."

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        prompt = f"""
        Ты программист по python. Отвечай на вопрос, используя ТОЛЬКО контекст и историю. Ничего не придумывай!
        Если в контексте и истории ответ не найден, тогда сообщи, "в базе знаний ответ не найден".

        Контекст:
        {context_text}
        
        История диалога:
        {history_text}
        
        Вопрос пользователя: {query}
        """

        response = self.llm.invoke(prompt)
        return response.content
    
    @traceable(name="get_relevants", run_type="tool") # <--- Декорируем функцию
    def get_relevants(self, query: str, user_id: int, numberofchunks: int,
                      section_name: Optional[str] = None):
        """Семантический поиск по всей базе знаний.

        ИЗМЕНЕНИЕ: фильтр по metadata.user_id удалён — поиск ведётся по всей
        коллекции, независимо от того, кто загрузил данные. База знаний общая.

        Единственный оставшийся фильтр — section_name (опциональный):
        - Если раздел определён classify_query_node — ищем только в нём.
        - Если None — поиск по всей базе без каких-либо фильтров.

        Параметр user_id сохранён в сигнатуре для обратной совместимости
        (он передаётся из графа), но в запрос больше не подставляется.
        """
        if section_name:
            # Фильтр только по разделу — без привязки к пользователю
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=numberofchunks,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.section_name",
                            match=models.MatchValue(value=section_name)
                        )
                    ]
                )
            )
            print(f"🔎 Поиск с фильтром по разделу: '{section_name}'")
        else:
            # Никаких фильтров — поиск по всей базе
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=numberofchunks,
            )
            print("🔎 Поиск по всей базе знаний (раздел не определён)")

        return search_results
    
    @traceable(name="rerank_relevants", run_type="tool") # <--- Декорируем функцию
    def rerank_relevants(self, documents):
        """Сортируем по убыванию схожести и берём топ-M выше порога.
        
        ИЗМЕНЕНИЕ: Для каждого чанка определяем его раздел из поля metadata.section_name.
        Если поле отсутствует (старые чанки без классификации) — ставим «Без раздела».
        Возвращаем список кортежей (doc, score, section_name) вместо (doc, score),
        чтобы информация о разделе была доступна на следующих шагах пайплайна.
        """
        M = 5
        threshold = 0.7

        # Фильтруем по порогу и сортируем по убыванию score
        ranked_docs = sorted(
            [(doc, score) for doc, score in documents if score >= threshold],
            key=lambda x: x[1],
            reverse=True
        )[:M]

        # НОВЫЙ БЛОК: обогащаем каждый чанк именем раздела из его метаданных.
        # metadata — словарь, который LangChain сохраняет в payload Qdrant.
        # Поле section_name туда попадает при add_document и теперь при add_text.
        ranked_with_sections = []
        for doc, score in ranked_docs:
            section_name = doc.metadata.get("section_name", "Без раздела")
            ranked_with_sections.append((doc, score, section_name))

        # Логируем для отладки: какие разделы попали в контекст
        unique_sections = sorted({s for _, _, s in ranked_with_sections})
        print(f"📂 Разделы в контексте запроса: {unique_sections}")

        return ranked_with_sections
    
    @traceable(name="generate_answer", run_type="tool") # <--- Декорируем функцию
    def generate_answer(self, final_context_docs, query: str):
        """Итоговый ответ пользователю.
        
        ИЗМЕНЕНИЕ: Принимает кортежи (doc, score, section_name).
        В промпт передаётся контекст, где каждый чанк помечен своим разделом.
        Это позволяет LLM понимать, из какой области знаний взята информация,
        и при необходимости упомянуть источник в ответе.
        Также возвращает список уникальных разделов — для отображения в боте.
        """
        if not final_context_docs:
            # Возвращаем AIMessage-совместимый объект с пустым контекстом
            response = self.llm.invoke(
                "Скажи пользователю: 'В базе знаний ответ не найден.'"
            )
            response.sections_used = []
            return response

        # НОВЫЙ БЛОК: строим контекст с метками разделов.
        # Формат каждого блока: "[Раздел: <название>]\n<текст чанка>"
        # Это явно показывает LLM, что фрагменты могут относиться к разным темам.
        context_blocks = []
        for doc, score, section_name in final_context_docs:
            block = f"[Раздел: {section_name}]\n{doc.page_content}"
            context_blocks.append(block)

        context_text = "\n\n---\n\n".join(context_blocks)

        # Собираем уникальные разделы (сохраняем порядок появления)
        seen = set()
        sections_used = []
        for _, _, section_name in final_context_docs:
            if section_name not in seen:
                seen.add(section_name)
                sections_used.append(section_name)

        prompt = f"""
        Ты программист по python. Отвечай на вопрос, используя ТОЛЬКО контекст. Ничего не придумывай!
        Если в контексте ответ не найден, тогда сообщи ТОЛЬКО ЭТУ ФРАЗУ: "в базе знаний ответ не найден".
        Контекст разбит на фрагменты. Каждый фрагмент помечен разделом базы знаний, из которого он взят.
        
        Контекст:
        {context_text}
        
        Вопрос пользователя: {query}
        """

        response = self.llm.invoke(prompt)

        # Прикрепляем список разделов прямо к объекту ответа —
        # так бот сможет прочитать их без дополнительных методов.
        response.sections_used = sections_used

        return response