import os
from typing import List, Dict

# Импорты для Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langsmith import traceable
import pymupdf4llm
import pathlib
import pymupdf
from langchain.schema import Document

load_dotenv()

# Настройки
COLLECTION_NAME = "telegram_kb"
# Используем модель Gemini 1.5 Flash (быстрая и дешевая/бесплатная)
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

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

        self.qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.user_id",       # Путь к полю в JSON
            field_schema=models.PayloadSchemaType.INTEGER # Тип данных (число)
        )

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
            chunk_size=500,
            chunk_overlap=100
        )

    def add_text(self, text: str, user_id: int, source: str = "message"):
        """Добавляет текст в базу (синхронно)"""
        docs = [Document(page_content=text, metadata={"user_id": user_id, "source": source})]
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
    

    def add_document(self, file_path: str, user_id: int):
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
            
        elif file_path.endswith(".txt"):
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
    
    @traceable(name="get_answer", run_type="tool") # <--- Декорируем функцию
    def get_answer(self, query: str, user_id: int, chat_history: List[Dict]):
        """RAG пайплайн"""
        # Поиск с фильтром по user_id
        search_results = self.vector_store.similarity_search(
            query,
            k=6,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        )

        context_text = "\n\n".join([doc.page_content for doc in search_results])
        if not context_text:
            context_text = "Нет информации в базе знаний."

        # Формирование истории для промпта
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

        # LLM принимает список сообщений или строку
        response = self.llm.invoke(prompt)
        return response.content