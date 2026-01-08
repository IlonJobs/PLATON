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

load_dotenv()

# Настройки
COLLECTION_NAME = "telegram_gemini_kb"
# Используем модель Gemini 1.5 Flash (быстрая и дешевая/бесплатная)
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

class KnowledgeBase:
    def __init__(self):
        # 1. Настройка Embeddings от Google
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # 2. Клиент Qdrant
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Создаем коллекцию, если нет
        if not self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,  # Размерность text-embedding-004
                    distance=models.Distance.COSINE
                )
            )

        # Интеграция LangChain и Qdrant
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings
        )

        # 3. LLM Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.3,
            convert_system_message_to_human=True # Иногда нужно для старых версий langchain
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_text(self, text: str, user_id: int, source: str = "message"):
        """Добавляет текст в базу (синхронно)"""
        docs = [Document(page_content=text, metadata={"user_id": user_id, "source": source})]
        splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)

    def add_document(self, file_path: str, user_id: int):
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

    def get_answer(self, query: str, user_id: int, chat_history: List[Dict]):
        """RAG пайплайн"""
        # Поиск с фильтром по user_id
        search_results = self.vector_store.similarity_search(
            query,
            k=4,
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
        Ты помощник. Отвечай на вопрос, используя контекст и историю.
        
        Контекст:
        {context_text}
        
        История диалога:
        {history_text}
        
        Вопрос пользователя: {query}
        """

        # Gemini принимает список сообщений или строку
        response = self.llm.invoke(prompt)
        return response.content