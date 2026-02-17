import telebot
import os
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, trim_messages
from typing import TypedDict, List, Annotated, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from knowledge_base import KnowledgeBase

load_dotenv();

bot = telebot.TeleBot(os.environ.get("TELEGRAM_BOT_TOKEN"))
bot_username = bot.get_me().username  # Получаем имя бота
kb_service = KnowledgeBase()

# ==========================================
# Настройка LangGraph
# ==========================================

def process_message_node(state: MessagesState, config: RunnableConfig):
    """
    Узел графа, который обращается к базе знаний.
    LangGraph сам хранит всю историю в state["messages"].
    """
    # Получаем ID пользователя из конфигурации (LangGraph передает его как thread_id)
    user_id = int(config["configurable"]["thread_id"])
    
    messages = state["messages"]
    current_text = messages[-1].content
    
    # Формируем историю для kb_service (как в вашей старой логике — берем последние 10 сообщений)
    # Исключаем самое последнее сообщение, так как это текущий вопрос
    history_messages = messages[-11:-1] if len(messages) > 1 else []
    
    history = []
    for msg in history_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append({"role": role, "content": msg.content})

    # Получаем ответ от базы знаний (внутри которой, видимо, крутится RAG и LLM)
    answer = kb_service.get_answer(current_text, user_id, history)
    
    # Возвращаем ответ. LangGraph автоматически добавит его в общую историю сообщений.
    return {"messages": [AIMessage(content=answer)]}

# 1. Определяем состояние графа
class GraphState(TypedDict):
    user_id: int
    query: str
    # add_messages позволяет накапливать историю диалога
    messages: Annotated[List[BaseMessage], add_messages]
    # Список документов, найденных в базе
    retrieved_docs: List[dict] 
    # Итоговый контекст после ранжирования
    final_retrieved_docs: List[dict] 

# 2. Узлы графа
def retrieve_node(state: GraphState, config: RunnableConfig):
    """Шаг 1: Получаем топ N релевантных результатов"""
    user_id = state["user_id"]
    query = state["query"]
    
    # Получаем документы (N=10)
    # Предполагаем, что метод возвращает список объектов со свойством page_content или словарей
    raw_docs = kb_service.get_relevants(query, user_id, 15)
    
    # formatted_docs = []
    # for doc in raw_docs:
    #     # Универсальная обработка: текст и метаданные (если есть скор)
    #     text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    #     score = doc.metadata.get('score', 0) if hasattr(doc, 'metadata') else 0
    #     formatted_docs.append({"text": text, "score": score})
        
    return {"retrieved_docs": raw_docs}

def rerank_node(state: GraphState):
    """Шаг 2: Reranking (выбираем M=3 лучших из N=10)"""
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        return {"final_retrieved_docs": "Информация в базе знаний не найдена."}

    final_retrieved_docs = kb_service.rerank_relevants(docs)

    return {"final_retrieved_docs": final_retrieved_docs}

def generate_node(state: GraphState):
    # Шаг 3: Генерация ответа
    response = kb_service.generate_answer(state["final_retrieved_docs"], state["query"])
    return {"messages": [response]}

# 3. Сборка графа
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ==========================================
# Хэндлеры Telegram бота
# ==========================================

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 
                 "Привет! Я бот с памятью на базе LLM.\n" 
                 "1. Пришли PDF файл — я его прочитаю и сохраню в базу знаний.\n" 
                 "2. Напиши 'Запомни: [текст]' — я сохраню заметку в базу знаний.\n"
                 "3. Задай вопрос — я отвечу по базе знаний.\n"
                 "4. /help - список доступных команд")

@bot.message_handler(func=lambda message: message.chat.type in ['group', 'supergroup'])
def handle_group_message(message):
    if f'@{bot_username}' in message.text:
        bot.reply_to(message, "Слушаю внимательно!")

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = '''Список доступных команд бота:\n
                1. /clear - очистка базы знаний пользователя \n            
                2. /clean - очистка ВСЕЙ базы знаний'''
    bot.send_message(message.chat.id, help_text)

@bot.message_handler(commands=['clear'])
def clear_db(message):
    kb_service.clear_user_db(message.from_user.id)
    # Опционально: здесь можно было бы очищать и память LangGraph для конкретного thread_id, 
    # но MemorySaver в базовом виде хранит всё. Обычно в таких случаях просто меняют thread_id.
    bot.send_message(message.chat.id, "База знаний пользователя очищена!")

@bot.message_handler(commands=['clean'])
def clear_db(message):
    kb_service.clean_db()
    bot.send_message(message.chat.id, "База знаний очищена!")

@bot.message_handler(content_types=['document'])
def handle_docs(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        file_name = message.document.file_name
        
        # Скачиваем
        downloaded_file = bot.download_file(file_info.file_path)
        
        os.makedirs("temp", exist_ok=True)
        save_path = f"temp/{file_name}"
        
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        msg = bot.reply_to(message, "Читаю файл и векторизую...")
        
        # Добавляем в базу
        result = kb_service.add_document(save_path, message.from_user.id)
        
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg.message_id, 
                              text=f"✅ Файл '{file_name}' обработан. {result}")
        
        # Удаляем локальную копию
        os.remove(save_path)
        
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

@bot.message_handler(content_types=['text'])
def handler_message(message):
    user_id = message.from_user.id
    text = message.text

    # Обработка прямой команды на запоминание
    if text.lower().startswith("запомни:"):
        content = text[8:].strip()
        if content:
            kb_service.add_text(content, user_id)
            bot.reply_to(message, "✅ Записал в базу знаний.")
        else:
            bot.reply_to(message, "Текст пустой.")
        return
    
    wait_msg = bot.reply_to(message, "🤔 Анализ данных...")
    
    try:
        # Настраиваем конфигурацию LangGraph (связываем историю с ID пользователя)
        config = {"configurable": {"thread_id": str(user_id)}}
        input_messages = [HumanMessage(content=text)]
        
        # Запускаем граф! Он сам добавит вопрос в память, вызовет узел и сохранит ответ
        # Формируем входные данные для графа
        input_state = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "query": text
        }
        output = app.invoke(input_state, config)
        
        # Извлекаем финальный ответ из состояния
        bot_answer = output["messages"][-1].content
        
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.send_message(message.chat.id, bot_answer, parse_mode="Markdown")
        
    except Exception as e:
        bot.edit_message_text(chat_id=message.chat.id, message_id=wait_msg.message_id, 
                              text=f"Ошибка генерации: {e}")

# ==========================================
# Запуск
# ==========================================

def main():
    print("Бот запущен...")
    bot.polling(none_stop=True)

if __name__ == '__main__':
    main()