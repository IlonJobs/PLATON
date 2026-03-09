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
from typing import TypedDict, List, Annotated, Union, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from knowledge_base import KnowledgeBase

load_dotenv()

bot = telebot.TeleBot(os.environ.get("TELEGRAM_BOT_TOKEN"))
bot_username = bot.get_me().username
kb_service = KnowledgeBase()

# ==========================================
# Контроль доступа на запись
# ==========================================

def _load_allowed_writers() -> set:
    """Загружает список user_id, которым разрешено добавлять данные в базу знаний.

    Переменная окружения ALLOWED_WRITER_IDS содержит user_id через запятую:
        ALLOWED_WRITER_IDS=123456789,987654321

    Правила:
    - Пробелы вокруг запятых игнорируются.
    - Нечисловые значения пропускаются с предупреждением (защита от опечаток).
    - Если переменная не задана или пуста — множество будет пустым,
      и никто не сможет писать в базу. Это безопасное поведение по умолчанию.

    Возвращает: set[int]
    """
    raw = os.environ.get("ALLOWED_WRITER_IDS", "")
    if not raw.strip():
        print("⚠️  ALLOWED_WRITER_IDS не задан — запись в базу знаний отключена для всех.")
        return set()

    allowed = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            allowed.add(int(part))
        else:
            print(f"⚠️  ALLOWED_WRITER_IDS: пропущено некорректное значение '{part}'")

    print(f"✅ Запись в базу знаний разрешена для user_id: {allowed}")
    return allowed

# Множество загружается один раз при старте бота.
# Для изменения списка достаточно перезапустить бота.
ALLOWED_WRITERS: set = _load_allowed_writers()


def require_write_access(handler_func):
    """Декоратор: проверяет право на запись в базу знаний.

    Применяется к хэндлерам, которые изменяют базу знаний:
        - handle_docs     — загрузка документа
        - handler_message — команда «Запомни:»
        - clear_db        — /clear, очистка базы пользователя
        - clean_db_cmd    — /clean, полная очистка базы

    Если user_id отправителя НЕ входит в ALLOWED_WRITERS:
        - бот отвечает сообщением об отказе
        - оригинальный хэндлер не вызывается
        - в консоль пишется предупреждение для мониторинга

    Хэндлеры только на чтение (вопросы) декоратор не получают —
    задавать вопросы может любой пользователь.
    """
    def wrapper(message):
        user_id = message.from_user.id
        if user_id not in ALLOWED_WRITERS:
            print(f"🚫 Отказ в записи: user_id={user_id} (@{message.from_user.username})")
            bot.reply_to(
                message,
                "⛔ У вас нет прав для добавления информации в базу знаний.\n"
                "Обратитесь к администратору бота."
            )
            return
        return handler_func(message)
    # Сохраняем имя оригинальной функции — telebot использует его для маршрутизации
    wrapper.__name__ = handler_func.__name__
    return wrapper


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
    # НОВОЕ: раздел, к которому отнесён вопрос пользователя.
    # None — раздел не определён, поиск пойдёт по всей базе.
    # str  — имя конкретного раздела, поиск будет отфильтрован по нему.
    matched_section: Optional[str]

# 2. Узлы графа

def classify_query_node(state: GraphState) -> GraphState:
    """Шаг 0: Определяем раздел базы знаний, к которому относится вопрос.

    Вызывает kb_service.find_section_for_query — метод делает векторный поиск
    по коллекции sections и возвращает наиболее похожий раздел (или None).

    Результат записывается в state["matched_section"]:
    - строка с именем раздела → следующий шаг будет искать только в нём
    - None → следующий шаг будет искать по всей базе пользователя

    Этот узел не трогает state["messages"] и не вызывает LLM —
    только один векторный запрос в Qdrant, поэтому он быстрый и дешёвый.
    """
    user_id = state["user_id"]
    query   = state["query"]

    matched = kb_service.find_section_for_query(query, user_id)

    if matched:
        return {"matched_section": matched["name"]}
    else:
        return {"matched_section": None}


def retrieve_node(state: GraphState, config: RunnableConfig):
    """Шаг 1: Получаем топ N релевантных результатов.

    ИЗМЕНЕНИЕ: читаем state["matched_section"].
    - Если раздел определён — передаём его в get_relevants, и Qdrant вернёт
      только чанки из этого раздела данного пользователя.
    - Если None — поиск идёт по всей базе пользователя (прежнее поведение).
    """
    user_id         = state["user_id"]
    query           = state["query"]
    matched_section = state.get("matched_section")   # None или строка

    raw_docs = kb_service.get_relevants(query, user_id, 15,
                                        section_name=matched_section)
    return {"retrieved_docs": raw_docs}

def rerank_node(state: GraphState):
    """Шаг 2: Reranking — выбираем M=5 лучших из N=15.
    
    ИЗМЕНЕНИЕ: rerank_relevants теперь возвращает кортежи (doc, score, section_name).
    Сохраняем их как есть в final_retrieved_docs — section_name понадобится в generate_node.
    """
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        return {"final_retrieved_docs": []}

    # ranked_with_sections — список кортежей (doc, score, section_name)
    ranked_with_sections = kb_service.rerank_relevants(docs)

    return {"final_retrieved_docs": ranked_with_sections}

def generate_node(state: GraphState):
    """Шаг 3: Генерация ответа.
    
    ИЗМЕНЕНИЕ: generate_answer теперь возвращает объект с атрибутом sections_used.
    Мы читаем этот атрибут и сохраняем список разделов в состояние графа
    через дополнительное поле additional_kwargs у AIMessage — это стандартный
    способ прикрепить произвольные данные к сообщению в LangChain.
    """
    final_docs = state.get("final_retrieved_docs", [])

    if not final_docs:
        # Нет чанков выше порога — сообщаем без вызова LLM
        no_result_msg = AIMessage(
            content="В базе знаний ответ не найден.",
            additional_kwargs={"sections_used": []}
        )
        return {"messages": [no_result_msg]}

    response = kb_service.generate_answer(final_docs, state["query"])

    # Прикрепляем разделы к сообщению через additional_kwargs,
    # чтобы хэндлер бота мог их прочитать из output["messages"][-1]
    sections = getattr(response, "sections_used", [])
    result_msg = AIMessage(
        content=response.content,
        additional_kwargs={"sections_used": sections}
    )
    return {"messages": [result_msg]}

# 3. Сборка графа
workflow = StateGraph(GraphState)

workflow.add_node("classify_query", classify_query_node)  # НОВЫЙ узел
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)

# НОВАЯ топология: classify_query → retrieve → rerank → generate
#
#   START
#     ↓
#  classify_query   ← векторный поиск по коллекции sections
#     ↓                 matched_section = "Название" | None
#  retrieve         ← similarity_search с фильтром (или без)
#     ↓
#  rerank           ← топ-5 по score >= 0.7
#     ↓
#  generate         ← LLM формирует ответ
#     ↓
#    END

workflow.add_edge(START, "classify_query")
workflow.add_edge("classify_query", "retrieve")
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
                2. /clean - очистка ВСЕЙ базы знаний \n            
                3. /sections - показать все разделы базы знаний'''
    
    bot.send_message(message.chat.id, help_text)

@bot.message_handler(commands=['clear'])
@require_write_access   # очистка своей базы — тоже операция записи
def clear_db(message):
    kb_service.clear_user_db(message.from_user.id)
    # Опционально: здесь можно было бы очищать и память LangGraph для конкретного thread_id, 
    # но MemorySaver в базовом виде хранит всё. Обычно в таких случаях просто меняют thread_id.
    bot.send_message(message.chat.id, "База знаний пользователя очищена!")

@bot.message_handler(commands=['clean'])
@require_write_access   # полная очистка — только для доверенных пользователей
def clean_db_cmd(message):   # переименовано: в оригинале была дублирующая функция clear_db
    kb_service.clean_db()
    bot.send_message(message.chat.id, "База знаний очищена!")

@bot.message_handler(commands=['sections'])
def show_sections(message):
    """Показать все разделы пользователя"""
    user_id = message.from_user.id
    summary = kb_service.get_sections_summary(user_id)
    bot.send_message(message.chat.id, summary, parse_mode="Markdown")
    
@bot.message_handler(content_types=['document'])
@require_write_access   # только пользователи из ALLOWED_WRITER_IDS могут загружать файлы
def handle_docs(message):
    #try:
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
        
    #except Exception as e:
    #    bot.reply_to(message, f"Ошибка: {e}")

@bot.message_handler(content_types=['text'])
def handler_message(message):
    user_id = message.from_user.id
    text = message.text

    # Обработка прямой команды на запоминание
    if text.lower().startswith("запомни:"):
        # Проверяем права на запись только здесь — задавать вопросы может любой.
        # Декоратор не используем для всего хэндлера, чтобы не блокировать чтение.
        if user_id not in ALLOWED_WRITERS:
            print(f"🚫 Отказ в записи (Запомни): user_id={user_id} (@{message.from_user.username})")
            bot.reply_to(
                message,
                "⛔ У вас нет прав для добавления информации в базу знаний.\n"
                "Обратитесь к администратору бота."
            )
            return
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
            "messages":        [HumanMessage(content=text)],
            "user_id":         user_id,
            "query":           text,
            "matched_section": None,   # НОВОЕ: будет заполнено в classify_query_node
        }
        output = app.invoke(input_state, config)
        
        # Извлекаем финальный ответ из состояния
        last_msg = output["messages"][-1]
        bot_answer = last_msg.content

        # НОВЫЙ БЛОК: читаем разделы, которые были использованы при генерации ответа.
        # Если LLM нашла релевантные чанки — показываем пользователю, из каких
        # разделов базы знаний взята информация. Это повышает прозрачность ответа.
        sections_used = last_msg.additional_kwargs.get("sections_used", [])
        if sections_used:
            sections_footer = "\n\n📂 *Источники:* " + ", ".join(
                f"_{s}_" for s in sections_used
            )
            bot_answer = bot_answer + sections_footer

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