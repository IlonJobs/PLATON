import telebot
import os
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from knowledge_base import KnowledgeBase

load_dotenv();

bot = telebot.TeleBot(os.environ.get("TELEGRAM_BOT_TOKEN"))
bot_username = bot.get_me().username  # Получаем имя бота
kb_service = KnowledgeBase()

# История в памяти 
user_histories = {}

def update_history(user_id, role, content):
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append({"role": role, "content": content})
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

# реагируем на команду /start
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
    pass

# реагируем на команду /help
@bot.message_handler(commands=['help'])
def help(message):
    user = message.chat.id
    config = {"configurable": {"thread_id": user}}
    help_text   =   "Список доступных команд бота:\n " \
                    "1. /clear - очистка базы знаний"
    bot.send_message(user, help_text)

# реагируем на команду /clear
@bot.message_handler(commands=['clear'])
def clear_db(message):
    kb_service.clear_user_db(message.from_user.id)
    bot.send_message(message.chat.id, "База знаний очищена!")


# Обработка файлов (документов)
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
    config = {"configurable": {"thread_id": user_id}}
    text = message.text

    if text.lower().startswith("запомни:"):
        content = text[8:].strip()
        if content:
            kb_service.add_text(content, user_id)
            bot.reply_to(message, "✅ Записал в базу знаний.")
        else:
            bot.reply_to(message, "Текст пустой.")
        return
    
    # Сценарий RAG (ответ на вопрос)
    wait_msg = bot.reply_to(message, "🤔 Анализ данных...")
    
    try:
        history = user_histories.get(user_id, [])
        answer = kb_service.get_answer(text, user_id, history)
        
        # Обновляем историю
        update_history(user_id, "user", text)
        update_history(user_id, "assistant", answer)
        
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.send_message(message.chat.id, answer, parse_mode="Markdown")
        
    except Exception as e:
        bot.edit_message_text(chat_id=message.chat.id, message_id=wait_msg.message_id, 
                              text=f"Ошибка генерации: {e}")

    '''
    input_messages = [HumanMessage(text)]
    output = app.invoke({"messages": input_messages}, config)
    bot_anwser = output["messages"][-1].content
    bot.send_message(message.chat.id, bot_anwser)
    '''

# Функция main
def main():
    bot.polling(none_stop=True)

# Запускаем программу
if __name__ == '__main__':
    model = GigaChat(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2",
        verify_ssl_certs=False,
    )
    # Инициализируйте граф
    workflow = StateGraph(state_schema=MessagesState)


    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Задайте вершину графа
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Добавьте память
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    main()