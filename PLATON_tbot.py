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


# реагируем на команду /start
@bot.message_handler(commands=['start'])
def help(message):
    user = message.chat.id
    bot.send_message(user, "Стартуем!")
    
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
    bot.send_message(user, str(app.get_state(config)))



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
    
    input_messages = [HumanMessage(text)]
    output = app.invoke({"messages": input_messages}, config)
    bot_anwser = output["messages"][-1].content
    bot.send_message(message.chat.id, bot_anwser)

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