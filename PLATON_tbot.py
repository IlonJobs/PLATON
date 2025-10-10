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

load_dotenv();

bot = telebot.TeleBot(os.environ.get("TELEGRAM_BOT_TOKEN"))


@bot.message_handler(func=lambda _: True)
def handler_message(message):
    config = {"configurable": {"thread_id": message.from_user.id}}
    query = message.text
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    bot_anwser = output["messages"][-1].content
    bot.send_message(message.chat.id, bot_anwser)

# реагируем на команду /help
@bot.message_handler(commands=['help'])
def help(message):
    user = message.chat.id
    bot.send_message(user, "Это бот ПЛАТОН! ")

# Функция main
def main():
    bot.polling(none_stop=True)

# Запускаем программу
if __name__ == '__main__':
    model = GigaChat(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Max",
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