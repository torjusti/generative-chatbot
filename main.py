from Chatbot import Chatbot


if __name__ == '__main__':
    bot = Chatbot()
    bot.print_model()
    while True:
        print('Bot: ', bot.reply(input('> ')))

