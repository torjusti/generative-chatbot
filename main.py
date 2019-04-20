from Chatbot import Chatbot

def main():
    bot = Chatbot()
    bot.print_model()

    while True:
        print('Bot:', bot.reply(input('> ')))


if __name__ == '__main__':
    main()
