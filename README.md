# generative-chatbot

Generative chatbot project created during the course TDT4310.

## Environment variables

| Variable            | Description                                       | Example value  | Default value |
|---------------------|---------------------------------------------------|----------------|---------------|
| MAX_NUM_TOKENS      | Maximum number of tokens allowed in a sentence.   | 50             | 50            |
| MAX_NUM_UTTERANCES  | Maximum number of utterances to load from corpus. | 5000           | 5000          |
| TARGET_USER         | Only use this username to generate responses.     | Torjus Iveland | None          |
| VERIFY_UTTERANCES   | Check the quality of utterances.                  | False          | True          |
| REMOVE_SELF_REPLIES | Filter away self-responses.                       | False          | True          |
| MAX_VOCABULARY_SIZE | Max number of different tokens to consider.       | 7500           | 5000          |