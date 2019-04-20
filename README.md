# generative-chatbot

Generative chatbot project created during the course TDT4310.

## Environment variables

| Variable            | Description                                       | Example value  | Default value |
|---------------------|---------------------------------------------------|----------------|---------------|
| MAX_NUM_TOKENS      | Maximum number of tokens allowed in a sentence.   | 50             | 20            |
| MAX_NUM_UTTERANCES  | Maximum number of utterances to load from corpus. | 5000           | 250000        |
| TARGET_USER         | Only use this username to generate responses.     | Torjus Iveland | None          |
| REMOVE_SELF_REPLIES | Filter away self-responses.                       | False          | True          |
| MAX_VOCABULARY_SIZE | Max number of different tokens to consider.       | 7500           | 1000          |
| USE_CORNELL_CORPUS  | Whether or not to use an English corpus.          | True           | False         |
