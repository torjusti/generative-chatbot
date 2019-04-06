from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import os

from data import get_utterance_pairs, get_word_map


# Pair utterances as inputs and outputs.
input_utterances, target_utterances = get_utterance_pairs()

# Create dictionaries which map tokens from input and output
# utterances to an unique number, and vice versa.
input_token_to_num, num_to_input_token = get_word_map(input_utterances)
target_token_to_num, num_to_target_token = get_word_map(target_utterances)

# The longest utterances which occur in the data.
max_encoder_seq_length = max(len(utterance) for utterance in input_utterances)
max_decoder_seq_length = max(len(utterance) for utterance in target_utterances)

# The number of different tokens in the data.
num_encoder_tokens = len(input_token_to_num)
num_decoder_tokens = len(target_token_to_num)

encoder_input_data = np.zeros((len(input_utterances), max_encoder_seq_length), dtype='float32')
decoder_input_data = np.zeros((len(target_utterances), max_decoder_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(target_utterances), max_decoder_seq_length, num_decoder_tokens), dtype='float32')


for i, (input_utterance, target_utterance) in enumerate(zip(input_utterances, target_utterances)):
    # Splice to remove <u> and </u>
    target_utterance = target_utterance[1:-1]
    for j, token in enumerate(input_utterance):
        # One-hot encoding for which tokens appear in the encoder input data.
        encoder_input_data[i][j] = input_token_to_num[token]

    for k, token in enumerate(target_utterance):
        # One-hot encoding for decoder input data.
        decoder_input_data[i][j] = input_token_to_num[token]

        if k > 0:
            # One-hot encoding for decoder targets. This is the same
            # as the decoder input data, but the index at which we start
            # adding tokens from is shifted to the right by one so that
            # the special token marking the start of an utterance is
            # no longer included.
            decoder_target_data[i, k-1, target_token_to_num[token]] = 1

LATENT_DIM = 256
BATCN_SIZE = 64
NUM_EPOCHS = 10

encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, LATENT_DIM)(encoder_inputs)
x, state_h, state_c = LSTM(LATENT_DIM, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, LATENT_DIM)(decoder_inputs)
x = LSTM(LATENT_DIM, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

MODEL_PATH = 'models/model.h5'
if not os.path.isfile('models/model.h5'):
    # TODO: "Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!"
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCN_SIZE,
              epochs=NUM_EPOCHS,
              validation_split=0.2)
    model.save_weights(MODEL_PATH)
else: 
    model.load_weights(MODEL_PATH)

model.summary()
