from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import os

from data import get_utterance_pairs, TokenMapper


LATENT_DIM = 256
BATCN_SIZE = 64
NUM_EPOCHS = 10
MODEL_PATH = 'models/model.h5'


class Chatbot():
    def __init__(self):
        self.__get_training_data()
        self.__build_model()
        self.__train()

    def __get_training_data(self):
        # Pair utterances as inputs and outputs.
        self.input_utterances, self.target_utterances = get_utterance_pairs()

        # Create dictionaries which map tokens from input and output
        # utterances to an unique number, and vice versa.
        self.input_mapper = TokenMapper(self.input_utterances)
        self.target_mapper = TokenMapper(self.target_utterances)

        # The longest utterances which occur in the data.
        self.max_encoder_seq_length = max(len(utterance) for utterance in self.input_utterances)
        self.max_decoder_seq_length = max(len(utterance) for utterance in self.target_utterances)

        # The number of different tokens in the data.
        self.num_encoder_tokens = len(self.input_mapper.tok2num)
        self.num_decoder_tokens = len(self.target_mapper.tok2num)

        self.encoder_input_data = np.zeros((len(self.input_utterances), self.max_encoder_seq_length), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

        for i, (input_utterance, target_utterance) in enumerate(zip(self.input_utterances, self.target_utterances)):
            for j, token in enumerate(input_utterance):
                # One-hot encoding for which tokens appear in the encoder input data.
                self.encoder_input_data[i][j] = self.input_mapper.tok2num[token]

            for k, token in enumerate(target_utterance):
                # One-hot encoding for decoder input data.
                self.decoder_input_data[i][j] = self.input_mapper.tok2num[token]

                if k > 0:
                    # One-hot encoding for decoder targets. This is the same
                    # as the decoder input data, but the index at which we start
                    # adding tokens from is shifted to the right by one so that
                    # the special token marking the start of an utterance is
                    # no longer included.
                    self.decoder_target_data[i, k-1, self.target_mapper.tok2num[token]] = 1

    def __build_model(self):
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=self.num_encoder_tokens, output_dim=LATENT_DIM)(encoder_inputs)
        encoder_LSTM = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=self.num_decoder_tokens, output_dim=LATENT_DIM)(decoder_inputs)
        decoder_LSTM = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs,_,_ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # Used for encoding incoming sentences when predicting output
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(LATENT_DIM,)), Input(shape=(LATENT_DIM,))]
        decoder_outputs, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        # Used for decoding incoming sentences when predicting output
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def __train(self):
        if not os.path.isfile(MODEL_PATH):
            # TODO: "Note that `decoder_target_data` needs to be one-hot encoded,
            # rather than sequences of integers like `decoder_input_data`!"
            self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                      batch_size=BATCN_SIZE,
                      epochs=NUM_EPOCHS,
                      validation_split=0.2)
            self.model.save_weights(MODEL_PATH)
        else: 
            self.model.load_weights(MODEL_PATH)

    def print_model(self):
        self.model.summary()

    def reply(self):
        pass
