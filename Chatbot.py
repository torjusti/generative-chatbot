from keras.models import Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
import numpy as np
import os
from collections import defaultdict

from data import (get_utterance_pairs, TokenMapper, pad_tokens, tokenize, wrap_utterance, START_UTTERANCE,
END_UTTERANCE, UNKNOWN_TOKEN, PAD_TOKEN, filter_unknown)


LATENT_DIM = 256
BATCN_SIZE = 64
NUM_EPOCHS = 25
DROPOUT_RATE = 0.2
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

        # Filter out unknown tokens from training data for testing purposes.
        self.input_utterances, self.target_utterances = filter_unknown(self.input_utterances, self.target_utterances,
                                                                       self.input_mapper, self.target_mapper)

        # The longest utterances which occur in the data.
        self.max_encoder_seq_length = max(len(utterance) for utterance in self.input_utterances)
        self.max_decoder_seq_length = max(len(utterance) for utterance in self.target_utterances)

        # Pad tokens to the maximum length.
        self.input_utterances = [pad_tokens(tokens, self.max_encoder_seq_length) for tokens in self.input_utterances]
        self.target_utterances = [pad_tokens(tokens, self.max_encoder_seq_length) for tokens in self.target_utterances]

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
                self.decoder_input_data[i][k] = self.target_mapper.tok2num[token]

                if k > 0:
                    # One-hot encoding for decoder targets. This is the same
                    # as the decoder input data, but the index at which we start
                    # adding tokens from is shifted to the right by one so that
                    # the special token marking the start of an utterance is
                    # no longer included.
                    self.decoder_target_data[i, k-1, self.target_mapper.tok2num[token]] = 1


    def __build_model(self):

        # Encoder input layer
        encoder_input = Input(shape=(None,))

        # Encoder hidden layers
        encoder_embedding = Embedding(input_dim=self.num_encoder_tokens, output_dim=LATENT_DIM)(encoder_input)
        encoder_dropout = (TimeDistributed(Dropout(rate = DROPOUT_RATE)))(encoder_embedding)
        encoder_LSTM = LSTM(LATENT_DIM, return_sequences=True)(encoder_dropout)

        # Encoder output layers
        encoder_LSTM2 = LSTM(LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM2(encoder_LSTM)

        # Only need the states
        encoder_states = [state_h, state_c]

        # Decoder input layer
        decoder_input = Input(shape=(None,))

        # Decoder hidden layers
        decoder_embedding_layer = Embedding(input_dim=self.num_decoder_tokens, output_dim=LATENT_DIM)
        decoder_embedding = decoder_embedding_layer(decoder_input)

        decoder_dropout_layer = (TimeDistributed(Dropout(rate=DROPOUT_RATE)))
        decoder_dropout = decoder_dropout_layer(decoder_embedding)


        decoder_LSTM_layer = LSTM(LATENT_DIM, return_sequences=True)
        decoder_LSTM = decoder_LSTM_layer(decoder_dropout, initial_state=encoder_states)

        decoder_LSTM2_layer = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_LSTM2, state_h, state_c = decoder_LSTM2_layer(decoder_LSTM)
        decoder_states = [state_h, state_c]

        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_LSTM2)

        self.model = Model([encoder_input, decoder_input], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # define the encoder model 
        self.encoder_model = Model(encoder_input, encoder_states)

        # Redefine the decoder model with decoder will be getting below inputs from encoder while in prediction
        decoder_state_input_h = Input(shape=(LATENT_DIM,))
        decoder_state_input_c = Input(shape=(LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_LSTM = decoder_LSTM_layer(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_LSTM2, state_h, state_c = decoder_LSTM2_layer(decoder_LSTM)
        decoder_states = [state_h, state_c]
        
        decoder_outputs = decoder_dense(decoder_LSTM2)

        # sampling model will take encoder states and decoder_input(seed initially) and output the predictions(french word index) We dont care about decoder_states2
        self.decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)



    def __train(self):
        if not os.path.isfile(MODEL_PATH):
            # Save token mappers.
            np.save('models/target_mapper_tok2num.npy', self.target_mapper.tok2num)
            np.save('models/target_mapper_num2tok.npy', self.target_mapper.num2tok)
            np.save('models/input_mapper_tok2num.npy', self.input_mapper.tok2num)
            np.save('models/input_mapper_num2tok.npy', self.input_mapper.num2tok)

            # TODO: "Note that `decoder_target_data` needs to be one-hot encoded,
            # rather than sequences of integers like `decoder_input_data`!"
            self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                      batch_size=BATCN_SIZE,
                      epochs=NUM_EPOCHS,
                      validation_split=0.2)
            self.model.save_weights(MODEL_PATH)
        else: 
            # Load token mapper
            self.target_mapper.tok2num = np.load('models/target_mapper_tok2num.npy').item()
            self.target_mapper.num2tok = np.load('models/target_mapper_num2tok.npy').item()
            self.input_mapper.tok2num = np.load('models/input_mapper_tok2num.npy').item()
            self.input_mapper.num2tok = np.load('models/input_mapper_num2tok.npy').item()

            self.model.load_weights(MODEL_PATH)

    def print_model(self):
        self.model.summary()

    def reply(self, input_query):
        tokens = pad_tokens(wrap_utterance(tokenize(input_query)), self.max_encoder_seq_length)

        # Map each token to a number.
        input_sequence = [self.target_mapper.tok2num[token] for token in tokens]

        # Get decoder inputs/encoder outputs
        states = self.encoder_model.predict(input_sequence)

        # Setup decoder inputs
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = self.target_mapper.tok2num[START_UTTERANCE]

        output = []

        while True:
            # Predict output
            output_tokens, state_h, state_c = self.decoder_model.predict([target_sequence] + states)
            token_idx = np.argmax(output_tokens[0, -1, :])
            print(token_idx, self.target_mapper.num2tok[token_idx])

            word = self.target_mapper.num2tok[token_idx]


            if word == END_UTTERANCE or len(output) >= self.max_decoder_seq_length:
                break

            if word != START_UTTERANCE and word != END_UTTERANCE:
                output.append(word)

            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = token_idx

            states = [state_h, state_c]

        return ' '.join(output).replace(UNKNOWN_TOKEN, '')
