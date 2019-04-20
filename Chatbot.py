from keras.models import Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
import numpy as np
import os
from collections import defaultdict

from data import (get_utterance_pairs, TokenMapper, pad_tokens, tokenize, wrap_utterance, START_UTTERANCE,
                  END_UTTERANCE, UNKNOWN_TOKEN, PAD_TOKEN, filter_unknown, MAX_NUM_TOKENS, USE_CORNELL_CORPUS,
                  get_cornell_utterance_pairs)


LATENT_DIM = 256
BATCN_SIZE = 128
NUM_EPOCHS = 500
MODEL_PATH = 'models/model.h5'


class Chatbot():
    def __init__(self):
        ''' Configure the chatbot. '''
        self.__get_training_data()
        self.__build_model()
        self.__train()

    def __get_training_data(self):
        ''' Load data and prepare training matrices. '''
        # Pair utterances as inputs and outputs.
        if USE_CORNELL_CORPUS:
            # Load English data.
            self.input_utterances, self.target_utterances = get_cornell_utterance_pairs()
        else:
            # Load Norwegian data.
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
        self.target_utterances = [pad_tokens(tokens, self.max_decoder_seq_length) for tokens in self.target_utterances]

        self.max_encoder_seq_length = max(len(utterance) for utterance in self.input_utterances)
        self.max_decoder_seq_length = max(len(utterance) for utterance in self.target_utterances)

        # The number of different tokens in the data.
        self.num_encoder_tokens = len(self.input_mapper.tok2num)
        self.num_decoder_tokens = len(self.target_mapper.tok2num)

        # Configure input matrices. Input data consists of vectors containg token codes, while
        # the decoder target data is one-hot encoded.
        self.encoder_input_data = np.zeros((len(self.input_utterances), self.max_encoder_seq_length), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

        for i, (input_utterance, target_utterance) in enumerate(zip(self.input_utterances, self.target_utterances)):            
            for j, token in enumerate(input_utterance):
                # Encoding for which tokens appear in the encoder input data.
                self.encoder_input_data[i][j] = self.input_mapper.tok2num[token]

            for k, token in enumerate(target_utterance):
                # Encoding for decoder input data.
                self.decoder_input_data[i][k] = self.target_mapper.tok2num[token]

                if k > 0:
                    # One-hot encoding for decoder targets. This is the same as the decoder
                    # input data, but the index at which we start adding tokens from is
                    # shifted to the right by one so that the special token marking the
                    # start of an utterance is not included.
                    self.decoder_target_data[i, k - 1, self.target_mapper.tok2num[token]] = 1


    def __build_model(self):
        ''' Construct the model used to train the chatbot. '''
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=self.num_encoder_tokens, output_dim=LATENT_DIM)(encoder_inputs)
        encoder_LSTM = CuDNNLSTM(LATENT_DIM, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=self.num_decoder_tokens, output_dim=LATENT_DIM)(decoder_inputs)
        decoder_LSTM = CuDNNLSTM(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _,_ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
 
        # Used for encoding incoming sentences when predicting output.
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(LATENT_DIM,)), Input(shape=(LATENT_DIM,))]
        decoder_outputs, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=decoder_state_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states = [state_h, state_c]

        # Used for decoding incoming sentences when predicting output.
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)


    def __train(self):
        ''' Train the model, or load an existing model. '''
        # Load existing model and exit early.
        if os.path.isfile(MODEL_PATH):
            self.target_mapper.tok2num = np.load('models/target_mapper_tok2num.npy').item()
            self.target_mapper.num2tok = np.load('models/target_mapper_num2tok.npy').item()
            self.input_mapper.tok2num = np.load('models/input_mapper_tok2num.npy').item()
            self.input_mapper.num2tok = np.load('models/input_mapper_num2tok.npy').item()

            return self.model.load_weights(MODEL_PATH)

        # Save token mappers.
        np.save('models/target_mapper_tok2num.npy', self.target_mapper.tok2num)
        np.save('models/target_mapper_num2tok.npy', self.target_mapper.num2tok)
        np.save('models/input_mapper_tok2num.npy', self.input_mapper.tok2num)
        np.save('models/input_mapper_num2tok.npy', self.input_mapper.num2tok)

        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                        batch_size=BATCN_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.2)

        self.model.save_weights(MODEL_PATH)


    def print_model(self):
        ''' Print a summary of the model. '''
        self.model.summary()


    def reply(self, input_query):
        ''' Generate a reply using sampling. '''
        # Perform preprocessing on input.
        tokens = pad_tokens(wrap_utterance(tokenize(input_query)[:MAX_NUM_TOKENS]), self.max_encoder_seq_length)

        # Map each token to a number.
        input_sequence = [self.target_mapper.tok2num[token] for token in tokens]

        # Get decoder inputs/encoder outputs
        states = self.encoder_model.predict([input_sequence])

        # Setup decoder inputs
        target_sequence = np.zeros((1, 1))

        # Set first character of target sequence to the start token.
        target_sequence[0, 0] = self.target_mapper.tok2num[START_UTTERANCE]

        output = []

        while True:
            # Predict output.
            output_tokens, state_h, state_c = self.decoder_model.predict([target_sequence] + states)
            token_idx = np.argmax(output_tokens[0, -1, :])
            word = self.target_mapper.num2tok[token_idx]

            if word == END_UTTERANCE or len(output) >= self.max_decoder_seq_length:
                break

            if word != START_UTTERANCE and word != END_UTTERANCE:
                output.append(word)

            target_sequence = np.zeros((1, 1))

            target_sequence[0, 0] = token_idx

            states = [state_h, state_c]

        return ' '.join(output)
