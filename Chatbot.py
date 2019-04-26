import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Input, TimeDistributed, Dropout, Concatenate
from collections import defaultdict
from attention_keras.layers.attention import AttentionLayer
import numpy as np
import os

from data import (get_utterance_pairs, TokenMapper, pad_tokens, tokenize, wrap_utterance, START_UTTERANCE,
                  END_UTTERANCE, UNKNOWN_TOKEN, PAD_TOKEN, filter_unknown, MAX_NUM_TOKENS, USE_CORNELL_CORPUS,
                  get_cornell_utterance_pairs)


LATENT_DIM = 256
BATCN_SIZE = 64
NUM_EPOCHS = 200
DROPOUT_RATE = 0.2

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

        # The number of different tokens in the data.
        self.num_encoder_tokens = len(self.input_mapper.tok2num)
        self.num_decoder_tokens = len(self.target_mapper.tok2num)

        # Configure input matrices. Input data consists of vectors containg token codes, while
        # the decoder target data is one-hot encoded.
        self.encoder_input_data = np.zeros((len(self.input_utterances), self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.target_utterances), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

        for i, (input_utterance, target_utterance) in enumerate(zip(self.input_utterances, self.target_utterances)):
            for t, token in enumerate(input_utterance):
                # One-hot encoding for encoder input data.
                self.encoder_input_data[i, t, self.input_mapper.tok2num[token]] = 1

            for t, token in enumerate(target_utterance):
                # One-hot encoding for decoder input data.
                self.decoder_input_data[i, t, self.target_mapper.tok2num[token]] = 1

                if t > 0:
                    # One-hot encoding for decoder targets. This is the same as the decoder
                    # input data, but the index at which we start adding tokens from is
                    # shifted to the right by one so that the special token marking the
                    # start of an utterance is not included.
                    self.decoder_target_data[i, t - 1, self.target_mapper.tok2num[token]] = 1


    def __build_model(self):
        ''' Construct the model used to train the chatbot. '''
        encoder_inputs = Input(shape=(self.max_encoder_seq_length, self.num_encoder_tokens), name='encoder_input')
        encoder_dropout = (TimeDistributed(Dropout(rate=DROPOUT_RATE, name='encoder_dropout')))(encoder_inputs)
        encoder = GRU(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_gru')

        encoder_outputs, encoder_state = encoder(encoder_dropout)

        decoder_inputs = Input(shape=(self.max_decoder_seq_length, self.num_decoder_tokens), name='decoder_input')
        decoder_dropout = (TimeDistributed(Dropout(rate=DROPOUT_RATE, name='decoder_dropout')))(decoder_inputs)

        decoder_gru = GRU(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_dropout, initial_state=encoder_state)

        # Attention mechanism
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        decoder_outputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_activation_softmax')
        decoder_outputs = TimeDistributed(decoder_dense, name='time_distributed_layer')(decoder_outputs)

        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.summary()

        self.encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, encoder_state])

        decoder_inf_inputs = Input(batch_shape=(None, 1, self.num_decoder_tokens), name='decoder_word_inputs')
        encoder_inf_states = Input(batch_shape=(None, self.max_encoder_seq_length, LATENT_DIM), name='encoder_inf_states')
        decoder_init_state = Input(batch_shape=(None, LATENT_DIM), name='decoder_init')

        decoder_outputs, decoder_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
        attn_out, attn_states = attn_layer([encoder_inf_states, decoder_outputs])
        decoder_outputs = Concatenate(axis=-1, name='concat')([decoder_outputs, attn_out])
        decoder_outputs = TimeDistributed(decoder_dense)(decoder_outputs)

        self.decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs], 
                                   outputs=[decoder_outputs, attn_states, decoder_state])


    def __train(self):
        ''' Train the model, or load an existing model. '''
        # Load existing model and exit early.
        if os.path.isfile(MODEL_PATH):
            self.target_mapper.tok2num = np.load('models/target_mapper_tok2num.npy').item()
            self.target_mapper.num2tok = np.load('models/target_mapper_num2tok.npy').item()
            self.input_mapper.tok2num = np.load('models/input_mapper_tok2num.npy').item()
            self.input_mapper.num2tok = np.load('models/input_mapper_num2tok.npy').item()

            self.model.load_weights(MODEL_PATH)

            if 'INITIAL_EPOCH' not in os.environ:
                return

        # Save token mappers.
        np.save('models/target_mapper_tok2num.npy', self.target_mapper.tok2num)
        np.save('models/target_mapper_num2tok.npy', self.target_mapper.num2tok)
        np.save('models/input_mapper_tok2num.npy', self.input_mapper.tok2num)
        np.save('models/input_mapper_num2tok.npy', self.input_mapper.num2tok)

        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                        batch_size=BATCN_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.2,
                        initial_epoch=int(os.getenv('INITIAL_EPOCH', 1))
                        )

        self.model.save_weights(MODEL_PATH)


    def print_model(self):
        ''' Print a summary of the model. '''
        self.model.summary()


    def reply(self, query):
        ''' Generate a reply using sampling. '''
        # Perform preprocessing on input.
        if not isinstance(query, list):
            query = pad_tokens(wrap_utterance(tokenize(query.lower())[:MAX_NUM_TOKENS]), self.max_encoder_seq_length)

        input_sequence = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')

        for i, token in enumerate(query):
            input_sequence[0, i, self.target_mapper.tok2num[token]] = 1

        # Get decoder inputs/encoder outputs
        enc_out, enc_state = self.encoder_model.predict(input_sequence)
        dec_state = enc_state

        # Setup decoder inputs
        target_sequence = np.zeros((1, 1, self.num_decoder_tokens))

        # Set first character of target sequence to the start token.
        target_sequence[0, 0, self.target_mapper.tok2num[START_UTTERANCE]] = 1

        output = []

        while True:
            # Predict output.
            output_tokens, attention, dec_state = self.decoder_model.predict([enc_out, dec_state, target_sequence])
            sample = np.argmax(output_tokens[0, -1, :])
            word = self.target_mapper.num2tok[sample]

            if word == END_UTTERANCE or len(output) >= self.max_decoder_seq_length:
                break

            if word != START_UTTERANCE and word != END_UTTERANCE and word != PAD_TOKEN:
                output.append(word)

            target_sequence = np.zeros((1, 1, self.num_decoder_tokens))

            target_sequence[0, 0, sample] = 1

        return ' '.join(output)

    
    def test_replies(self):
        ''' Show responses to a portion of the training set. '''
        for utterance in self.input_utterances[:200]:
            print(utterance, '=>', self.reply(utterance))
