# model.py
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

def crear_modelo(vocab_size, maxlen):
    # Modelo de codificador
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(vocab_size, 256)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Modelo de decodificador
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(vocab_size, 256)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Modelo final
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
