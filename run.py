# Simple way to train a model to create lofi music
# Dataset is too small and it currently does not 
# support pauses

import glob
import pickle
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from music21 import converter, instrument, note, chord, stream

# Run the program
def main():
    # Get initial data
    notes = generate_music()
    n_vocab = len(set(notes))

    ### Train the model, uncomment if model is not trained
    """
    network_input, network_output = generate_sequences(notes, n_vocab)
    model = build_model(network_input, n_vocab)
    train_model(model, network_input, network_output)
    """

    ### Make actual music with the trained model
    # Run the weights obtained in training
    network_input, normalized_input = prepare_sequences(notes, n_vocab)
    model = build_model(normalized_input, n_vocab)
    model.load_weights("./weights/weights.hdf5")
    prediction_output = create_music(model, network_input, notes, n_vocab)
    save_music(prediction_output)


# Create the new music 
def create_music(model, network_input, notes, n_vocab):
    print("-----------------------------")
    print("Creating new music...")
    print("-----------------------------")

    start = np.random.randint(0, len(network_input)-1)

    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(60):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("<----------------------------->")
    print("New music created!")
    print("<----------------------------->")

    return prediction_output

# Save the created music to MIDI
def save_music(predictions):
    print("-----------------------------")
    print("Saving music...")
    print("-----------------------------")
    offset = 0
    output_notes = []

    for pattern in predictions:
        if("." in pattern) or pattern.isdigit():
            chord_notes = pattern.split(".")
            notes = []
            for current_note in chord_notes:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="./result/final_output.midi")
    print("<----------------------------->")
    print("Music saved!")
    print("<----------------------------->")

# Prepare the sequences
def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 50
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape and normalize
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


# Parse music from MIDI to string
def generate_music():
    print("-----------------------------")
    print("Generating music...")
    print("-----------------------------")
    notes = [] 

    for file in glob.glob("./midi/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        # File has instrument parts
        try: 
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        # File has notes in a flat structure
        except: 
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            # Add notes
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            # Add chords (note1.note2)
            elif isinstance(element, chord.Chord):
                nchord = ".".join(str(n) for n in element.normalOrder)
                notes.append(nchord)

        with open("notes.txt", "w") as filepath:
            for n in notes:
                filepath.write(n)
            # pickle.dump(notes,filepath)

    print("<----------------------------->")
    print("Notes generated!")
    print("<----------------------------->")

    return notes

# Tokenize data and batch
def generate_sequences(notes, n_vocab):
    print("-----------------------------")
    print("Generating sequences...")
    print("-----------------------------")
    # Every pitch name
    pitchnames = sorted(set(item for item in notes))

    # Map
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))


    # This probably can be optimized with a batch system
    network_input = []
    network_output = []

    seq_length = 100
    # Input and output sequences
    for i in range(0, len(notes) - seq_length, 1):
        seq_in = notes[i:i+seq_length]
        seq_out = notes[i+seq_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_pattern = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_pattern, seq_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    print("<----------------------------->")
    print("Sequences generated!")
    print("<----------------------------->")

    return (network_input, network_output)

# Generate model
def build_model(network_input, n_vocab):
    model = tf.keras.Sequential([
            tf.keras.layers.LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_vocab),
            tf.keras.layers.Activation("softmax"),
                ])
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model

# Train the model
def train_model(model, network_input, network_output):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


# Run
if __name__ == "__main__":
    main()
