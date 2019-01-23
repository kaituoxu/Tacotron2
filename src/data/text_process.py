import hyperparams as hp

char_to_id = {char: i for i, char in enumerate(hp.chars)}
id_to_char = {i: char for i, char in enumerate(hp.chars)}


def text_to_sequence(text, eos=hp.eos):
    text += eos
    return [char_to_id.get(char, hp.unk_idx) for char in text]


def sequence_to_text(sequence):
    return "".join(id_to_char.get(i, '<unk>') for i in sequence)
