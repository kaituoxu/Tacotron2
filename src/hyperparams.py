# Text
eos = '~'
pad = '_'
chars = pad + eos + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
unk_idx = len(chars)

# Audio
sample_rate = 22050  # Hz
frame_length_ms = 50  # ms
frame_shift_ms = 12.5  # ms
num_mels = 80  # filters
min_freq = 125  # Hz
max_freq = 7600  # Hz
floor_freq = 0.01  # clip value, prior to log compression
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
max_abs_value = 4
power = 1.5
fft_size = 1024
hop_size = 256

# Encoder
num_chars = len(chars) + 1  # + 1 is <unk>
padding_idx = chars.find(pad)

# Decoder
feature_dim = 513

# Eval:
griffin_lim_iters = 60
