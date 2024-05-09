import librosa

from ddc_onset import FRAME_RATE, compute_onset_salience, find_peaks

# audio is (5864816,) in [-1, 1]
audio, sr = librosa.load(librosa.example('fishin'), sr=44100, mono=True)
# onset salience is (13301,) in [0, 1]
onset_salience = compute_onset_salience(audio, sr)
onset_times = [frame / FRAME_RATE for frame in find_peaks(onset_salience)]