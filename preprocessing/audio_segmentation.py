import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import os
import time
from os import path
# from pydub import AudioSegment

class ToeicAudioProcessor:
    def __init__(self):
        pass

    def load(self, file_path, sr=22050, time_it=False):
        if time_it == True:
            start = time.time()
            y, sr = librosa.load(file_path, sr=sr)
            print(time.time() - start)
        else:
            y, sr = librosa.load(file_path, sr=sr)
        return y, sr

    def split_on_silence(self, y, top_db=60):
        nonMuteSections = librosa.effects.split(y, top_db=top_db)
        return nonMuteSections

    def save_audio(self,audio, path, sample_rate=22050):
        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        sf.write(path, audio.astype(np.int16), samplerate=sample_rate)
        print(" [*] Audio saved: {}".format(path))

    def save_audio_files(self, y, save_folder, filename_pattern, nonMuteSections, min_concat_seconds=1.5, padding = 100,
                         sr = 22050):
        os.makedirs(save_folder, exist_ok=True)
        n_concat_signals = sr * min_concat_seconds
        for idx, i in enumerate(nonMuteSections):
            sent_len = i[1] - i[0]
            if sent_len <= n_concat_signals:
                nonMuteSections[idx + 1][0] = i[0]
            else:
                if i[0] != 0:
                    self.save_audio(y[i[0] - padding:i[1] + padding],
                               path=os.path.join(save_folder + filename_pattern + str(idx + 1)) + '.wav', sample_rate=sr)
                else:
                    self.save_audio(y[i[0]:i[1] + padding], path=os.path.join(save_folder + filename_pattern + str(idx + 1)) + '.wav',sample_rate=sr)

    # def mp3_to_wav(self, src, dst):
    #     print(path.exists(src))
    #     # convert wav to mp3
    #     sound = AudioSegment.from_mp3(src)
    #     sound.export(dst, format="wav")





if __name__ == '__main__':

    # instantiate ToeicAudioProcessor instance
    toeic = ToeicAudioProcessor()
    # load wav file
    y, sr = toeic.load('test2.wav')
    # split sounds on predefined threshold
    nonMuteSections = toeic.split_on_silence(y,top_db=70)
    # save splitted files
    toeic.save_audio_files(y, save_folder='./wavs/',filename_pattern='test', nonMuteSections=nonMuteSections,
                           min_concat_seconds=1)
