import speech_recognition as sr
from pydub import AudioSegment
import os
from pydub.silence import split_on_silence
import pocketsphinx
from jiwer import wer
import time
from datetime import datetime




# 1db  = -5dbfs
def db_2_dbfs(db):
    return db * -5

def GetCurrentDatetime():
    now = datetime.now()
    return ('%s_%s%s_%s%s%s' % (now.year, now.month, now.day, now.hour, now.minute, now.second))

def audio2question(path, save_path, min_silence_len, threshold, min_duration, seek_step=1, time_it=False,
                  sr=None):
    """

    :param path: Path of Audio Folder
    :param save_path: Path to save
    :param min_silence_len: Minimum length of Silence
    :param threshold: threshold (in DB)
    :param min_duration: Minimum Duration of a sliced chunk
    (if a sliced chunk is shorter than predefined minimum duration, concat with the following chunk.)
    :param seek_step:
    :param time_it: True -> print processing time
    :param sr: sampling rate
    :return: None
    """

    print('<Pydub>')
    start_load = time.time()
    base_dir = os.getcwd()
    if sr != None:
        song = AudioSegment.from_mp3(path).set_frame_rate(sr)
    else:
        song = AudioSegment.from_mp3(path)

    print('Audio Loading Time: ', time.time() - start_load)
    print('Audio Length: ', song.duration_seconds)
    print('Loaded Sampling Rate', song.frame_rate)


    start = time.time()
    chunks = split_on_silence(song,
                              min_silence_len=min_silence_len,
                              silence_thresh=threshold,
                              seek_step=seek_step)
    print('Audio Segmentation Time: ', time.time() - start)
    # print('Total Processing Time: ', time.time() - start_load)
    # print(len(chunks), ' chunks detected')
    # os.mkdir(save_path)
    try:
        os.makedirs(save_path)
    except(FileExistsError):
        pass

    os.chdir(save_path)

    i = 0
    # process each chunk
    for chunk in chunks:
        if len(chunk) <= min_duration:
            pass
        else:
            print("saving chunk{0}.wav".format(i))
            chunk.export("./chunk{0}.wav".format(i), bitrate='22.5k', format="wav")
            i += 1

    if time_it:
        print('time:',time.time() - start_load)

    os.chdir(base_dir)


def inference_audio(path, api_type='google'):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    ## wav 파일 읽어오기
    audio = sr.AudioFile(path)

    with audio as source:
        audio = recognizer.record(source)
    if api_type == 'google':
        text = recognizer.recognize_google(audio_data=audio, language="en-US")
        # text = recognizer.recognize_google(audio_data=audio, language="en-US", enable_automatic_punctuation=True)
    elif api_type == 'sphinx':
        text = recognizer.recognize_sphinx(audio_data=audio, language="en-US")


    print(text)
    return text


if __name__ == '__main__':
    # 문제 분리
    path = '/root/data/TOEIC_Audio/Toeic_3/'
    tests = os.listdir(path)

    for test in tests:
        #Question Segmentation BEST hparams
        test_file_path = os.path.join(path, test)
        test = test.replace('.mp3','')
        print(test)
        audio2question(path=test_file_path, save_path='./Toeic_3/'+test+'/'+GetCurrentDatetime()+'/', min_silence_len=4500, min_duration=10000, threshold=-50, time_it=True, seek_step=150, sr=22050)


    # Sentence Segmentation BEST hparams
    # audio2question(path=path,save_path='./sent_chunks', min_silence_len=700, min_duration=1, threshold=-50, time_it=True, seek_step=5)


    # with open('./ground_truth/gr.txt', 'r') as f:
    #     ground_truth = f.read().strip()
    # print('Infer_text')
    # print()
    # print(infer_text)
    #
    # error = wer(ground_truth, infer_text)
    # print('WER')
    # print(error)
