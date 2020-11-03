import librosa
import librosa.display
import soundfile as sf
import speech_recognition
from pydub.silence import split_on_silence
import numpy as np
import os
import time
import multiprocessing
from pydub import AudioSegment
from sr_test import GetCurrentDatetime
from hparams import Question_Segmentation, Sentence_Segmentation
import io
import os
import parmap
from mfa import find_all_first_word_location
from google.cloud import speech_v1


class ToeicAudioProcessor:
    def __init__(self):
        pass

    def load_question(self, file_path, sr=None, time_it=False, res_type='kaiser_fast'):
        if time_it == True:
            start = time.time()
            y, sr = librosa.load(file_path, sr=sr, res_type=res_type)
            print('question loading time: ', time.time() - start)
        else:
            y, sr = librosa.load(file_path, sr=sr, res_type=res_type)

        # The length of each question must not over 5 minutes.
        assert len(y) / sr <= 300

        return y, sr

    def load_full_test(self, path, sr=None):
        if sr != None:
            audio = AudioSegment.from_mp3(path).set_frame_rate(sr)
        else:
            audio = AudioSegment.from_mp3(path)

        # The length of each question must over 50 minutes.
        assert audio.duration_seconds > 300

        return audio

    def question2sentence(self, y, top_db=60, time_it=False):
        start = time.time()
        nonMuteSections = librosa.effects.split(y, top_db=top_db)
        if time_it:
            print('Audio Segmentation Time: ', time.time() - start)
        return nonMuteSections

    def save_audio(self, audio, path, sample_rate=22050):
        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        sf.write(path, audio.astype(np.int16), samplerate=sample_rate)
        print(" [*] Audio saved: {}".format(path))

    def save_audio_files(self, y, save_folder, nonMuteSections, filename_pattern='sent', min_concat_seconds=1.5, padding = 100,
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

    def audio2question(self, audio, save_path, min_silence_len, threshold, min_duration, seek_step=1, time_it=False):
        """

        :param path: Path of Audio Folder
        :param save_path: Path to save
        :param min_silence_len: Minimum length of Silence
        :param threshold: threshold (in DB)
        :param min_duration: Minimum Duration of a sliced chunk
        (if a sliced chunk is shorter than predefined minimum duration, concat with the following chunk.)
        :param seek_step:
        :param time_it: True -> print processing time
        :return: None
        """
        base_dir = os.getcwd()
        start = time.time()
        chunks = split_on_silence(audio,
                                  min_silence_len=min_silence_len,
                                  silence_thresh=threshold,
                                  seek_step=seek_step)
        print('Audio Segmentation Time: ', time.time() - start)

        try:
            os.makedirs(save_path)
        except(FileExistsError):
            pass

        os.chdir(save_path)

        i = 0
        # process each chunk
        output_ranges = []
        for output_range, chunk in chunks:
            output_ranges.append(output_range)
            if len(chunk) <= min_duration:
                pass
            else:
                print("saving chunk{0}.wav".format(i))
                chunk.export("./chunk{0}.wav".format(i), bitrate='22.5k', format="wav")
                i += 1

        if time_it:
            print('audio2question time:', time.time() - start)

        os.chdir(base_dir)

        return output_ranges, save_path


    def inference_splitted_audios(self, splitted_audio):
        print(splitted_audio)

        for audio in splitted_audio:
            text = self.inference_audio(audio)
            with open(audio.replace('.wav','.txt'), 'w', encoding='utf-8-sig') as f:
                f.write(text)
                print('[*] inferred text for {} saving..'.format(audio))



    def inference_all_questions(self, question_folder, cpu_num=1):
        base_dir = os.getcwd()

        os.chdir(question_folder)
        print('numcores: ', cpu_num)
        audios = os.listdir()
        audios = [audio for audio in audios if audio.endswith('.wav')]


        if cpu_num > 1:
            audios_for_one_process = len(audios) // cpu_num

            if audios_for_one_process < 1:
                audios_for_one_process = 1
            print('audios count: ', len(audios))
            print('audio_for_one_proces: ',audios_for_one_process)
            tmp = []
            splitted_audio = []

            for audio in audios:
                if audios_for_one_process == len(tmp):
                    splitted_audio.append(tmp)
                    tmp = []
                    tmp.append(audio)
                elif audio == audios[-1]:
                    # 마지막까지 갔을 때..
                    splitted_audio.append(tmp)
                else:
                    tmp.append(audio)
        else:
            splitted_audio = audios

        print('splitted audios: {}'.format(len(splitted_audio)))


        if cpu_num > 1:
            result = parmap.map(self.inference_splitted_audios, splitted_audio, pm_pbar=True, pm_processes=cpu_num)
        else:
            result = parmap.map(self.inference_audio, splitted_audio, pm_pbar=True, pm_processes=cpu_num)

        os.chdir(base_dir)

        return question_folder

    def mfa(self, sample_path, save_path, lexicon='/root/data/sba/CWK/MFA/montreal-forced-aligner/librispeech-lexicon.txt'):
        result = os.popen(
            '/root/data/sba/CWK/MFA/montreal-forced-aligner/bin/mfa_align ' + sample_path + ' ' + lexicon + ' ' + 'english' + ' ' + save_path).read()
        print(result)

    def inference_audio(self, path, api_type='google'):
        recognizer = speech_recognition.Recognizer()
        recognizer.energy_threshold = 300

        print(path)

        ## wav 파일 읽어오기
        audio = speech_recognition.AudioFile(path)

        with audio as source:
            audio = recognizer.record(source)

        if api_type == 'google':
            text = recognizer.recognize_google(audio_data=audio, language="en-US")
            # text = recognizer.recognize_google_cloud(audio_data=audio, language="en-US", enable_automatic_punctuation=True)
        elif api_type == 'sphinx':
            text = recognizer.recognize_sphinx(audio_data=audio, language="en-US")
        elif api_type == 'google_stt':
            text = self.google_STT(audio=path)

        return text

    def google_STT(self,audio):
        client = speech_v1.SpeechClient.from_service_account_json(
            '/data/second-conquest-293723-05738e995f8f.json')

        # Loads the audio into memory
        with io.open(audio, "rb") as audio_file:
            content = audio_file.read()
            audio = speech_v1.RecognitionAudio(content=content)

        encoding = speech_v1.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

        config = speech_v1.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=22050,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        # Detects speech in the audio file
        start = time.time()
        response = client.recognize(request={"config": config, "audio": audio})
        text = ''
        for result in response.results:
            text = text + result.alternatives[0].transcript
            # print("Transcript: {}".format())
        return text

if __name__ == '__main__':

    ### Question2Sentence Example ###

    # instantiate ToeicAudioProcessor instance
    # toeic = ToeicAudioProcessor()
    #
    # # load question audio file
    # path = '/data/sba/HBJ/sr_test/Toeic_3/TEST_1/2020_1025_171741/chunk14.wav'
    # y, sr = toeic.load_question(path, res_type='kaiser_fast', time_it=True)
    #
    # # split question to sentence on predefined threshold
    # nonMuteSections = toeic.question2sentence(y, top_db=Sentence_Segmentation.top_db, time_it=True)
    #
    # # save splitted sentence audio file of one question into save_folder
    # toeic.save_audio_files(y, save_folder='./wavs/', nonMuteSections=nonMuteSections,
    #                        min_concat_seconds=0.1)


    ### Audio2question Example ###

    # # instantiate ToeicAudioProcessor instance
    # toeic = ToeicAudioProcessor()
    # # cpu_num = multiprocessing.cpu_count()
    # cpu_num = 16
    # # multiprocessing
    # start = time.time()
    # toeic.inference_all_questions('/data/sba/HBJ/sr_test/Toeic_3/TEST_1/2020_1026_153731', cpu_num=cpu_num)
    #
    # print('cpu_num: ', cpu_num)
    # print('inference time for one full test: {}'.format(str(time.time()-start)))
    #
    #
    #
    # print(toeic[100000000])


    ### Full TEST ###

    # load full test audio file
    start = time.time()
    toeic = ToeicAudioProcessor()
    path = '/root/data/TOEIC_Audio/Toeic_2/TEST_2.mp3'
    audio = toeic.load_full_test(path, sr=22050)
    # print(len(audio), audio[1])

    # split audio into question and save sentences
    test = path.split('/')[-1].replace('.mp3', '') # TEST_1

    # audio2Question
    output_ranges, question_save_path = toeic.audio2question(audio, save_path='./Toeic_2/' + test + '/' + GetCurrentDatetime() + '/',
                         min_silence_len=Question_Segmentation.min_silence_len,
                         min_duration=Question_Segmentation.min_duration,
                         threshold=Question_Segmentation.threshold,
                         time_it=True, seek_step=Question_Segmentation.seek_step)


    ms = 1000
    i = 1
    print(output_ranges)
    for output in output_ranges:
        dur = output[1] - output[0]
        q_start = output[0] / ms
        q_end = output[1] / ms
        # print(dur)
        if dur > 10000:
            print('question {} duration: {}~{}'.format(i,q_start, q_end))
            i += 1



    print('audio2question finished...')

    # Inference Script of each Question w/ multiprocessing
    cpu_num = multiprocessing.cpu_count()
    question_script_folder = toeic.inference_all_questions(question_save_path, cpu_num=int(cpu_num * 6))


    question_script_folder = os.path.abspath(question_script_folder)
    print(question_script_folder)

    print('full audio inferencing finished..')

    # MFA
    save_path = '/data/sba/HBJ/sr_test/MFA_result'
    toeic.mfa(question_script_folder, save_path)

    print('MFA finished')
    print('aligmenting with script and Textgrid..')

    # get last word location based on the result of mfa
    text_path = question_script_folder
    text_grid_path = '/data/sba/HBJ/sr_test/MFA_result'
    all_questions_with_sentence_start_time = find_all_first_word_location(text_path, text_grid_path)
    print(all_questions_with_sentence_start_time)

    print('finished!!')
    print('processing time: ', time.time() - start)




    ### Inference Example ###

    # instantiate ToeicAudioProcessor instance
    # toeic = ToeicAudioProcessor()
    # path = '/data/sba/HBJ/sr_test/Toeic_3/TEST_1/2020_1025_171741/chunk32.wav'
    # inferred_text = toeic.inference_audio(path, api_type='google')
    # print(inferred_text)
