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
from nltk.tokenize import sent_tokenize
import re


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

        # The length of each question must over 5 minutes.
        assert audio.duration_seconds > 300

        return audio

    def load_chunk_before(self, path, sr=None):
        if sr != None:
            audio = AudioSegment.from_mp3(path).set_frame_rate(sr)
        else:
            audio = AudioSegment.from_mp3(path)

        print(len(audio))

        if len(audio) > Question_Segmentation.max_duration:
            trim = True
        else: 
            trim = False

        return trim, audio

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
        output_ranges, chunks = split_on_silence(audio,
                                  min_silence_len=min_silence_len,
                                  silence_thresh=threshold,
                                  seek_step=seek_step)
        print('Audio Segmentation Time: ', time.time() - start)
        print(output_ranges)
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
            print('audio2question time:', time.time() - start)

        os.chdir(base_dir)

        return output_ranges, save_path


    def inference_splitted_audios(self, splitted_audio):
        print(splitted_audio)

        for audio in splitted_audio:
            text = self.inference_audio(audio)
            with open(audio.replace('.wav','.txt'), 'w') as f:
                f.write(text)
                print('[*] inferred text for {} saving..'.format(audio))



    def inference_all_questions(self, question_folder, cpu_num=1):
        base_dir = os.getcwd()

        os.chdir(question_folder)
        print('numcores: ', cpu_num)
        audios = os.listdir()
        audios = [audio for audio in audios if audio.endswith('.wav')]


        if cpu_num > 1:
            audios_for_one_process = len(audios) // cpu_num + 2
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
                elif audio == audios[-1]:
                    # 마지막까지 갔을 때..
                    splitted_audio.append(tmp)
                else:
                    tmp.append(audio)
        else:
            splitted_audio = audios

        print('splitted audios: {}'.format(splitted_audio))


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

    def inference_audio(self, path, api_type='google_stt'):
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
            text, sent_start_time = self.google_STT(path=path)

        return text, sent_start_time

    def google_STT(self, path):
        client = speech_v1.SpeechClient.from_service_account_json(
            '/data/second-conquest-293723-05738e995f8f.json')

        # Loads the audio into memory
        with io.open(path, "rb") as audio_file:
            content = audio_file.read()
            audio = speech_v1.RecognitionAudio(content=content)

        encoding = speech_v1.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

        config = speech_v1.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=22050,
            language_code="en-US",
            audio_channel_count=2 if path.endswith('.wav') else 1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        )

        # Detects speech in the audio file
        start = time.time()
        response = client.recognize(request={"config": config, "audio": audio})

        sent = ''
        text = []
        end_word = []
        sent_start_time = [0, ]

        for result in response.results:
            alternative = result.alternatives[0]
            #print("Transcript: {}".format(alternative.transcript))
            #print("Confidence: {}".format(alternative.confidence))
            print(alternative.transcript)

            if re.search('Number to', alternative.transcript) != None:
                print('find!!!')
                re.sub('Number to ', 'Number 2.\n', alternative.transcript)
            #     print(alternative.transcript)
            #     re.sub('Number to ', 'Number 2.\n', alternative.transcript)
            #     # re.sub('Number one', 'Number 1.\n', alternative.transcript)
            #     print(alternative.transcript)
            # re.sub('Number to l', 'Number 2.\n L', alternative.transcript)
            # re.sub('Number for l', 'Number 4.\n L', alternative.transcript)
            # re.sub('in your test book a', 'in your test book.\n A.', alternative.transcript)
            

            # seperate_sentence = sent_tokenize(alternative.transcript)
            # for i in range(len(seperate_sentence)):
            #     end_word.append(re.split(" ",text)[-1])

            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time

                sent = sent + ' ' + word

                if (re.search('\\.', word) != None) | (re.search('\\?', word) != None):
                    sent_start_time.append(end_time.total_seconds())
                    text.append(sent)
                    sent = ''

        for raw_text in text:
            if re.search('Number to', raw_text) != None:
                print('find!!!')
                re.sub('Number to ', 'Number 2.\n', raw_text)
            re.sub('Number one', 'Number 1.\n', raw_text)
            re.sub('Number to ', 'Number 2.\n', raw_text)
            re.sub('Number for l', 'Number 4.\n L', raw_text)
            re.sub('in your test book a', 'in your test book.\n A.', raw_text)
            print(raw_text)


        return text, sent_start_time[:-1]
# Return 결과값
# [0, 5.3, 8.1, 13.7, 22.8, 32.4, 38.0, 43.6, 50.8]
# [' Question 71 through 73 refer to the following telephone message.', " Marion it's Haley.", ' One of my friends invited me to go to the front and music festival with him on Saturday.', ' I said I would go like you to come along to the festival is going to be held up call from Park from noon until 10 p.m.', " We plan to go at 5:30 because our favorite then takes the stage at 6:30 and we'd like to get an hour earlier to find good seats.", " Please call or text me whenever you get this message to let me know if you'll be joining us.", ' I want to buy tickets this afternoon since today is the last day to buy them at reduced prices.', ' Number 71, what did the speaker agree to do?']


# toeic = ToeicAudioProcessor()
# rawdata_path = '/data/sba/HBJ/sr_test/Toeic_4'
# chunk_path = os.path.join(rawdata_path, "chunk6.wav")
# inferred_text = toeic.inference_audio(chunk_path, api_type='google_stt')

if __name__ == '__main__':

    # instantiate ToeicAudioProcessor instance
    toeic = ToeicAudioProcessor()

    ### Seperate 
    # raw_path = '/data/sba/HBJ/sr_test/Toeic_2/TEST_2/before' # 4.5sec로 문제 구분한 chunk path
    # out_path = '/data/sba/HBJ/sr_test/Toeic_2/TEST_2/after2' # raw chunk를 1분 이하의 음원으로만 자른 path
    # _, _, files = next(os.walk(raw_path))
    # chunk_num = len(files)
    # j = 0
    # for i in range(chunk_num):
    #     chunk_path = os.path.join(raw_path, "chunk{0}.wav".format(i))
    #     trim, audio30 = toeic.load_chunk_before(chunk_path, sr=22050)

    #     if trim:
    #         print('!!!!!!chunk{0}.wav need to trim!!!!!!'.format(i))
    #         chunk_before = split_on_silence(audio30,
    #                         min_silence_len=1000,
    #                         silence_thresh=Question_Segmentation.threshold,
    #                         seek_step=Question_Segmentation.seek_step)

    #         for out, chunk in chunk_before:
    #             print("saving chunk{0}_".format(i) + "{0}.wav".format(j))
    #             print(j, len(chunk))
    #             chunk.export("/data/sba/HBJ/sr_test/Toeic_2/TEST_2/after2/chunk{0}.wav".format(j), bitrate='22.5k', format="wav")
    #             j += 1

    #     else:
    #         print("saving chunk{0}.wav".format(i))
    #         audio30.export("/data/sba/HBJ/sr_test/Toeic_2/TEST_2/after2/chunk{0}.wav".format(j), bitrate='22.5k', format="wav")
    #         j += 1

    out_path = '/data/sba/HBJ/sr_test/Toeic_2/TEST_2/after2' # raw chunk를 1분 이하의 음원으로만 자른 path
    _, _, files = next(os.walk(out_path))
    chunk_num = len(files)

    for i in range(10,chunk_num):
        chunk_path = os.path.join(out_path, "chunk{0}.wav".format(i))
        inferred_text, sent_start_time = toeic.inference_audio(chunk_path, api_type='google_stt')

        with open(out_path + "chunk{0}.txt".format(i), 'w', encoding='utf-8-sig') as f:
                for sentence in inferred_text:
                    f.write(sentence+'\n')
        print(inferred_text, sent_start_time)

