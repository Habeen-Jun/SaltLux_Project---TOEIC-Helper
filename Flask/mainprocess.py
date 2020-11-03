import os
from ToeicAudioProcessor import ToeicAudioProcessor
from ResponseVO import Object
from hparams import Question_Segmentation
from utils import GetCurrentDatetime
from time import time
import json
from pydub import AudioSegment

def sort_by_start_time(response):

    # with statement

    sorted_result = sorted(response.result, key=lambda x: x.time, reverse=False)

    response.result = sorted_result
    # res = json.dumps(sorted_result)
    return response



    # response.content('ddd')

    # print(response.toJSON())
def process_toeic_full_test(path):
    response = Object()
    start = time()
    response.result = []

    # instantiate ToeicAudioProcessor instance
    toeic = ToeicAudioProcessor()
    # toeic.google_STT('/data/sba/HBJ/sr_test/Toeic_2/TEST_1/2020_1025_174928/chunk12.wav')
    # print(toeic[10])
    audio = toeic.load_full_test(path, sr=22050)  #
    # print(len(audio), audio[1])
    # split audio into question and save sentences
    test = path.split('/')[-1].replace('.mp3', '')  # TEST_1
    # audio2Question
    output_ranges, question_save_path = toeic.audio2question(audio,
                                                             save_path='./Toeic_3/' + test + '/' + GetCurrentDatetime() + '/',
                                                             min_silence_len=Question_Segmentation.min_silence_len,
                                                             min_duration=Question_Segmentation.min_duration,
                                                             threshold=Question_Segmentation.threshold,
                                                             time_it=True,
                                                             seek_step=Question_Segmentation.seek_step)
    print('output_ranges: ', output_ranges)
    ms = 1000
    # filter output_ranges over 10s.
    output_ranges = [[output[0] / ms, output[1] / ms] for output in output_ranges if
                     (output[1] / ms) - (output[0] / ms) > 10]

    print('output_ranges(secs): ', output_ranges)


    result_save_path = './Flask_Toeic_Result/'
    os.makedirs(result_save_path, exist_ok=True)

    parse_file_path = result_save_path + GetCurrentDatetime() + '.txt'
    with open(parse_file_path, 'w') as f:
        for idx, output in enumerate(output_ranges):
            question = Object()
            question.idx = idx
            question.start_time = output[0]
            response.result.append(question)
            f.write(str(idx) + ' ' + str(output[0]) + '\n')
    print('base_dir: ', os.getcwd())
    question_folder, results = toeic.inference_all_questions(question_save_path, cpu_num=24)

    print('results:', results)
    for pool in results:
        for future in pool:
            question_idx = str(future[0])
            question = response.result[int(question_idx)]
            sentences = future[1]
            sent_start_time = future[2]

            question.sentences = []
            for sentence, start_time in zip(sentences, sent_start_time):
                question.sentences.append((start_time + question.start_time, sentence))

    response.processing_time = time() - start

    return response.toJSON()


def process_toeic_full_test_Arraylist(path):
    response = Object()
    start = time()
    response.result = []

    # instantiate ToeicAudioProcessor instance
    toeic = ToeicAudioProcessor()
    # toeic.google_STT('/data/sba/HBJ/sr_test/Toeic_2/TEST_1/2020_1025_174928/chunk12.wav')
    # print(toeic[10])
    audio = toeic.load_full_test(path, sr=22050)  #
    # print(len(audio), audio[1])
    # split audio into question and save sentences
    test = path.split('/')[-1].replace('.mp3', '')  # TEST_1
    # audio2Question
    output_ranges, question_save_path = toeic.audio2question(audio,
                                                             save_path='./Toeic_3/' + test + '/' + GetCurrentDatetime() + '/',
                                                             min_silence_len=Question_Segmentation.min_silence_len,
                                                             min_duration=Question_Segmentation.min_duration,
                                                             threshold=Question_Segmentation.threshold,
                                                             time_it=True,
                                                             seek_step=Question_Segmentation.seek_step)
    print('output_ranges: ', output_ranges)
    ms = 1000
    # filter output_ranges over 10s.
    output_ranges = [[output[0] / ms, output[1] / ms] for output in output_ranges if
                     (output[1] / ms) - (output[0] / ms) > 10]

    print('output_ranges(secs): ', output_ranges)


    result_save_path = './Flask_Toeic_Result/'
    os.makedirs(result_save_path, exist_ok=True)

    parse_file_path = result_save_path + GetCurrentDatetime() + '.txt'
    with open(parse_file_path, 'w') as f:
        for idx, output in enumerate(output_ranges):
            # question = Object()
            # question.idx = idx
            # question.start_time = output[0]
            # response.result.append(question)
            f.write(str(idx) + ' ' + str(output[0]) + '\n')
    print('base_dir: ', os.getcwd())
    question_folder, results = toeic.inference_all_questions(question_save_path, cpu_num=24)


    result = []

    print('results:', results)
    for pool in results:
        for future in pool:
            question_idx = str(future[0])
            sentences = future[1]
            sent_start_time = future[2]
            for sentence, start_time in zip(sentences, sent_start_time):
                sentence_ = Object()
                sentence_.content = sentence
                sentence_.time = output_ranges[int(question_idx)][0] + start_time
                sentence_.protime = output_ranges[int(question_idx)][0]
                response.result.append(sentence_)
    # return sorted json
    response = sort_by_start_time(response)


    response.processing_time = time() - start

    return response

def test_over_1_min(path):
    toeic = ToeicAudioProcessor()
    audio = AudioSegment.from_wav(path)

    text, sent_start_time = toeic.process_over_1_min(audio)

    return text, sent_start_time

if __name__ == '__main__':

    path = '/data/sba/HBJ/sr_test/Toeic_2/TEST_1/2020_1025_174928/chunk0.wav'
    text, sent_start_time = test_over_1_min(path)
    print(text, sent_start_time)
    # path = '/data/TOEIC_Audio/Toeic_3/TEST_2.mp3'
    # res = process_toeic_full_test_Arraylist(path)
    # print(res)

    # with open('test_res.json', 'w') as f:
    #     f.write(res)
    #
    # print('json file saved')