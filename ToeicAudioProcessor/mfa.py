import os
from praatio import tgio
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from text.cleaners import english_cleaners

def get_first_words_idx(script_path):
    with open(script_path) as f:
        text = f.read()
        sents = sent_tokenize(text)
        print('sentences number: ',len(sents))
        text = english_cleaners(text)
        words = text.split(' ')
        first_words_idx = [0]
        # 문장 마지막 단어 인덱스 구하기
        for idx, word in enumerate(words):
            if word.endswith('.') or word.endswith('?') or word.endswith('!'):
                first_words_idx.append(idx + 1)

        return first_words_idx

def find_first_word_location(script_file_path, textgrid_file_path):
    tg = tgio.openTextgrid(textgrid_file_path)
    first_words_idx = get_first_words_idx(script_file_path)

    first_words_idx = first_words_idx[:len(first_words_idx)-1]
    sent_start_time = []
    for idx in first_words_idx:
        start = tg.tierDict["words"].entryList[idx].start
        sent_start_time.append(start)

    return sent_start_time


def find_all_first_word_location(script_file_folder, textgrid_file_folder):
    base_dir = os.getcwd()

    os.chdir(textgrid_file_folder)
    textgrids = [textgrid for textgrid in os.listdir() if textgrid.endswith('.TextGrid')]

    os.chdir(base_dir)


    os.chdir(script_file_folder)
    scripts = [script for script in os.listdir() if script.endswith('.txt') and script.replace('.txt','.TextGrid') in textgrids]



    assert len(scripts) == len(textgrids)

    all_questions_with_sentence_start_time = []
    for i in range(len(scripts)):
        sent_start_time = find_first_word_location(os.path.join(script_file_folder,scripts[i]),
                                                   os.path.join(textgrid_file_folder, textgrids[i]))
        print(sent_start_time)
        all_questions_with_sentence_start_time.append(sent_start_time)

    return all_questions_with_sentence_start_time








def mfa(sample_path, save_path, lexicon='/data/sba/HBJ/sr_test/MFA_dic/librispeech-lexicon.txt'):
    result = os.popen('/data/sba/CWK/MFA/montreal-forced-aligner/bin/mfa_align '+sample_path+' '+lexicon+' '+'english'+' '+save_path).read()
    print(result)


if __name__ == '__main__':
    # mfa - only supports .wav file
    # sample_path = '/data/sba/HBJ/sr_test/MFA_sample/'
    # save_path = '/data/sba/HBJ/sr_test/MFA_result/'
    # mfa(sample_path, save_path)

    # get last word location based on the result of mfa
    text_file = '/data/sba/HBJ/sr_test/MFA_sample/test2.txt'
    text_grid_file = '/data/sba/HBJ/sr_test/MFA_sample/test2.TextGrid'
    sent_start_time = find_first_word_location(text_file, text_grid_file)
    print(sent_start_time)
    # print(last_word_location)