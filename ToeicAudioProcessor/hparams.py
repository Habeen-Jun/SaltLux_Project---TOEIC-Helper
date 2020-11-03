# Question Segmentation
class Question_Segmentation:
    min_silence_len = 4500
    min_duration = 10000
    max_duration = 60000
    threshold = -50
    seek_step = 150
    sr = 22050


class Sentence_Segmentation:
    top_db = 70

if __name__ == '__main__':
    print(Question_Segmentation.threshold)


