import io
import os
from google.cloud import speech_v1
import time
from ToeicAudioProcessor import GetCurrentDatetime


def STT(audio_path, save_path=None):
    client = speech_v1.SpeechClient.from_service_account_json(
            '/data/second-conquest-293723-05738e995f8f.json')

    # Loads the audio into memory
    with io.open(audio_path, "rb") as audio_file:
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
    print(text)

    audio_name = audio_path.split('/')[-1].replace('.mp3','')
    save_file_name = audio_name + GetCurrentDatetime() + '.txt'

    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        os.chdir(save_path)

    with open(save_file_name, 'w') as f:
        f.write(text)


    print('Inferred Audio File Name: ', audio_path)
    print('Transcribed Script File Saved: ', save_file_name)
    print('Processing Time: ', time.time() - start)


if __name__ == '__main__':
    # The name of the audio file to transcribe
    file_name = "/data/sba/HBJ/sr_test/Test02_68-70.mp3"
    save_path = "./MFA_sample"
    STT(file_name, save_path)