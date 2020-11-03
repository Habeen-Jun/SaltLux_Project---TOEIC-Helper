import pronouncing
from text.cleaners import english_cleaners

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
            text = english_cleaners(text)
        return text

oov_path = './MFA_result/oovs_found.txt'

    def oov(self, oov_path, save_path, lexicon='/data/sba/HBJ/sr_test/MFA_dic/librispeech-lexicon.txt'):
        with open(oov_path,'r+') as f_oov:
            oov = f_oov.read()
            f_oov.truncate(0)
            oov = oov.split('\n')
    
        if len(oov) > 1:
            new_dictionary = {}
            for oov_idx in range(len(oov)-1):
                oov_pronounce = pronouncing.phones_for_word(oov[oov_idx])
                if oov_pronounce != []:
                    new_dictionary[oov[oov_idx]] = oov_pronounce[0]
                
            print(new_dictionary)
            
            with open(dic_path, 'a+') as f_dic:
                for new_dic in new_dictionary:
                    f_dic.writelines(str(new_dic) + '\t' + str(new_dictionary[new_dic]) + '\n')