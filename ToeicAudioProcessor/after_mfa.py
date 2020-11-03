script = "Nor would it thaw out his hands and feet."
sentence_start_time = [1,2,3,5,6,7,9]
question_start_time = [3,6,8,9,7,3,2]
save_web_data_path = 'web_process_data'
# audios = os.listdir()
# audios = [audio for audio in audios if audio.endswith('.txt')]
# def total_audio_sentence_time(self, script, sentence_start_time, question_start_time):

web_process_data = []
for per_question_start_time in question_start_time: 
    for per_sentence_start_time in sentence_start_time:
        web_process_data_tmp = []
        web_process_data_tmp.append(script)
        web_process_data_tmp.append(int(per_question_start_time)+ int(per_sentence_start_time))
        web_process_data_tmp.append(int(per_question_start_time))
        web_process_data.append(str(web_process_data_tmp[0]) + '|' + str(web_process_data_tmp[1])
                                                            + '|' + str(web_process_data_tmp[2]))
with open('web_process_data.txt','w') as f:
    f.write('\n'.join(web_process_data))
        
# total_audio_sentence_time(script, sentence_start_time, question_start_time,)
        