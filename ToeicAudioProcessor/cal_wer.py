# from jiwer import wer
import jiwer

class CalculationWER:
    def __init__(self):
        pass

    def load_script(self, script):
        lines = script.readlines()
        lines = [line.lower() for line in lines]
        for line in lines:
            #line.append(script.readline())
            print(line)
        
        script.close()

        return lines
    
    def Cal_WER(self, answer_script, predict_script):
        # transformation = jiwer.Compose([
        #     jiwer.ToLowerCase(),
        #     jiwer.RemoveMultipleSpaces(),
        #     jiwer.RemoveWhiteSpace(replace_by_space=False),
        #     jiwer.RemovePunctuation()
        #     ]) 
        
        predict_error = jiwer.wer(answer_script, predict_script
                                #   ,truth_transform=transformation, 
                                #   hypothesis_transform=transformation
                                )
    
        return predict_error
        

if __name__ == '__main__':
    testWER = CalculationWER()

    answer_script = open("Script.txt", 'r')
    google_script = open("GoogleSTTResults.txt", 'r')
    
    print('~~~~~~~ Answer Script ~~~~~~~~')
    ground_truth = testWER.load_script(answer_script)
    #print(ground_truth)
    
    print('~~~~~~~ Google Script ~~~~~~~~')
    hypothesis = testWER.load_script(google_script)
    #print(hypothesis)

    pred_error = testWER.Cal_WER(ground_truth, hypothesis)
    print(pred_error)