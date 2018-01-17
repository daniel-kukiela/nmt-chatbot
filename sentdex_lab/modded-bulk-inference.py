import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.realpath(os.path.dirname(__file__)) + "/nmt")
import argparse
from setup.settings import hparams, out_dir
from nmt import nmt
import tensorflow as tf
from core.tokenizer import tokenize, detokenize
from core.sentence import score_answers, replace_in_answers
import colorama
import scoring
import random

class InferenceBaseClass:
    def __init__(self,out_dir,hparams):
        self.current_stdout = None
        # Model, flags and hparams
        self.infer_model,self.flags,self.hparams = self.do_start_inference(out_dir,hparams)

    # Start inference "engine". ## Could merge this inside __init__ but keeping it seperate for readablity sake.
    def do_start_inference(self,out_dir, hparams):

        # Silence all outputs
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.current_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        # Modified autorun from nmt.py (bottom of the file)
        # We want to use original argument parser (for validation, etc)
        nmt_parser = argparse.ArgumentParser()
        nmt.add_arguments(nmt_parser)
        # But we have to hack settings from our config in there instead of commandline options
        flags, unparsed = nmt_parser.parse_known_args(['--'+k+'='+str(v) for k,v in hparams.items()])
        # And now we can run TF with modified arguments
        #tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)

        # Add output (model) folder to flags
        flags.out_dir = out_dir

        # Make hparams
        hparams = nmt.create_hparams(flags)

        ## Train / Decode
        if not tf.gfile.Exists(flags.out_dir):
            nmt.utils.print_out("# Model folder (out_dir) doesn't exist")
            sys.exit()

        # Load hparams from model folder
        hparams = nmt.create_or_load_hparams(flags.out_dir, hparams, flags.hparams_path, save_hparams=True)

        # Choose checkpoint (provided with hparams or last one)
        if not flags.ckpt:
            flags.ckpt = tf.train.latest_checkpoint(flags.out_dir)

        # Create model
        if not hparams.attention:
            model_creator = nmt.inference.nmt_model.Model
        elif hparams.attention_architecture == "standard":
            model_creator = nmt.inference.attention_model.AttentionModel
        elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = nmt.inference.gnmt_model.GNMTModel
        else:
            raise ValueError("Unknown model architecture")
        infer_model = nmt.inference.model_helper.create_infer_model(model_creator, hparams, None)

        return (infer_model, flags, hparams)

    # Inference
    def do_inference(self,phrase):

        infer_data = [phrase]

        # Disable TF logs for a while
        # Workaround for bug: https://github.com/tensorflow/tensorflow/issues/12414
        # Already fixed, available in nightly builds, but not in stable version
        # Maybe that will stay here to silence any outputs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Spawn new session
        with tf.Session(graph=infer_model.graph, config=nmt.utils.get_config_proto()) as sess:

            # Load model
            loaded_infer_model = nmt.inference.model_helper.load_model(self.infer_model.model, self.flags.ckpt, sess, "infer")

            # Run model (translate)
            sess.run(
                self.infer_model.iterator.initializer,
                feed_dict={
                    self.infer_model.src_placeholder: infer_data,
                    self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size
                })


            # calculate number of translations to be returned
            num_translations_per_input = max(min(self.hparams.num_translations_per_input, self.hparams.beam_width), 1)
            translations = []

            try:
                nmt_outputs, _ = loaded_infer_model.decode(sess)
                if self.hparams.beam_width == 0:
                    nmt_outputs = nmt.inference.nmt_model.np.expand_dims(nmt_outputs, 0)

                # Iterate through responses
                for beam_id in range(num_translations_per_input):

                    if self.hparams.eos: tgt_eos = self.hparams.eos.encode("utf-8")

                    # Select a sentence
                    output = nmt_outputs[beam_id][0, :].tolist()

                    # If there is an eos symbol in outputs, cut them at that point
                    if tgt_eos and tgt_eos in output:
                        output = output[:output.index(tgt_eos)]

                    # Format response
                    if self.hparams.subword_option == "bpe":  # BPE
                        translation = nmt.utils.format_bpe_text(output)
                    elif self.hparams.subword_option == "spm":  # SPM
                        translation = nmt.utils.format_spm_text(output)
                    else:
                        translation = nmt.utils.format_text(output)

                    # Add response to array
                    translations.append(translation.decode('utf-8'))

            except tf.errors.OutOfRangeError:
                pass

            # bug workaround end
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            sys.stdout.close()
            sys.stdout = self.current_stdout
            self.current_stdout = None

            return translations

    # Internal inference function (for direct call)
    def inference_internal(self,question):
        answers = self.do_inference(tokenize(question))
        answers = detokenize(answers)
        answers = replace_in_answers(answers, 'answers')
        answers_rate = score_answers(answers, 'answers')
        return (answers, answers_rate)

    # Main inference function
    def inference(self, question, include_blacklisted = True):
        answers, answers_rate = inference_internal(question)
        
        #Compacting the if-else block.
        check = [1,0,-1]          
        # In future, if you need to perform quality analysis and assign more value,
        # simply append them to list instead of writing so many if statements.
        while len(check) > 0:
            try:
                # I believe answers_rate.index(0) and include_blacklist imply each other's existence.
                index = answers_rate.index(check[0]) if check[0] != -1 else 0 
                score = check[0]
            except:
                check.pop(0)       

        # If include_blacklist must be checked at any cost:
        # check = [(1,True),(0,include_blacklist),(-1,True)]
        # while len(check) > 0:
        #    try:
        #        index = answers_rate.index(check[0][0]) if (check[0][0] != -1) and check[0][1] else 0 
        #        score = check[0][0]
        #    except:
        #        check.pop(0)        

        return {'answers': answers, 'index': index, 'score': score}

# interactive mode
if __name__ == "__main__":

    files = ['df_no_reply_questions','df_reply_questions']

    for file in files:
    
        print("\n\nStarting...")
        colorama.init()
        infer = InferenceBaseClass(out_dir,hparams)
        with open("comparisons/{}.txt".format(file),"r", encoding='utf8') as f:
            contents = f.read().split('\n')
            for question in contents[:-1]:
                answers, answers_rate = infer.inference_internal(question)

                ans_score = {}
                for i, answer in enumerate(answers):

                    wrapper = {}
                    for i, answer in enumerate(answers):
                        wrapped = {"question": question, "answer": answer, "score": answers_rate[i]}
                        wrapper.append(wrapped)
                    choice_answer = scoring.get_choice(wrapper)

                with open('comparisons/{}.out'.format(file),"a", encoding='utf8') as f:
                    f.write(choice_answer+'\n')
                  
                    
                '''with open("comparisons/full_{}.out".format(file),"a", encoding='utf8') as f:
                    
                    for i, _ in enumerate(answers):

                        f.write('>>>' + str(question)+'\n')
                        print('>>>', question)

                        f.write(str(answers_rate[i])+' ::: '+str(answers[i])+'\n')
                        print("{}- {}{}".format(colorama.Fore.GREEN if answers_rate[i] == 1 else colorama.Fore.YELLOW if answers_rate[i] == 0 else colorama.Fore.RED, answers[i], colorama.Fore.RESET))
                    f.write('\n\n\n')'''
