
# Custom summary for math model
def custom_summary(data):

    try:
        model_in, actual_out, model_out = list(zip(*data))

        correct = 0
        total = 0
        total_off = 0

        muls = 0
        divs = 0
        adds = 0
        subs = 0

        for i, _ in enumerate(model_out[:-1]):

            if actual_out[i] == model_out[i]:
                correct += 1
            total += 1

            try:
                diff = abs(float(actual_out[i].replace(" ", "")) - float(model_out[i].replace(" ", "")))
            except:
                diff = abs(float(actual_out[i].replace(" ", "")))
            total_off += diff

            if "+" in model_in[i]:
                adds += diff
            elif "*" in model_in[i]:
                muls += diff
            elif "/" in model_in[i]:
                divs += diff
            else:
                subs += diff

        evaluations = {
            'math_total_acc': float(round(correct / total, 4)),
            'math_tdiff':     float(round(total_off, 4)),
            'math_muls':      float(round(muls, 4)),
            'math_divs':      float(round(divs, 4)),
            'math_adds':      float(round(adds, 4)),
            'math_subs':      float(round(subs, 4)),
        }

        print("!!! Math model evaluation: Total Acc {} | TDiff: {} | Muls {} | Divs {} | Adds {} | Subs {}".format(
                                                                                             evaluations['math_total_acc'],
                                                                                             evaluations['math_tdiff'],
                                                                                             evaluations['math_muls'],
                                                                                             evaluations['math_divs'],
                                                                                             evaluations['math_adds'],
                                                                                             evaluations['math_subs']))
        return evaluations

    except Exception as e:
        print('!!! Error in custom summary function: {}'.format(str(e)))
        return {}

# Example custom summary
def example_custom_summary(data):

    # Data object is a list of tuples, where tuple contains:
    #  - source phrase
    #  - target phrase
    #  - nmt phrase
    #
    # Return must be a dictionary, where:
    #  - key - lowercase ascii letters only plus underscore
    #  - value - float value
    #
    # Function is called on every evaluation
    # As a result returned values will be saved in logs and plot in TensorBoard

    # Example usage - return average length of phrases

    # Always use try .. except - in case of error in your code it will ensure training will continue
    try:
        lengths = list(zip(*[(len(src), len(tgt), len(nmt)) for src, tgt, nmt in data]))
        return {
            'average_len_tgt': sum(lengths[0]) / float(len(lengths[0])),
            'average_len_src': sum(lengths[1]) / float(len(lengths[1])),
            'average_len_nmt': sum(lengths[2]) / float(len(lengths[2])),
        }
    except Exception as e:
        print('Error in custom summary function: {}'.format(str(e)))
        return {}
