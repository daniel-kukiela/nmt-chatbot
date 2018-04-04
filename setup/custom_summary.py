
# Default custom summary (does nothing)
def custom_summary(data):
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
