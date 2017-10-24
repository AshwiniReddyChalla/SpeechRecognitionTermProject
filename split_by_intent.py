import os
import sys
import pandas as pd


def print_file(file, set):
    # Open the label file
    label_file = "ATIS/" + set + "/" + set + ".label"
    label = pd.read_csv(label_file)

    # This contains a list of all intents
    intents = label.flight.unique()

    # This is the full input file
    file_name = "ATIS/" + set + "/" + file
    file_df = pd.read_csv(file_name).values.tolist()

    intent_index = {}

    for intent in intents:
        # Filter the input label file for indexes for current intent
        intent_index[intent] = label.loc[label["flight"] == intent].index.values

        # Output file
        intent_file_dir = sys.path[0] + "/processed/" + intent + "/" + set
        inten_file_name = intent_file_dir + "/" + file

        # Create the directory if it does not exist
        if not os.path.exists(intent_file_dir):
            try:
                os.makedirs(intent_file_dir)
            except OSError as exc:
                print "Error creating directory!"

        # Open the output file
        intent_file = open(inten_file_name, 'w')

        # Write each intent entry of the input file
        for l in intent_index[intent]:
            intent_file.write(file_df[l][0] + "\n")

        intent_file.close()


if __name__ == "__main__":
    set = ['train', 'test', 'valid']
    file_set = ['.seq.in', '.seq.out', '.label']

    # Call print file for each combination of set and file_set
    for s in set:
        for f in file_set:
            print_file(file=s+f,
                       set=s)
