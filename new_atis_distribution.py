#old atis distribution doesn't have a good distribution by label.
# this script creates a new train test split that follows a distribution by label
from sklearn.model_selection import train_test_split

from collections import Counter
import os.path

old_atis_folder = "./ATIS"
new_atis_folder = "./NEW_ATIS"
folders = ["train", "test", "valid"]
files = [".seq.in", ".label"]
test_percentage = 20.0
valid_percentage = 20.0

input_sequence = []
output_label = []

if os.path.exists(new_atis_folder):
	exit()
	
#collect all of train, test and valid data from old atis dataset 
for folder in folders:
	input_file = os.path.join(old_atis_folder, folder, folder+files[0])
	output_file = os.path.join(old_atis_folder, folder, folder+files[1])
	if os.path.exists(input_file):
		with open(input_file) as i_f:
			input_sequence.extend(i_f.readlines())

	if os.path.exists(output_file):
		with open(output_file) as o_f:
			output_label.extend(o_f.readlines())
	

X_train, X_test, Y_train, Y_test  = train_test_split(input_sequence, output_label, test_size=float(test_percentage + valid_percentage)/ 100, random_state=42)
X_test, X_valid, Y_test, Y_valid  = train_test_split(X_test, Y_test, test_size=float(valid_percentage)/float(test_percentage + valid_percentage) , random_state=42)

train = Counter(Y_train)
test = Counter(Y_test)
valid = Counter(Y_valid)

print train.most_common()
print "\n\n"
print test.most_common()
print "\n\n"
print valid.most_common()


inputs =  [X_train, X_test, X_valid]
outputs = [Y_train, Y_test, Y_valid]
for j in range(3):
	try:
		os.makedirs(os.path.join(new_atis_folder, folders[j]))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise
	with open(os.path.join(new_atis_folder, folders[j], folders[j]+files[0]), 'w') as f:
		f.writelines(inputs[j])
	with open(os.path.join(new_atis_folder, folders[j], folders[j]+files[1]), 'w') as f:
		f.writelines(outputs[j])
