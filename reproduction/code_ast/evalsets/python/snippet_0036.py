import sys
import math
from os.path import isfile

def cut_string(stringin, size):

	out = []

	in_length = len(stringin)
	fsize = float(size)

	for i in range(int(math.ceil(in_length/fsize))):

		start = i * size
		end = min(in_length, (i + 1) * size)
		out += [stringin[start:end]]

	return out


def to_fasta(filepath_in, filepath_out):

	with open(filepath_in, 'r') as go:
		strin = go.read()
		#print strin
		strin_split = strin.split('$$\n')[:-1]
		# print strin_split[0]
		strin_split_content = [line.split('\n')[0] for line in strin_split]
		# print strin_split_content

		strout_seq = [line.split('\t')[2] for line in strin_split_content]
		# print strout_seq[0]
		strout_seq_cut = [cut_string(line, 60) for line in strout_seq]
		# print strout_seq_cut[0]
		strout_seq_cut_merge = ['\n'.join(line) for line in strout_seq_cut]
		# print strout_seq_cut_merge[0]
		strout_name = [line.split('\t')[1] for line in strin_split]
		# print strout_name[0]
		strout_zip = zip(strout_name, strout_seq_cut_merge)
		# print strout_zip[0]
		strout_unzip = ['>' + '\n'.join(line) for line in strout_zip]
		# print strout_unzip[0]
		
		with open(filepath_out, 'w') as out:
			out.write('\n'.join(strout_unzip))
			out.close()


if  __name__ =='__main__':

	if len(sys.argv) < 5 or not isfile(sys.argv[1]) or not isfile(sys.argv[2]):
		print "Usage: transform_binding_files_to_fasta.py <dna.out> <rna.out> <dna.fasta> <rna.fasta>";
		sys.exit();

	dna_in = sys.argv[1]
	rna_in = sys.argv[2]
	dna_out = sys.argv[3]
	rna_out = sys.argv[4]

	to_fasta(dna_in, dna_out)
	to_fasta(rna_in, rna_out)