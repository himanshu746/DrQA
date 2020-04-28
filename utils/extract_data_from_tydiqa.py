import json

def extract_data (req_language, input_json_path, output_json_path):
	print ('Required Language : {rl}'.format (rl = req_language))
	print ('Input Json Path : {ijp}'.format (ijp = input_json_path))
	print ('Output Jons Path : {ojp}'.format (ojp = output_json_path))
	fin = open (input_json_path, 'r')
	fout = open (output_json_path, 'w')
	cnt = 0
	for line in fin:
		single_data = json.loads (line)
		if single_data['language'].lower() == req_language:
			fout.write (line)
			cnt += 1

	print ('size of data', cnt)
	# json.dump (data, fout)
	fin.close()
	fout.close()

def main():
	req_languages = ['english', 'bengali', 'telugu']
	input_json_path = 'v1.0_tydiqa-v1.0-train.jsonl'
	for req_language in req_languages:
		out_json_path = req_language + '-train.json'
		extract_data (req_language, input_json_path, out_json_path)

if __name__ == '__main__':
    main()

