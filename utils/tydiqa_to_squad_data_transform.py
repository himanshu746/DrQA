import os, sys, json, argparse

def transformData (args):
    src, dest, err_file = args.src_file, args.target_file, args.error_log_file

    tydiqa_data = list()
    with open (src, 'r', encoding = 'utf-8') as f:
        for line in f:
            tydiqa_data.append (json.loads (line))

    data = list()
    errors = list()
    for td in tydiqa_data:
        annotations = td['annotations']
        if len (annotations) == 0:
            continue
        annotation = annotations[0]
        if annotation['yes_no_answer'] != 'NONE':
            continue
        passage_candidate = annotation['passage_answer']['candidate_index']
        minimal_answer = annotation['minimal_answer']
        if passage_candidate == -1:
            continue
        ans_start_byte, ans_end_byte = minimal_answer['plaintext_start_byte'], minimal_answer['plaintext_end_byte']

        if ans_start_byte == -1 or ans_end_byte == -1:
            continue

        ct = td['document_plaintext']
        cb = ct.encode ('utf-8')
        title = td['document_title']
        example_id = td['example_id']
        qt = td['question_text']

        try:
            par_limits = td['passage_answer_candidates'][passage_candidate]
            par_start_byte = par_limits['plaintext_start_byte']
            par_end_byte = par_limits['plaintext_end_byte']
            parb = cb[par_start_byte:par_end_byte]
            part = parb.decode ('utf-8')

            byte_answer = cb[ans_start_byte : ans_end_byte]
            answer_text = byte_answer.decode ('utf-8')

            ans_start_index = part.find (answer_text)

            if (ans_start_index == -1):
                print (example_id)
                exit()

            ans_end_index = ans_start_index + len (answer_text)

            data.append (
                {
                    'title': title,
                    'paragraphs': [
                        {
                            'context': part,
                            'qas': [
                                {
                                    'answers': [
                                        {
                                            "answer_start": ans_start_index,
                                            "text": answer_text
                                        }
                                    ],
                                    "question": qt,
                                    "id": str (example_id)
                                }
                            ]
                        }
                    ]
                }
            )
        except:
            errors.append (title)

    with open (dest, 'w', encoding='utf-8') as f:
        json.dump ({
            'data': data
        }, f)

    with open (err_file, 'w', encoding='utf-8') as f:
        json.dump ({
            'ErrorTitles': errors
        }, f)
    print ('{num_samples} written to {destination}'.format (num_samples = len (data), destination = dest))
    print ('error in {num_error} files. The title of all these documents saved in {fname}'.format (num_error = len (errors), fname = err_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument ("--src_file", default="bengali-dev.jsonl", help="Source tydiqa dataset file", type = str)
    parser.add_argument ("--target_file", default="bengali-squad-dev.json", help="Destination squad dataset file", type = str)
    parser.add_argument ("--error_log_file", default="error_log.json", help = "File to write title of documents with error", type = str)

    args = parser.parse_args()
    
    transformData (args)

