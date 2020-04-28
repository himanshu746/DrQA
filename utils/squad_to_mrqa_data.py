import json
import os
# from nltk.tokenize import TweetTokenizer
from cltk.tokenize.word import WordTokenizer

src = "/home/du0/15CS30016/MTP2/DrQA/data/datasets/bengali/v1.2/SQuAD-v1.1-train.json"
dest = "SQuAD_mrqa_bengali_train-v1.2.jsonl"
dataset_name = "squad_bengali"
split_type = "train" # train/dev

with open (src, 'r', encoding='utf-8') as f:
    json_data = json.load (f)

# tknzr = TweetTokenizer()
tknzr = WordTokenizer ('multilingual')
f = open (dest, 'w')
# f.write ('{"header": {"dataset": "{dname}", "split": "{dataset_type}"}}'.format (dname = dataset_name, dataset_type = dataset_type))
f.write (json.dumps ({"header": {"dataset": dataset_name, "split": split_type}}))
f.write ('\n')

data = json_data['data']

def getTokenSpans (s : str, tokens : list):
    offset = 0
    spans = list()
    for token in tokens:
        position = s.find (token, offset)
        spans.append ([token, position])
        offset = position + len (token)
    
    return spans
    
def getSpans (token_spans : list, start: int, end : int):
    # First token-span which has token index > start
    index = -1
    for token_span in token_spans:
        if token_span[1] > start:
            break
        index += 1
    
    start_token_index, start_char_index = index, token_spans[index][1]

    # First token-span which has position > end
    index = start_token_index
    for i in range (start_token_index + 1, len (token_spans), 1):
        if token_spans[i][1] > end:
            break
        index += 1

    end_token_index, end_char_index = index, token_spans[index][1] + len (token_spans[index][0]) - 1

    return {
        "token_spans": [
            [
                start_token_index,
                end_token_index
            ]
        ],
        "char_spans": [
            [
                start_char_index,
                end_char_index
            ]
        ]
    }

cnt = 0
for q in data:
    # if cnt == 5:
    #     break
    cnt += 1
    
    for par in q['paragraphs']:
        d = dict()
        context = par['context']
        d['context'] = context
        context_tokens = getTokenSpans (context, tknzr.tokenize (context))
        d['context_tokens'] = context_tokens
        d['qas'] = list()
        for question in par['qas']:
            question_text = question['question']
            answer_text = question['answers'][0]['text']
            answer_start_byte = question['answers'][0]['answer_start']
            answer_end_byte = answer_start_byte + len (answer_text) - 1
            qid = question['id']

            qa = {}
            qa['question'] = question_text
            qa['answers'] = [answer_text]
            qa['qid'] = qid
            tokens = tknzr.tokenize(question_text)
            qa['question_tokens'] = getTokenSpans (question_text, tokens)
            detected_answer = dict()
            detected_answer['text'] = answer_text
            spans = getSpans (context_tokens, answer_start_byte, answer_end_byte)
            detected_answer['token_spans'] = spans['token_spans']
            detected_answer['char_spans'] = spans ['char_spans']
            qa['detected_answers'] = [detected_answer]
            d['qas'].append (qa)

        json_str = json.dumps (d)
        f.write (json_str)
        f.write ('\n')

f.close()


