import json
import en_core_web_sm
import random
import argparse
from utils_prompt import *
import re

parser = en_core_web_sm.load()
special_words = ["am", "is", "was", "are", "were", "can", "could", "will",
                 "would", "shall", "should", "may", "must", "might"]
position_words =         ['east', 'west', 'south', 'north', 'front', 'back', 'left', 'right', 'far', 'near']
convert_position_words = ['west', 'east', 'north', 'south', 'back', 'front', 'right', 'left', 'near', 'far']

unit_words = ["second", 'minute', "hour",'°C','°F','mm','cm','m','km', 'l', 'ml', 'g', 'kg']  #0-2, 3-4, 5-8, 9-10, 11-12
# TODO: 替换反义词
# 肯定-》否定，否定-》肯定，with和without，动词肯定与否定。 主要涉及回答yes or no问题。
# 数字变换，数字理解
# 方位变换，方向或地图理解
# 单位换算变换，单位理解，不同度量的比较。



# 文本过长，能处理的点太多，怎么选？E更重要
# 一个sample尽可能的少变化，这样sample与golden的分布更近，

def get_index(ref, key):
    return [i for i,val in enumerate(ref) if val==key]
def get_num(n):
    """
        Returns: 等长随机数字
        ret:random n num
    """
    ret = ""
    num = random.randint(1, 9)
    s = str(random.choice([num]))
    ret += s
    for i in range(1,n):
        num = random.randint(0, 9)
        s = str(random.choice([num]))
        ret += s
    return ret
def convert_unit_index(i):
    # 随机返回一个其他的同类单位的下标
    if 0 <= i <= 2:
        l, r = 0, 2
    elif 3 <= i <= 4:
        l, r = 3, 4
    elif 5 <= i <= 8:
        l, r = 5, 8
    elif 9 <= i <= 10:
        l, r = 9, 10
    elif 11 <= i <= 12:
        l, r = 11, 12
    else:
        print("out of index boundary!")
    if i - 1 < l:
        return i + 1
    if i + 1 > r:
        return i - 1
    return random.choice([i-1, i+1])


def convert_text_no_random(solution, type = 0, choice = None, answer = None, parser=parser, cache = None ):
    '''
    :param parser: parser
    :param solution: sentence
    :param type: [0,1,2,3,4]. 0: 肯定否定转换; 1: 数字变换; 2: 方位变换; 3: 单位变换; 4: 选项替换; 5. 全部.
    :return: converted text
    '''
    if cache is not None:
        tokens = cache['tokens'].copy()
        deps = cache['deps'].copy()
        tags = cache['tags'].copy()
        lemmas = cache['lemmas'].copy()
    else:
        parsered_sentence = parser(solution)
        tokens = [str(_) for _ in parsered_sentence]
        deps = [_.dep_ for _ in parsered_sentence]
        # print(deps)
        tags = [_.tag_ for _ in parsered_sentence]
        lemmas = [_.lemma_ for _ in parsered_sentence]
        # print(tokens)
        # print(lemmas)
    ok = False
    if type == 0:
        if "not" in tokens:     # 否定-》肯定
            index = tokens.index("not") # index = get_index(tokens, "not")
            del tokens[index]
            sentence_negation = " ".join(tokens)
            return sentence_negation
        if "n't" in tokens:
            index = tokens.index("n't")
            if lemmas[index-1] != 'be':
                tokens[index-1] = lemmas[index-1]
            del tokens[index]
            sentence_negation = " ".join(tokens)
            return sentence_negation

        # 肯定-》否定
        flag = 0
        for dep in deps:
            if dep == "aux" or dep == "auxpass":
                flag = 1
                break
            if dep == "ROOT":
                flag = 2
                break

        if flag == 1:
            for i, dep in enumerate(deps):
                if dep == "aux" or dep == "auxpass":
                    tokens[i] += " not"
                    break
        elif flag == 2:
            index = deps.index("ROOT")
            if tokens[index].lower() in special_words:
                tokens[index] += " not"
            elif tags[index] == "VBP":
                tokens[index] = "do not " + lemmas[index]
            elif tags[index] == "VBZ":
                tokens[index] = "does not " + lemmas[index]
            elif tags[index] == "VBD":
                tokens[index] = "did not " + lemmas[index]
            else:
                tokens.insert(0, "Not")
        else:
            tokens.insert(0, "Not")
        ok=True
    elif type == 1:         # 数字变换
        for dep in deps:
            if dep == "nummod":
                index = deps.index("nummod")
                tmp_num = get_num(n = len(tokens[index]))
                while tmp_num == tokens[index]:
                    tmp_num = get_num(n=len(tokens[index]))
                tokens[index] = tmp_num  # 随机替换一个等长的数字
                ok = True
                break
    elif type == 2:     # 方位变换
        for i, token in enumerate(tokens):
            if token in position_words:
                index = position_words.index(token)
                tokens[i] = convert_position_words[index]
                ok = True
                break
    elif type == 3:     # 单位变换
        for i, token in enumerate(tokens):
            if lemmas[i] in unit_words:
                index = unit_words.index(lemmas[i])
                tokens[i] = unit_words[convert_unit_index(index)]
                ok = True
                break
    elif type == 4:     # 选项替换
        pattern = "\([A-Z]\)"
        res = re.findall(pattern, choice, re.IGNORECASE|re.S)
        choice_num = len(res)
        pre_choice_ctr = ord(answer[1]) - ord('A')
        choice_ctr = pre_choice_ctr
        while choice_ctr == pre_choice_ctr:
            choice_ctr = pre_choice_ctr + random.randint(0 - pre_choice_ctr, choice_num - pre_choice_ctr - 1)
        content_pattern = res[choice_ctr].replace('(', '\(').replace(')', '\)') + " (.*?)" + '[\.\(]'
        choice_content = re.search(content_pattern, choice + '(', re.IGNORECASE|re.S)
        content_pattern = res[pre_choice_ctr].replace('(', '\(').replace(')', '\)') + " (.*?)" + '[\(\.]'
        answer_content = re.search(content_pattern, choice + '(', re.IGNORECASE|re.S)[0][4:-1].strip().lower()
        # print("[Follow]:" + res[choice_ctr])
        if choice_content is not None:
            choice_content = choice_content[0][4:-1].strip()
            # print("[new choice]:" + choice_content)
            s = solution.lower().replace(answer_content, choice_content.lower(), 1)
            if s==solution:
                cache={
                    'tokens':tokens,
                    'deps':deps,
                    'tags':tags,
                    'lemmas':lemmas,
                }
                return convert_text(solution, parser=parser,cache=cache)
            return s
    elif type==5:
        # 因为存在句法解析，一起搞可能会快一些
        cache = {
            'tokens': tokens,
            'deps': deps,
            'tags': tags,
            'lemmas': lemmas,
        }
        sentence_negations = []
        ret = convert_text(solution, type=0, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=1, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=2, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=3, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=4, choice=choice,answer=answer,parser=parser, cache=cache)
        sentence_negations.append(ret)
        return sentence_negations
    else:
        print("error type!")
    if ok is not True:
        cache = {
            'tokens': tokens,
            'deps': deps,
            'tags': tags,
            'lemmas': lemmas,
        }
        return convert_text(solution, parser=parser,cache=cache)
    sentence_negation = " ".join(tokens)

    return sentence_negation


def convert_text(solution, type = 0, choice = None, answer = None, parser=parser, cache = None, l = 1 ):
    '''
    :param parser: parser
    :param solution: sentence
    :param type: [0,1,2,3,4]. 0: 肯定否定转换; 1: 数字变换; 2: 方位变换; 3: 单位变换; 4: 选项替换; 5. 全部.
    :return: converted text
    '''
    if cache is not None:
        # 缓存，同个句子的句法解析结果一致，重复解析浪费时间
        tokens = cache['tokens'].copy()
        deps = cache['deps'].copy()
        tags = cache['tags'].copy()
        lemmas = cache['lemmas'].copy()
    else:
        parsered_sentence = parser(solution)
        tokens = [str(_) for _ in parsered_sentence]
        deps = [_.dep_ for _ in parsered_sentence]
        # print(deps)
        tags = [_.tag_ for _ in parsered_sentence]
        lemmas = [_.lemma_ for _ in parsered_sentence]
        # print(tokens)
        # print(lemmas)
    ok = False
    if type == 0:
        if "not" in tokens:     # 否定-》肯定
            indexs = get_index(tokens, "not")
            if l == -1:   # 全部替换
                tokens = [i for num, i in enumerate(tokens) if num not in indexs]
            else:
                ltmp = min(l,len(indexs))
                index =random.sample(indexs, ltmp)
                # index = random.randint(0,len(indexs)-1)
                # index = indexs[index]
                tokens = [i for num, i in enumerate(tokens) if num not in index]
            sentence_negation = " ".join(tokens)
            return sentence_negation
        if "n't" in tokens:
            indexs = get_index(tokens, "n't")
            if l == -1:
                for i in range(len(indexs)):
                    index = indexs[i]
                    if lemmas[index - 1] != 'be':
                        tokens[index - 1] = lemmas[index - 1]
                tokens = [i for num, i in enumerate(tokens) if num not in indexs]
            else:
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                for i in index:
                    if lemmas[i-1] != 'be':
                        tokens[i-1] = lemmas[i-1]
                tokens = [i for num, i in enumerate(tokens) if num not in index]
            sentence_negation = " ".join(tokens)
            return sentence_negation

        # 肯定-》否定
        flag = 0
        for dep in deps:
            if dep == "aux" or dep == "auxpass":
                flag = 1
                break
            if dep == "ROOT":
                flag = 2
                break

        if flag == 1:
            indexs = []
            for i, dep in enumerate(deps):
                if dep == "aux" or dep == "auxpass":
                    indexs.append(i)
            if l == -1:
                for i in range(len(indexs)):
                    index = indexs[i]
                    tokens[index] += " not"
            else:
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                for i in index:
                    tokens[i] += " not"
        elif flag == 2:
            index = deps.index("ROOT")
            if tokens[index].lower() in special_words:
                tokens[index] += " not"
            elif tags[index] == "VBP":
                tokens[index] = "do not " + lemmas[index]
            elif tags[index] == "VBZ":
                tokens[index] = "does not " + lemmas[index]
            elif tags[index] == "VBD":
                tokens[index] = "did not " + lemmas[index]
            else:
                tokens.insert(0, "Not")
        else:
            tokens.insert(0, "Not")
        ok=True
    elif type == 1:         # 数字变换
        if "nummod" in deps:
            indexs = get_index(deps, "nummod")
            if l == -1:
                for i in range(len(indexs)):
                    index = indexs[i]
                    tmp_num = get_num(n=len(tokens[index]))
                    while tmp_num == tokens[index]:
                        tmp_num = get_num(n=len(tokens[index]))
                    tokens[index] = tmp_num  # 随机替换一个等长的数字
            else:
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                for i in index:
                    tmp_num = get_num(n = len(tokens[i]))
                    while tmp_num == tokens[i]:
                        tmp_num = get_num(n=len(tokens[i]))
                    tokens[i] = tmp_num  # 随机替换一个等长的数字
            ok = True
    elif type == 2:     # 方位变换
        indexs = []
        for i, token in enumerate(tokens):
            if token in position_words:
                indexs.append(i)
        if len(indexs) > 0:
            if l == -1:
                for i in range(len(indexs)):
                    index = indexs[i]
                    tokens[index] = convert_position_words[position_words.index(tokens[index])]
            else:
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                for i in index:
                    tokens[i] = convert_position_words[position_words.index(tokens[i])]
            ok = True
    elif type == 3:     # 单位变换
        indexs = []
        for i, token in enumerate(tokens):
            if lemmas[i] in unit_words:
                indexs.append(i)
        if len(indexs) > 0:
            if l == -1:
                for i in range(len(indexs)):
                    index = indexs[i]
                    tokens[index] = unit_words[convert_unit_index(unit_words.index(lemmas[index]))]
            else:
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                for i in index:
                    tokens[i] = unit_words[convert_unit_index(unit_words.index(lemmas[i]))]
            ok = True
    elif type == 4:     # 选项替换
        pattern = "\([A-Z]\)"
        res = re.findall(pattern, choice, re.IGNORECASE|re.S)
        choice_num = len(res)
        pre_choice_ctr = ord(answer[1]) - ord('A')
        choice_ctr = pre_choice_ctr
        while choice_ctr == pre_choice_ctr:
            choice_ctr = pre_choice_ctr + random.randint(0 - pre_choice_ctr, choice_num - pre_choice_ctr - 1)
        content_pattern = res[choice_ctr].replace('(', '\(').replace(')', '\)') + " (.*?)" + '[\.\(]'
        choice_content = re.search(content_pattern, choice + '(', re.IGNORECASE|re.S)
        content_pattern = res[pre_choice_ctr].replace('(', '\(').replace(')', '\)') + " (.*?)" + '[\(\.]'
        answer_content = re.search(content_pattern, choice + '(', re.IGNORECASE|re.S)[0][4:-1].strip().lower()
        # print("[Follow]:" + res[choice_ctr])
        if choice_content is not None:
            choice_content = choice_content[0][4:-1].strip().lower()
            s = solution.lower()
            # print("[new choice]:" + choice_content)
            pos = -1
            indexs = []
            while True:
                pos = s.find(answer_content, pos+1)
                if pos == -1:
                    break
                indexs.append(pos)
            if len(indexs) == 0:
                cache={
                    'tokens':tokens,
                    'deps':deps,
                    'tags':tags,
                    'lemmas':lemmas,
                }
                return convert_text(solution, parser=parser,cache=cache)
            if l == -1:
                s.replace(answer_content, choice_content)
            else:
                ltmp = min(l, len(indexs))
                index = random.sample(indexs, ltmp)
                # index = random.randint(0, len(indexs)-1)
                # index = indexs[index]
                for i in index:
                    s = s[:i]+s[i:].replace(answer_content, choice_content, 1)
            return s
    elif type==5:
        # 因为存在句法解析，一起搞可能会快一些
        cache = {
            'tokens': tokens,
            'deps': deps,
            'tags': tags,
            'lemmas': lemmas,
        }
        sentence_negations = []
        ret = convert_text(solution, type=0, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=1, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=2, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=3, parser=parser, cache=cache)
        sentence_negations.append(ret)
        ret = convert_text(solution, type=4, choice=choice,answer=answer,parser=parser, cache=cache)
        sentence_negations.append(ret)
        return sentence_negations
    else:
        print("error type!")
    if ok is not True:
        cache = {
            'tokens': tokens,
            'deps': deps,
            'tags': tags,
            'lemmas': lemmas,
        }
        return convert_text(solution, parser=parser,cache=cache)
    sentence_negation = " ".join(tokens)

    return sentence_negation


if __name__ == "__main__":


    # type = eval(input())
    # s = input()
    # negation = convert_text(parser = parser, sentence = s, type = type)
    # print(negation)
    # exit()
    parser = argparse.ArgumentParser()
    from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])

    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--init_manual_template', action='store_true',
                        help='whether to use manual template to initialize the dense vectors')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
    dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps, 'image_features': image_features}
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    i = eval(input())
    qid = val_qids[i]
    choice = get_choice_text(problems[qid], args.options)
    answer = get_answer(problems[qid], args.options)
    solution = get_solution_text(problems[qid])
    parser = en_core_web_sm.load()
    print(convert_text(solution,type=4,choice=choice, answer=answer,parser=parser))
