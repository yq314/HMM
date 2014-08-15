'''
Created on Apr 5, 2013

@author: qingye
'''

import sys
import math

START_CODON = ['ATG']
STOP_CODON = ['TAA', 'TAG', 'TGA']
PLUS_END_STATE = 20
MINUS_END_STATE = 5
PSEUDO_PROB = 0.01
THRESHOLD = 0

def read_sequences(file_name):
    try:
        with open(file_name) as fh:
            for line in fh:
                yield line.strip()
    except IOError:
        print('ERROR: failed to read file: ' + file_name)
        sys.exit(-1)
        
def init_matrix(size):
    return [[0] * size for row in range(size)]

def print_matrix(matrix):
    for row in matrix:
        for col in row:
            print(str(col) + "\t", end="")
        print("")

def trans_matrix_to_prob(matrix):
    m = []
    for row in matrix:
        total = sum(row)
        if total == 0:
            m.append([0.0] * len(row))
        else:
            m.append([col / total for col in row])
    return m

def traverse_plus_matrix(seq, matrix, is_training):
    p = 1
    last_state = 0
    curate_len = len(seq) - len(seq) % 3
    for i in range(0, curate_len, 3):
        if i == 0:
            if seq[i:i + 3] in START_CODON:
                last_state = 3
                if is_training:
                    matrix[0][1] += 1
                    matrix[1][2] += 1
                    matrix[2][3] += 1
                else:
                    p += math.log(matrix[0][1]) + math.log(matrix[1][2]) + math.log(matrix[2][3])
            else:
                print("Invalid sequence: " + seq)
                break;
        else:
            for base in seq[i:i + 3]:
                new_state = base_state_plus(last_state, base)
                if is_training:
                    matrix[last_state][new_state] += 1
                else:
                    if matrix[last_state][new_state] == 0:
                        p += math.log(PSEUDO_PROB)
                    else:
                        p += math.log(matrix[last_state][new_state])
                last_state = new_state
        if seq[i:i + 3] in STOP_CODON and is_training:
            curate_len = i + 3
            break;
        
    if is_training:
        matrix[last_state][PLUS_END_STATE] += 1
        return matrix
    else:
        if matrix[last_state][PLUS_END_STATE] == 0:
            p += math.log(PSEUDO_PROB)
        else:
            p += math.log(matrix[last_state][PLUS_END_STATE])
        return p, curate_len

def traverse_minus_matrix(seq, matrix, is_training):
    p = 1
    last_state = 0
    curate_len = len(seq) - len(seq) % 3
    for i in range(0, curate_len, 3):
        for base in seq[i:i + 3]:
            new_state = base_state_minus(base)
            if is_training:
                matrix[last_state][new_state] += 1
            else:
                if matrix[last_state][new_state] == 0:
                    p += math.log(PSEUDO_PROB)
                else:
                    p += math.log(matrix[last_state][new_state])
            last_state = new_state
            
    if is_training:
        matrix[last_state][MINUS_END_STATE] += 1
        return matrix
    else:
        if matrix[last_state][MINUS_END_STATE] == 0:
            p += math.log(PSEUDO_PROB)
        else:
            p += math.log(matrix[last_state][MINUS_END_STATE])
        return p, curate_len
            
def base_state_plus(last_state, base):
    if last_state == 3 or last_state == 14 or last_state == 15 or last_state == 16 or  \
       last_state == 17 or last_state == 18 or last_state == 19:
        if base == 'A':
            return 4
        elif base == 'C':
            return 5
        elif base == 'G':
            return 6
        elif base == 'T':
            return 7
    elif last_state == 4 or last_state == 5 or last_state == 6:
        if base == 'A':
            return 8
        elif base == 'C':
            return 9
        elif base == 'G':
            return 10
        elif base == 'T':
            return 11
    elif last_state == 7:
        if base == 'A':
            return 12
        elif base == 'C':
            return 9
        elif base == 'G':
            return 13
        elif base == 'T':
            return 11
    elif last_state == 8 or last_state == 9 or last_state == 10 or last_state == 11:
        if base == 'A':
            return 14
        elif base == 'C':
            return 15
        elif base == 'G':
            return 16
        elif base == 'T':
            return 17
    elif last_state == 12:
        if base == 'A':
            return 18
        elif base == 'C':
            return 15
        elif base == 'G':
            return 19
        elif base == 'T':
            return 17
    elif last_state == 13:
        if base == 'A':
            return 18
        elif base == 'C':
            return 15
        elif base == 'G':
            return 16
        elif base == 'T':
            return 17
    print("Invalid base: " + base)
    return PLUS_END_STATE
    
def base_state_minus(base):
    if base == 'A':
        return 1
    elif base == 'C':
        return 2
    elif base == 'G':
        return 3
    elif base == 'T':
        return 4
    print("Invalid base: " + base)
    return MINUS_END_STATE

def log_ratio(file_name, prob_matrix_plus, prob_matrix_minus):
    ratio = []
    for seq in read_sequences(file_name):
        p_plus, seq_len = traverse_plus_matrix(seq, prob_matrix_plus, False)
        p_minus, seq_len = traverse_minus_matrix(seq, prob_matrix_minus, False)
        ratio.append((p_plus - p_minus) / seq_len)
    return ratio

def classify(ratio_list, threshold):
    plus_n = minus_n = 0
    for ratio in ratio_list:
        if ratio > threshold:
            plus_n += 1
        else:
            minus_n += 1
    return plus_n, minus_n

def analysis(tp, tn, fp, fn):
    result = {}
    # accuracy
    result['acc'] = (tp + tn) / (tp + fn + fp + tn)
    # sensitivity, recall
    result['sn'] = tp / (tp + fn)
    # specificity
    result['sp'] = tn / (tn + fp)
    # precision
    if not tp + fp == 0:
        result['pc'] = tp / (tp + fp)
    # f-measure
    result['f'] = 2 * tp / (2 * tp + fp + fn)
    return result

def main():
    ####################################
    # Define the data path
    ####################################
    train_plus_file = 'Data/train_plus.txt'
    train_minus_file = 'Data/train_minus.txt'
    test_plus_file = 'Data/test_plus.txt'
    test_minus_file = 'Data/test_minus.txt'
    
    ####################################
    # Training plus data
    ####################################
    print(">>>Training plus data...")
    state_matrix_plus = init_matrix(PLUS_END_STATE + 1)
    for seq in read_sequences(train_plus_file):
        state_matrix_plus = traverse_plus_matrix(seq, state_matrix_plus, True)
    print("State transition matrix for plus data:")
    print_matrix(state_matrix_plus)
    print("Probability matrix for plus data:")
    prob_matrix_plus = trans_matrix_to_prob(state_matrix_plus)
    print_matrix(prob_matrix_plus)
    
    ####################################
    # Training minus data
    ####################################
    print(">>>Training minus data...")
    state_matrix_minus = init_matrix(MINUS_END_STATE + 1)
    for seq in read_sequences(train_minus_file):
        state_matrix_minus = traverse_minus_matrix(seq, state_matrix_minus, True)
    print("State transition matrix for minus data:")
    print_matrix(state_matrix_minus)
    print("Probability matrix for minus data:")
    prob_matrix_minus = trans_matrix_to_prob(state_matrix_minus)
    print_matrix(prob_matrix_minus)
    
    ####################################
    # Performance benchmark
    ####################################
    print('>>>Performance benchmarking...')
    ratio_train_plus = log_ratio(train_plus_file, prob_matrix_plus, prob_matrix_minus)
    ratio_train_minus = log_ratio(train_minus_file, prob_matrix_plus, prob_matrix_minus)
    
    # print("threshold\tsn\t1 - sp\taccuracy")
    ts = []
    for r in ratio_train_plus:
        if not r in ts:
            ts.append(r)
    for r in ratio_train_minus:
        if not r in ts:
            ts.append(r)
    ts.sort()
    cut_off = 0;
    for t in ts:
        tp, fn = classify(ratio_train_plus, t)
        fp, tn = classify(ratio_train_minus, t)
        result = analysis(tp, tn, fp, fn)
        # print(str(t) + "\t" + str(result['sn']) + "\t" + str(1 - result['sp']) + "\t" + str(result['acc']))
        if result['sn'] + result['sp'] > cut_off:
            cut_off = result['sn'] + result['sp']
            THRESHOLD = t
            
    print("Threshold = " + str(THRESHOLD))
    
    ####################################
    # Evaluation on training data
    ####################################
    print(">>>Evaluation on training data...")
    tp, fn = classify(ratio_train_plus, THRESHOLD)
    fp, tn = classify(ratio_train_minus, THRESHOLD)
    print(' \tpredicted as positive\tprediceted as negative')
    print('positive\t' + str(tp) + '\t' + str(fn))
    print('negative\t' + str(fp) + '\t' + str(tn))
    result = analysis(tp, tn, fp, fn)
    print('Accuracy: ' + str(result['acc']))
    print('Sensitivity: ' + str(result['sn']))
    print('Specificity: ' + str(result['sp']))
    print('F-measure: ' + str(result['f']))
    
    ####################################
    # Evaluation on testing data
    ####################################
    print(">>>Evaluation on testing data...")
    ratio_test_plus = log_ratio(test_plus_file, prob_matrix_plus, prob_matrix_minus)
    ratio_test_minus = log_ratio(test_minus_file, prob_matrix_plus, prob_matrix_minus)
    
    tp, fn = classify(ratio_test_plus, THRESHOLD)
    fp, tn = classify(ratio_test_minus, THRESHOLD)
    print(' \tpredicted as positive\tprediceted as negative')
    print('positive\t' + str(tp) + '\t' + str(fn))
    print('negative\t' + str(fp) + '\t' + str(tn))
    result = analysis(tp, tn, fp, fn)
    print('Accuracy: ' + str(result['acc']))
    print('Sensitivity: ' + str(result['sn']))
    print('Specificity: ' + str(result['sp']))
    print('F-measure: ' + str(result['f']))
    
if __name__ == '__main__':
    main()
