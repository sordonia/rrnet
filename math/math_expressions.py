from random import random, randint, choice
import sys
import pickle

import pdb


class Expression:
    pass


class Number(Expression):
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)


class BinaryExpression(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return "(" + str(self.left) + self.op + str(self.right) + ")"


def randomExpression(prob):
    p = random()
    if p > prob:
        return Number(randint(0, 9)), 0
    else:
        left, leftdepth = randomExpression(prob / 1.2)
        op = choice(["+", "-", "*"])
        right, rightdepth = randomExpression(prob / 1.2)
        return BinaryExpression(left, op, right), max(leftdepth, rightdepth) + 1


def randomExpressionFixedLength(length):
    if length == 0:
        # return Number(randint(10, 99)), 0
        return Number(randint(0, 9)), 0
    else:
        leftlen = randint(0, length - 1)
        rightlen = length - leftlen -1
        left, leftdepth = randomExpressionFixedLength(leftlen)
        op = choice(["+", "-", "*", "%"])
        right, rightdepth = randomExpressionFixedLength(rightlen)
        return BinaryExpression(left, op, right), max(leftdepth, rightdepth) + 1


def generate(count, require_positive, anslen, explen=2, mindepth=0, maxdepth=0):
    if explen == 0:
        Expression = randomExpression
        explen = 1
    else:
        Expression = randomExpressionFixedLength
    if maxdepth == 0:
        maxdepth = explen

    exps = []
    i = 0
    while i < count:
        exp, depth = Expression(explen)
        exp = exp.__str__()
        try:
            expvalue = str(eval(exp))
        except ZeroDivisionError:
            continue

        is_negative = True if expvalue[0] == '-' else False
        if require_positive and is_negative:
            continue

        target_anslen = anslen + (1 if is_negative else 0)
        if len(expvalue) == target_anslen and depth <= maxdepth and depth >= mindepth:
            exps.append([exp, expvalue])
            i += 1

    return exps


#############Sample generation code###############
if __name__ == '__main__':
    size = int(sys.argv[1])
    require_positive = bool(sys.argv[2])
    anslen = int(sys.argv[3])
    explen = int(sys.argv[4])
    fname = str(sys.argv[5])

    # demo generation
    x = generate(10, require_positive, anslen, explen)
    for i in x:
        print("{} = {}".format(i[0], i[1]))


    dataset_dict = {}
    while len(dataset_dict) < 120000:
        x = generate(1, require_positive, anslen, explen)[0]
        str_x = "{} = {}".format(x[0], x[1])

        if not (str_x in dataset_dict):
            dataset_dict[str_x] = x

    key_list = list(dataset_dict.keys())

    f = open('math_' + str(explen) + 'op_mult_train.pkl', 'wb')
    data = [dataset_dict[k] for k in key_list[:100000]]
    pickle.dump(data, f)
    f.close()

    f = open('math_' + str(explen) + 'op_mult_valid.pkl', 'wb')
    data = [dataset_dict[k] for k in key_list[100000:110000]]
    pickle.dump(data, f)
    f.close()

    f = open('math_' + str(explen) + 'op_mult_test.pkl', 'wb')
    data = [dataset_dict[k] for k in key_list[110000:120000]]
    pickle.dump(data, f)
    f.close()

