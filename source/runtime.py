import os
import re
import nltk
import math
import torch
import string
import datetime
import time
import pickle
from copy import deepcopy
from scipy import stats
import csv
import sys
import _thread
from threading import Lock, Thread, Timer
import ctypes
import inspect
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from enum import Flag, auto
from numba import jit, njit, prange
from nltk.cluster import KMeansClusterer
from gensim.models import fasttext
from fuzzywuzzy import fuzz
import traceback
import collections
from paretoset import paretoset
from multiprocessing import cpu_count
import itertools
import functools
from scipy.special import logit, expit
import spacy
# from line_profiler import LineProfiler

from source import metainfo

class runtime:

    class regularpattern:

        nlp = spacy.load('en_core_web_lg')

        class matchway(Flag):
            startwith = auto()
            contain = auto()
            exact = auto()
            ner = auto()
            idf = auto()
            groups = auto()

        notunicodepattern = u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])"
        notunicodepattern_ex = u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a.*x%])"

        numberpattern = r'[+-]?([0-9]?[.])?[0-9]+'
        modelpattern = '^(([a-zA-Z]+[0-9-]+)|([0-9]+[a-zA-Z-]+))[a-zA-Z0-9-]*$'
        operatorpattern = r'[+-]?([+-]?([0-9]?[.])?[0-9]+[.*x%]{1})*([0-9]?[.])?[0-9]+'
        grouppatterns = [
            [None, matchway.idf, lambda x: runtime.regularpattern.grouppatterns_idf_para != None and len(x) >= 5 and runtime.regularpattern.grouppatterns_idf_para[x] == 1], \
            [None, matchway.idf, lambda x: runtime.regularpattern.grouppatterns_idf_para != None and len(x) >= 5 and runtime.regularpattern.grouppatterns_idf_para[x] == 2], \
        ]
        grouppatterns_cooccur_veto = [0, 1]
        grouppatterns_cooccur_veto_ground = {0:1}
        for eachgroupindex in range(0, len(grouppatterns_cooccur_veto)):
            grouppatterns_cooccur_veto[eachgroupindex] = 'w2group_' + str(grouppatterns_cooccur_veto[eachgroupindex])
        for eachgroup in dict(grouppatterns_cooccur_veto_ground):
            grouppatterns_cooccur_veto_ground['w2group_' + str(eachgroup)] = 'w2group_' + str(grouppatterns_cooccur_veto_ground[eachgroup])
            del grouppatterns_cooccur_veto_ground[eachgroup]
        grouppatterns_model_para = set(['DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY', 'CARDINAL'])
        grouppatterns_idf_para = None
        keytoken_extract = [
            [['NORP', 'GPE', 'LOC', 'PERSON', 'PRODUCT'], matchway.ner], \
            [['DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY', 'CARDINAL'], matchway.ner], \
            [modelpattern, matchway.exact, lambda x: len(x) >= 8 and len(set([r.label_ for r in runtime.regularpattern.nlp(x, disable=['tagger', 'parser']).ents]).intersection(runtime.regularpattern.grouppatterns_model_para)) == 0], \
            [modelpattern, matchway.exact, lambda x: len(x) >= 6 and len(set([r.label_ for r in runtime.regularpattern.nlp(x, disable=['tagger', 'parser']).ents]).intersection(runtime.regularpattern.grouppatterns_model_para)) == 0]
        ]

        @staticmethod
        def index(varname):
            if type(varname) == str:
                varname = varname.split(',')
                thepattern = eval('runtime.regularpattern.' + varname[0])
                thematchway = eval('runtime.regularpattern.matchway.' + varname[1])
                return thepattern, thematchway
            else:
                return None

        @staticmethod
        def ispattern(string, pattern, matchway, bool = False):
            if type(string) != str:
                return False
            result = False
            if matchway == runtime.regularpattern.matchway.groups:
                for eachgroupindex in range(0, len(pattern)):
                    currentresult = runtime.regularpattern.ispattern(string, pattern[eachgroupindex][0], pattern[eachgroupindex][1], bool)
                    if currentresult != False and \
                        (len(pattern[eachgroupindex]) <= 2 or \
                         len(pattern[eachgroupindex]) >= 3 and pattern[eachgroupindex][2](string) == True):
                        result = str(eachgroupindex)
                        break
            elif matchway == runtime.regularpattern.matchway.ner:
                ners = runtime.regularpattern.nlp(string, disable=['tagger', 'parser'])
                ents = ners.ents
                if len(ents) == 0:
                    return False
                else:
                    assert(len(ents) == 1)
                    for ent in ents:
                        start, end, label = ent.start, ent.end, ent.label_
                        if label in pattern:
                            return True
                        else:
                            return False
            elif matchway == runtime.regularpattern.matchway.idf:
                return True
            else:
                if matchway == runtime.regularpattern.matchway.contain:
                    contains = re.findall(pattern, string)
                    if len(contains) > 0:
                        result = contains
                    else:
                        result = False
                elif matchway == runtime.regularpattern.matchway.exact:
                    if pattern[0] != '^':
                        pattern = '^' + pattern
                    if pattern[-1] != '$':
                        pattern = pattern + '$'
                    thestring = re.search(pattern, string)
                    if thestring != None and thestring.string == string:
                        result = True
                    else:
                        result = False
                elif matchway == runtime.regularpattern.matchway.startwith:
                    if pattern[0] != '^':
                        pattern = '^' + pattern
                    if pattern[-1] == '$':
                        pattern = pattern[0:len(pattern) - 1]
                    pattern = re.compile(pattern)
                    startwith = pattern.match(string)
                    if startwith:
                        result = startwith.group()
                    else:
                        result = False
            if bool == True:
                return result != False
            else:
                return result

    class deepmatcherlinkage:

        @staticmethod
        def combine(table1filename, table2filename, pairsfilename, dmpairsfilename):
            csvpairsfile = open(pairsfilename, "r", encoding='ISO-8859-1')
            readerpairs = csv.reader(csvpairsfile)
            hpairs = next(readerpairs)
            csvtable1file = open(table1filename, "r", encoding='ISO-8859-1')
            readertable1 = csv.reader(csvtable1file)
            htable1 = next(readertable1)
            csvtable2file = open(table2filename, "r", encoding='ISO-8859-1')
            readertable2 = csv.reader(csvtable2file)
            htable2 = next(readertable2)
            leng = len(htable1) - 1
            for i in range(leng):
                htable1[i + 1] = 'left_' + htable1[i + 1]
                htable2[i + 1] = 'right_' + htable2[i + 1]
            df = pd.DataFrame(columns=["label"] + htable1[1:] + htable2[1:])
            e1 = []
            for item1 in readertable1:
                e1.append(item1)
            e2 = []
            for item2 in readertable2:
                e2.append(item2)
            progress = 0
            for item3 in readerpairs:
                data = []
                id = item3[0]
                label = item3[1]
                data.append(label)
                id1 = id.split(",")[0]
                id2 = id.split(",")[1]
                for item1 in e1:
                    len1 = len(item1) - 1
                    if item1[0] == id1:
                        for i in range(len1):
                            data.append(item1[i + 1])
                        break
                for item2 in e2:
                    len2 = len(item2) - 1
                    if item2[0] == id2:
                        for i in range(len2):
                            data.append(item2[i + 1])
                        break
                df.loc[len(df)] = data
            csvpairsfile.close()
            df.index.name = 'id'
            df.to_csv(dmpairsfilename)

        @staticmethod
        def division_specifiedtrain(dmpairsfilename, pairsfilename, tabletrainname, trainvalidproportion, outtrainname, outvalidname, outtestname, givenlabel=False):
            csvtrainpairsfile = open(tabletrainname, "r", encoding='ISO-8859-1')
            readertrainpairs = csv.reader(csvtrainpairsfile)
            trainset = {}
            headtrainpairs = next(readertrainpairs)
            for itemtrainpair in readertrainpairs:
                csvpairsfile = open(pairsfilename, "r", encoding='ISO-8859-1')
                readerpairs = csv.reader(csvpairsfile)
                index = 0
                for itempair in readerpairs:
                    index += 1
                    if (itempair[0] == itemtrainpair[0]):
                        trainset[str(index - 2)] = int(itemtrainpair[1])
                        break
            csvpairsfile = open(dmpairsfilename, "r", encoding='ISO-8859-1')
            readerpairs = csv.reader(csvpairsfile)
            headpairs = next(readerpairs)
            train = pd.DataFrame(columns=headpairs)
            test = pd.DataFrame(columns=headpairs)
            for itempair in readerpairs:
                if itempair[0] in trainset:
                    if givenlabel == True:
                        itempair[1] = trainset[itempair[0]]
                    train.loc[len(train)] = itempair
                else:
                    test.loc[len(test)] = itempair
            # N = train.shape[0] - 1
            # train_n = int (N * a)
            # validation_n = N - train_n
            outtrain = train.sample(frac=trainvalidproportion, replace=False, axis=0)
            outvalidation = train.drop(labels=outtrain.axes[0])
            outtrain.to_csv(outtrainname, index=False)
            outvalidation.to_csv(outvalidname, index=False)
            test.to_csv(outtestname, index=False)

    class types:
        Lock = type(Lock())
        NoneType = type(None)

    class key_token_mining:

        KEY_TOKEN_MIN_LENGTH = 3

        def __init__(self, gml):
            self.gml = gml
            self.run()

        @staticmethod
        def tokenize_target_text(text):
            """
            This implementation is consistent with 'basic_feature_analysis.py'
            :param text:
            :return:
            """
            text = str(text).lower()
            if len(text) == 0 or text == 'nan' or text == 'n/a':
                return []
            tokens = nltk.word_tokenize(text)
            result = []
            for eachtoken in tokens:
                if metainfo.paras.nlpw2vgroups is not None and 'ex' in metainfo.paras.nlpw2vgroups:
                    eachtoken = re.sub(runtime.regularpattern.notunicodepattern_ex, '', eachtoken)
                else:
                    eachtoken = re.sub(runtime.regularpattern.notunicodepattern, '', eachtoken)
                if len(eachtoken) >= runtime.key_token_mining.KEY_TOKEN_MIN_LENGTH:
                    result.append(eachtoken)
            return result

        @staticmethod
        def entity_target_text(text):
            text = str(text).lower()
            if len(text) == 0 or text == 'nan' or text == 'n/a':
                return []
            result = []
            for entity in text:
                normalized_entity = ' '.join(runtime.key_token_mining.tokenize_target_text(entity))
                result.append(normalized_entity)
            return result

        @staticmethod
        def process_text(text):
            if type(text) == str:
                return runtime.key_token_mining.tokenize_target_text(text)
            else:
                # entity list or set
                return runtime.key_token_mining.entity_target_text(text)

        def tokenize_target_attribute(self, target_attribute):
            """
            :param target_attribute: str
            :param entity_splitter: If not None, process text to entities.
            :return:
            """
            attr_values = []
            # data set #1
            attr_index1 = self.gml.data1.columns.get_loc(target_attribute)
            for eachrecordid in self.gml.records:
                recordtext = self.gml.records[eachrecordid]
                attr_values.append(runtime.key_token_mining.process_text(recordtext[attr_index1]))
            return attr_values

        @staticmethod
        def cal_token_idf(attribute_values):
            """
            calculate Inverse document frequency.
            ref: https://en.wikipedia.org/wiki/Tf-idf
            :param attribute_values: [[...], ...]
            :return:
            """
            token_2_idf = dict()
            token_2_freq = dict()
            if attribute_values is None:
                return token_2_idf
            token_2_docs = dict()
            docs_len = len(attribute_values)
            for i in range(docs_len):
                values = set(attribute_values[i])
                for token in values:
                    token_2_docs.setdefault(token, set())
                    token_2_docs[token].add(i)
            for eachtoken, v in token_2_docs.items():
                inv_doc_fre = len(token_2_docs.get(eachtoken)) + 1.0
                idf = np.log(1.0 * docs_len / inv_doc_fre)
                token_2_idf[eachtoken] = idf
                freq = len(token_2_docs.get(eachtoken))
                token_2_freq[eachtoken] = freq
            return token_2_idf, token_2_freq

        def mining_key_tokens(self, target_attribute_index):
            """
            Select tokens with high IDF values.
            :param target_attribute:
            :param top_percent:
            :param qualified_threshold: Tokens that appear less than qualified_threshold will be selected.
            :return:
            """
            target_attribute = self.gml.RecordAttributes[target_attribute_index]
            bottom_percent = 0
            low_qualified_threshold = 1
            top_percent = self.gml.data.infer_keytoken.idfrange
            qualified_threshold = self.gml.data.infer_keytoken.freqrange
            pattern_specified = runtime.regularpattern.index(self.gml.data.infer_keytoken.patternrestrict)
            if type(top_percent) == list:
                top_percent = top_percent[target_attribute_index]
            if type(top_percent) == list:
                bottom_percent = top_percent[0]
                top_percent = top_percent[1]
            if type(qualified_threshold) == list:
                qualified_threshold = qualified_threshold[target_attribute_index]
            if type(qualified_threshold) == list:
                low_qualified_threshold = qualified_threshold[0]
                qualified_threshold = qualified_threshold[1]
            column_name = '{} ~ {}, {} ~ {}, {}, {}'.format(bottom_percent, top_percent, low_qualified_threshold, qualified_threshold, self.gml.data.infer_keytoken.patternrestrict, self.gml.data.infer_keytoken.nlpw2vgroups)
            key_token_path = self.gml.processpath + "infer_keytoken_" + target_attribute + ".csv"
            if os.path.exists(key_token_path) == False and (top_percent == 0 and qualified_threshold == 0):
                return []
            elif os.path.exists(key_token_path):
                # Still preprocesscached_keytokens when (top_percent > 0 or qualified_threshold > 0) but pattern != None makes an empty keytoken csv cached.
                kt_pd = pd.read_csv(key_token_path, dtype=str, encoding="utf-8")
                if column_name == kt_pd.columns.tolist()[0]:
                    key_tokens = list(kt_pd.values.reshape(-1))
                    return key_tokens
            self.gml.preprocesscached_keytokens = False
            key_tokens = []
            if top_percent > 0 or qualified_threshold > 0:
                print("idf percent: {} ~ {}".format(bottom_percent, top_percent))
                print("qualified threshold: {} ~ {}".format(low_qualified_threshold, qualified_threshold))
                print("patternrestrict: {}: {}".format(self.gml.data.infer_keytoken.patternrestrict, pattern_specified))
                print("nlpw2vgroups: {}".format(self.gml.data.infer_keytoken.nlpw2vgroups))
                attr_2_tokens = self.tokenize_target_attribute(target_attribute)
                token_2_idf_values, token_2_freq_values = runtime.key_token_mining.cal_token_idf(attr_2_tokens)
                all_tokens_num = len(token_2_idf_values)
                bottom_percent_conform_countindex = np.maximum(int(bottom_percent * all_tokens_num), 1)
                top_percent_conform_countindex = np.maximum(int(top_percent * all_tokens_num), 1)
                idf_descending = sorted(token_2_idf_values.items(), key=lambda item: item[1], reverse=True)
                selected_number = 0
                selected_index = 0
                first_index = None
                last_index = None
                first_value = idf_descending[bottom_percent_conform_countindex - 1][1]
                last_value = idf_descending[top_percent_conform_countindex - 1][1]
                first_freq = token_2_freq_values[idf_descending[bottom_percent_conform_countindex - 1][0]]
                last_freq = token_2_freq_values[idf_descending[top_percent_conform_countindex - 1][0]]
                # Minimum IDF value.
                max_threshold = np.log(1.0 * len(attr_2_tokens) / (low_qualified_threshold + 1))
                min_threshold = np.log(1.0 * len(attr_2_tokens) / (qualified_threshold + 1))
                while selected_index < all_tokens_num:
                    if top_percent > 0 and (first_value >= idf_descending[selected_index][1] and last_value <= idf_descending[selected_index][1]) or (max_threshold >= idf_descending[selected_index][1] and min_threshold <= idf_descending[selected_index][1]):
                        if first_index == None:
                            first_index = selected_index
                        last_index = selected_index
                        key_tokens.append(idf_descending[selected_index][0])
                        selected_number += 1
                    selected_index += 1
                if runtime.isNone(self.gml.data.infer_keytoken.nlpw2vgroups) == False and 'sparse' in self.gml.data.infer_keytoken.nlpw2vgroups and pattern_specified != None:
                    for each_key_token in list(key_tokens):
                        if runtime.regularpattern.ispattern(each_key_token, pattern_specified[0], pattern_specified[1], True) == False:
                            key_tokens.remove(each_key_token)
                if metainfo.runningflags.refresh_cache == True:
                    kt_pd = pd.DataFrame(key_tokens, columns=[column_name])
                    kt_pd.to_csv(key_token_path, index=False, encoding="utf-8")
                print("# of key tokens: {} / {}, saving file path: {}".format(len(key_tokens), all_tokens_num, key_token_path))
                print("idf indexes from {} to {}".format(first_index, last_index))
                print("idf qualified threshold from {} to {}".format(first_freq, last_freq))
                print("patternrestrict: {}: {}".format(self.gml.data.infer_keytoken.patternrestrict, pattern_specified))
                print("nlpw2vgroups: {}".format(self.gml.data.infer_keytoken.nlpw2vgroups))
            return key_tokens

        def run(self):
            break_point = sys.stdout
            buffering_size = 1  # line buffering (ref: https://docs.python.org/3/library/functions.html#open)
            out_file = None
            if metainfo.runningflags.refresh_cache == True:
                out_file = open(self.gml.processpath + "infer_keytoken.txt", 'w', buffering_size, encoding='utf-8')
            else:
                out_file = open(self.gml.processpath + "infer_keytoken.txt", 'a+', buffering_size, encoding='utf-8')
            sys.stdout = out_file
            print("\n")
            time_info = '-' * 20 + str(datetime.datetime.fromtimestamp(time.time())) + '-' * 20
            print(time_info)
            out_file.flush()
            for target_attribute_index in range(1, len(self.gml.RecordAttributes)):
                print("\n")
                key_tokens = self.mining_key_tokens(target_attribute_index)
                self.gml.infer_keytoken[target_attribute_index] = key_tokens
                print("# of key tokens: {}".format(len(key_tokens)))
                print(type(key_tokens))
                print(key_tokens[:10])
                print("\n")
                out_file.flush()
            print('-' * len(time_info))
            print("\n")
            out_file.flush()
            out_file.close()
            sys.stdout = break_point

    @staticmethod
    def process(string1, tostring = False):

        def prepreprocess(string1):
            strstring1 = str(string1).lower()
            if len(strstring1) == 0 or strstring1 == 'nan' or strstring1 == 'n/a':
                return ''
            else:
                try:
                    string1 = float(string1)
                    if string1 == round(string1):
                        string1 = int(string1)
                    else:
                        string1 = runtime.round(string1)
                    return string1
                except:
                    return strstring1

        def preprocess(string1):
            if isinstance(string1, str):
                string1tokens = nltk.word_tokenize(string1)
                string1 = ''
                for eachtoken in string1tokens:
                    if metainfo.paras.nlpw2vgroups is not None and 'ex' in metainfo.paras.nlpw2vgroups:
                        eachtoken = re.sub(runtime.regularpattern.notunicodepattern_ex, '', eachtoken)
                    else:
                        eachtoken = re.sub(runtime.regularpattern.notunicodepattern, '', eachtoken)
                    if len(eachtoken) > 0:
                        if len(string1) > 0:
                            string1 += ' '
                        string1 += eachtoken
                multispacestring1 = str(string1)
                string1 = ''
                for eachtoken in multispacestring1.split():
                    if len(string1) > 0:
                        string1 += ' '
                    string1 += eachtoken
                abbrstring1 = ''
                if len(string1) > 0:
                    string1tokens = string1.split(' ')
                    for index in range(0, len(string1tokens)):
                        current1 = string1tokens[index]
                        if runtime.regularpattern.ispattern(current1, runtime.regularpattern.numberpattern, runtime.regularpattern.matchway.contain, True):
                            abbrstring1 += current1
                        else:
                            abbrstring1 += current1[0]
                return string1, abbrstring1
            else:
                return string1, string1

        if tostring == True:
            return str(preprocess(prepreprocess(string1))[0])
        else:
            return preprocess(prepreprocess(string1))[0]

    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    @staticmethod
    def find_lcsubstr(s1, s2):
        s1 = s1.lower()
        s2 = s2.lower()
        s1s = nltk.word_tokenize(s1.translate(runtime.remove_punctuation_map))
        s2s = nltk.word_tokenize(s2.translate(runtime.remove_punctuation_map))
        m = [[0 for i in range(len(s2s) + 1)] for j in range(len(s1s) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        mmax = 0  # 最长匹配的长度
        p = 0  # 最长匹配对应在s1中的最后一位
        for i in range(len(s1s)):
            for j in range(len(s2s)):
                if s1s[i] == s2s[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        maxcontinualwords = s1s[p - mmax:p]
        maxcontinualtokensrelativelength = float(len(maxcontinualwords)) / max(len(s1s), len(s2s))
        return len(maxcontinualwords), maxcontinualtokensrelativelength  # 返回最长子串及其长度

    @staticmethod
    def round(value):
        if type(value) == list or type(value) == np.ndarray:
            roundedvalue = []
            for index in range(0, len(value)):
                roundedvalue.append(runtime.round(value[index]))
            return roundedvalue
        else:
            if value == None or value == metainfo.top.NOT_AVAILABLE:
                return metainfo.top.NOT_AVAILABLE
            else:
                if type(value) == np.float64 or type(value) == np.float32 or type(value) == float:
                    return round(value, metainfo.paras.rounddigits)
                else:
                    return value

    @staticmethod
    def isnumber(x):
        if type(x) == np.float64 or type(x) == np.float32 or type(x) == float or type(x) == complex or type(x) == int:
            return True
        else:
            return False

    @staticmethod
    def enum(**enums):
        return type('Enum', (), enums)

    class linearregression:
        n_job = 20
        delta = None
        themetafeature = None
        regression = None
        residual = None
        meanX = None
        variance = None
        X = None
        Y = None
        BalanceWeightY0Count = None
        BalanceWeightY1Count = None
        HardLabelEnhance0 = None
        HardLabelEnhance1 = None
        N = None
        k = None
        b = None
        monotonyeffective = None
        effectivetrainingcount = 2
        polarenforce = None
        variablebound = None
        updatecache = None
        regupdatecallback = None

        def voidfunction(self):
            pass

        # The Linear Regression doesn't regress on Probability of Unit Aera Spliting,
        # instead, it regresses on the maximum weight values of all point instances,
        # to avoid a naive residual loss w.r.t samples count and constant-0-weight line indirect.
        # The highest confidence value is located in the intuitive average feature value and also the most naive weight value,
        # to conservatively avoid any early risk and better to believe those points with intuitive certainty.

        def __init__(self, themetafeature, XY, polarenforce, variablebound, regupdatecallback = None):
            self.themetafeature = themetafeature
            self.polarenforce = polarenforce
            self.variablebound = variablebound
            self.regupdatecallback = regupdatecallback
            if self.regupdatecallback == None:
                self.regupdatecallback = self.voidfunction
            self.delta = metainfo.paras.regressiondelta
            self.updatecache = 0
            self.effectivetrainingcount = runtime.linearregression.effectivetrainingcount
            if len(XY) > 0:
                XY = np.array(list(XY))
                self.X = XY[:, 0].reshape(-1, 1)
                self.Y = XY[:, 1].reshape(-1, 1)
            else:
                self.X = np.array([]).reshape(-1, 1)
                self.Y = np.array([]).reshape(-1, 1)
            self.BalanceWeightY0Count = 0
            self.BalanceWeightY1Count = 0
            self.HardLabelEnhance0 = []
            self.HardLabelEnhance1 = []
            for y in self.Y:
                if y > 0:
                    self.BalanceWeightY1Count += 1
                else:
                    self.BalanceWeightY0Count += 1
            self.performregression()

        def append(self, appendx, appendy, hardlabel):
            self.X = np.append(self.X, [[appendx]], axis=0)
            self.Y = np.append(self.Y, [[appendy]], axis=0)
            if appendy >= 0:
                self.BalanceWeightY1Count += 1
                if hardlabel == True:
                    self.HardLabelEnhance1.append(appendx)
            else:
                self.BalanceWeightY0Count += 1
                if hardlabel == True:
                    self.HardLabelEnhance0.append(appendx)

        def disable(self, delx, dely):
            for index in range(0, len(self.X)):
                if self.X[index][0] == delx and self.Y[index][0] == dely:
                    self.X = np.delete(self.X, index, axis=0)
                    self.Y = np.delete(self.Y, index, axis=0)
                    if dely > 0:
                        self.BalanceWeightY1Count -= 1
                    else:
                        self.BalanceWeightY0Count -= 1
                    break
            self.performregression()

        def monotonycheck(self):
            if self.regression == None or self.k < 0:
                self.monotonyeffective = False
            else:
                self.monotonyeffective = True

        def performregression(self):
            self.N = np.size(self.X)
            if self.themetafeature != metainfo.top.SIFT and self.N <= self.effectivetrainingcount:
                self.regression = None
                self.residual = None
                self.meanX = None
                self.variance = None
                self.k = None
                self.b = None
            else:
                if self.themetafeature != metainfo.top.SIFT and self.updatecache > 0:
                    self.updatecache -= 1
                else:
                    SampleWeightlist = []
                    SampleWeight = None
                    if len(np.unique(self.X)) == 1 or self.BalanceWeightY1Count == 0 or self.BalanceWeightY0Count == 0:
                        SampleWeight = 1
                    else:
                        SampleWeight = float(self.BalanceWeightY0Count) / self.BalanceWeightY1Count
                    HardLabelEnhanced0 = list(self.HardLabelEnhance0)
                    HardLabelEnhanced1 = list(self.HardLabelEnhance1)
                    for eachindex in range(0, self.N):
                        eachx = self.X[eachindex][0]
                        eachy = self.Y[eachindex][0]
                        if eachy >= 0:
                            if eachx in HardLabelEnhanced1:
                                SampleWeightlist.append(math.pow(metainfo.paras.hard_label_learn_enhance_multiplier, metainfo.paras.hard_label_learn_enhance_multiplier_coefficient) * math.pow(SampleWeight, metainfo.paras.class_weight_multiplier_coefficient))
                                HardLabelEnhanced1.remove(eachx)
                            else:
                                SampleWeightlist.append(SampleWeight)
                        else:
                            if eachx in HardLabelEnhanced0:
                                SampleWeightlist.append(math.pow(metainfo.paras.hard_label_learn_enhance_multiplier, metainfo.paras.hard_label_learn_enhance_multiplier_coefficient) * math.pow(1, metainfo.paras.class_weight_multiplier_coefficient))
                                HardLabelEnhanced0.remove(eachx)
                            else:
                                SampleWeightlist.append(1)
                    assert(len(HardLabelEnhanced0) == 0 and len(HardLabelEnhanced1) == 0)
                    self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=runtime.linearregression.n_job).fit(self.X, self.Y, sample_weight=SampleWeightlist)
                    self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
                    self.meanX = np.mean(self.X)
                    self.variance = np.sum((self.X - self.meanX) ** 2)
                    self.k = self.regression.coef_[0][0]
                    self.b = self.regression.intercept_[0]
                    self.monotonycheck()
                    self.updatecache = min(metainfo.paras.updatecache_abscapacity, int(metainfo.paras.updatecache_proportion * self.N))
                    self.regupdatecallback()

        def predictconfidence(self, x0):
            evidentialsupport = None
            espredict = None
            if self.regression != None and self.monotonyeffective == True:
                predict = self.regression.predict(np.array([x0]).reshape(-1, 1))[0][0]
                confidence = 1
                if self.residual > 0 and self.variance > 0:
                    tvalue = float(self.delta) / (self.residual * math.sqrt(1 + float(1) / self.N + math.pow(x0 - self.meanX, 2) / self.variance))
                    confidence = 1 - t.sf(tvalue, (self.N - 2)) * 2
                evidentialsupport = (1 + confidence)/2
                espredict = predict * evidentialsupport
                if self.polarenforce == 0:
                    espredict = min(espredict, 0)
                else:
                    if self.polarenforce == 1:
                        espredict = max(espredict, 0)
            else:
                confidence = 0
                evidentialsupport = (1 + confidence) / 2
                espredict = 0
            return evidentialsupport, espredict

    @staticmethod
    def consoleprogress(i, n, info):
        len_bar = 100
        if n == 0:
            return
        unit = int(n / 100)
        if unit == 0:
            unit = 1
        if i % unit == 0 or i == n:
            i = int(float(i) / n * 100)
            format_percent = None
            format_bar = None
            if i < 10:
                format_percent = "\r%d%%  "
            elif i < 100:
                format_percent = "\r%d%% "
            else:
                format_percent = "\r%d%%"
            format_bar = "%s%s%s%s%s%s"
            show_percent = format_percent % (i)
            len_blank = int((len_bar - len(info)) / 2)
            left_blank_process_pos = 0
            info_process_pos = 0
            right_blank_process_pos = 0
            if i <= len_blank:
                left_blank_process_pos = i
            elif i <= len_blank + len(info):
                left_blank_process_pos = len_blank
                info_process_pos = i - len_blank
            else:
                left_blank_process_pos = len_blank
                info_process_pos = len_blank + len(info)
                right_blank_process_pos = i - (len_blank + len(info))
            show_bar = format_bar % (runtime.console.color.BACKGROUND + runtime.console.color.DARKBLUE + " " * left_blank_process_pos,
                                     runtime.console.color.BACKGROUND + runtime.console.color.DARKCYAN + " " * (len_blank - left_blank_process_pos),
                                     runtime.console.color.BACKGROUND + runtime.console.color.DARKBLUE + info[0:info_process_pos],
                                     runtime.console.color.BACKGROUND + runtime.console.color.DARKCYAN + info[info_process_pos:len(info)],
                                     runtime.console.color.BACKGROUND + runtime.console.color.DARKBLUE + " " * right_blank_process_pos,
                                     runtime.console.color.BACKGROUND + runtime.console.color.DARKCYAN + " " * (len_blank - right_blank_process_pos))
            show = show_percent + '[' + show_bar + runtime.console.color.END + ']'
            sys.stdout.write(show)
            sys.stdout.flush()
            if i == 100:
                print()

    @staticmethod
    def pickledump(gml, structurename, flag):
        picklefile = gml.processpath + structurename + '.pkl'
        if flag == 'w':
            structure = eval('gml.' + structurename)
            output = open(picklefile, 'wb')
            pickle.dump(structure, output)
        else:
            if flag == 'r':
                output = open(picklefile, 'rb')
                exec('gml.' + structurename + ' = pickle.load(output)')
            else:
                if flag == 'e':
                    return os.path.exists(picklefile)

    class console:

        class color:
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
            BACKGROUND = '\033[7m'
            RED = '\033[91m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            BLUE = '\033[94m'
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            BLACK = '\033[30m'
            DARKRED = '\033[31m'
            DARKGREEN = '\033[32m'
            DARKYELLOW = '\033[33m'
            DARKBLUE = '\033[34m'
            DARKPURPLE = '\033[35m'
            DARKCYAN = '\033[36m'
            END = '\033[0m'
            COLORS = [RED, GREEN, YELLOW, BLUE, PURPLE, CYAN, BLACK, DARKRED, DARKGREEN, DARKYELLOW, DARKBLUE, DARKPURPLE, DARKCYAN]

        class styles(Flag):
            TOP = auto()
            INFO = auto()
            STRESS = auto()
            OUTLOOK = auto()
            REPORT = auto()
            CORRECTION = auto()
            EXCEPTION = auto()
            SIMPLE_CORRECTION = auto()
            SIMPLE_EXCEPTION = auto()
            PERIOD = auto()

        def __init__(self, title, content, style):
            exceptionframe = sys._getframe(1)
            functionname = exceptionframe.f_code.co_name
            functionlineno = exceptionframe.f_lineno
            header1 = None
            header2 = None
            header3 = None
            if style == runtime.console.styles.TOP:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.BOLD + runtime.console.color.RED
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.BOLD + runtime.console.color.DARKPURPLE
                header3 = runtime.console.color.END + runtime.console.color.BOLD + runtime.console.color.PURPLE
            elif style == runtime.console.styles.INFO:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKPURPLE
                header3 = runtime.console.color.END + runtime.console.color.BLACK
            elif style == runtime.console.styles.STRESS:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.PURPLE
                header3 = runtime.console.color.END + runtime.console.color.BLACK
            elif style == runtime.console.styles.OUTLOOK:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKCYAN
                header3 = runtime.console.color.END + runtime.console.color.BLACK
            elif style == runtime.console.styles.REPORT:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.BLUE
                header3 = runtime.console.color.END + runtime.console.color.BLACK
            elif style == runtime.console.styles.CORRECTION:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKGREEN
                header3 = runtime.console.color.END + runtime.console.color.DARKGREEN
            elif style == runtime.console.styles.EXCEPTION:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.RED
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKRED
                header3 = runtime.console.color.END + runtime.console.color.DARKRED
            elif style == runtime.console.styles.SIMPLE_CORRECTION:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.BLACK
                header3 = runtime.console.color.END + runtime.console.color.DARKGREEN
            elif style == runtime.console.styles.SIMPLE_EXCEPTION:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.DARKBLUE
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.BLACK
                header3 = runtime.console.color.END + runtime.console.color.DARKRED
            elif style == runtime.console.styles.PERIOD:
                header1 = runtime.console.color.END + runtime.console.color.UNDERLINE + runtime.console.color.BACKGROUND + runtime.console.color.DARKCYAN
                header2 = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKPURPLE
                header3 = runtime.console.color.END + runtime.console.color.BLACK
            if title != None:
                icon = ''
                title_lower = str(title).lower()
                if 'save' in title_lower:
                    icon = '💾'
                elif 'error' in title_lower or style == runtime.console.styles.EXCEPTION:
                    icon = '⚠'
                elif style == runtime.console.styles.TOP:
                    icon = '🎄🎊🔮🔮🔮'
                elif style == runtime.console.styles.PERIOD:
                    icon = '📁'
                else:
                    icon = '📧'
                showtext = str(title) + ' (@' + functionname + ', ' + str(functionlineno) + ')  █▓▒░'
                if style == runtime.console.styles.TOP:
                    showtext_unicodelist = list(showtext)
                    for eachindex in range(0, len(showtext_unicodelist)):
                        showtext_unicodelist[eachindex] = np.random.choice(runtime.console.color.COLORS, 1, False, None)[0] + showtext_unicodelist[eachindex]
                    showtext_unicodelist.append(header1)
                    showtext = ''.join(showtext_unicodelist)
                    print(header1 + icon + ' ' + showtext + ' ' * 10 + runtime.console.color.END)
                else:
                    print(header1 + icon + ' ' + showtext + ' ' * 10 + runtime.console.color.END)
            if content != None:
                if type(content) == dict:
                    splitmark = '  '
                    thecontent = ''
                    for eachcontentitemname in content:
                        contentvalue = content[eachcontentitemname]
                        if runtime.isnumber(contentvalue) == True and contentvalue != 0 and math.isnan(contentvalue) == False:
                            if contentvalue < 0:
                                contentvalue = contentvalue * (-1)
                            if contentvalue != 1:
                                for eachspecialvalue in metainfo.top.specialvalue:
                                    specialtag = metainfo.top.specialvalue[eachspecialvalue]
                                    if contentvalue == eachspecialvalue:
                                        contentvalue = specialtag
                                        break
                                    elif math.isinf(contentvalue) == True:
                                        contentvalue = '∞'
                                        break
                                    else:
                                        variation_contentvalue = int(math.log(contentvalue, eachspecialvalue))
                                        if math.pow(eachspecialvalue, variation_contentvalue) == contentvalue:
                                            contentvalue = specialtag + ' ^ ' + str(variation_contentvalue)
                                            break
                            if content[eachcontentitemname] < 0:
                                contentvalue = ' — ' + str(contentvalue)
                        str_eachcontentitemname = str(eachcontentitemname)
                        if type(contentvalue) == bool or 'polar' == str_eachcontentitemname:
                            if contentvalue == True:
                                header2_True = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKGREEN
                                header3_True = runtime.console.color.END + runtime.console.color.GREEN
                                thecontent += (header2_True + str_eachcontentitemname + ':' + header3_True + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                            else:
                                header2_False = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKRED
                                header3_False = runtime.console.color.END + runtime.console.color.RED
                                thecontent += (header2_False + str_eachcontentitemname + ':' + header3_False + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                        elif metainfo.top.GROUND_TRUTH == str_eachcontentitemname:
                            header2_GT = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKGREEN
                            header3_GT = runtime.console.color.END + runtime.console.color.DARKCYAN
                            thecontent += (header2_GT + str_eachcontentitemname + ':' + header3_GT + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                        elif 'weight' == str_eachcontentitemname:
                            header2_GT = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKPURPLE
                            header3_GT = runtime.console.color.END + runtime.console.color.PURPLE
                            thecontent += (header2_GT + str_eachcontentitemname + ':' + header3_GT + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                        elif '√' in str_eachcontentitemname:
                            header2_True = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKGREEN
                            header3_True = runtime.console.color.END + runtime.console.color.GREEN
                            thecontent += (header2_True + str_eachcontentitemname + ':' + header3_True + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                        elif '×' in str_eachcontentitemname:
                            header2_False = runtime.console.color.END + runtime.console.color.BACKGROUND + runtime.console.color.DARKRED
                            header3_False = runtime.console.color.END + runtime.console.color.RED
                            thecontent += (header2_False + str_eachcontentitemname + ':' + header3_False + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                        else:
                            thecontent += (header2 + str_eachcontentitemname + ':' + header3 + ' ' + str(contentvalue) + runtime.console.color.END + splitmark)
                    content = thecontent[0: len(thecontent) - len(splitmark)] + ' .'
                sys.stdout.write(header3 + str(content))
                print(runtime.console.color.END)
            sys.stdout.flush()

        @staticmethod
        def print(level, style, highlightindexes, * content):
            icon = None
            if level == 0:
                icon = runtime.console.color.BLUE + '💎 ' + runtime.console.color.END
            else:
                icon = '📝 '
            output = runtime.console.color.END + '  ' * level + icon
            for eachcontentindex in range(0, len(content)):
                currentoutput = runtime.console.color.END
                if highlightindexes != None and eachcontentindex in highlightindexes:
                    if style == runtime.console.styles.INFO:
                        currentoutput = runtime.console.color.END + runtime.console.color.DARKPURPLE
                    else:
                        if style == runtime.console.styles.STRESS:
                            currentoutput = runtime.console.color.END + runtime.console.color.YELLOW
                        else:
                            if style == runtime.console.styles.OUTLOOK:
                                currentoutput = runtime.console.color.END + runtime.console.color.DARKCYAN
                            else:
                                if style == runtime.console.styles.REPORT:
                                    currentoutput = runtime.console.color.END + runtime.console.color.BLUE
                                else:
                                    if style == runtime.console.styles.CORRECTION:
                                        currentoutput = runtime.console.color.END + runtime.console.color.DARKGREEN
                                    else:
                                        if style == runtime.console.styles.EXCEPTION:
                                            currentoutput = runtime.console.color.END + runtime.console.color.RED
                currentoutput += (str(content[eachcontentindex]) + ' ')
                output += currentoutput
            output += runtime.console.color.END
            print(output)

    @staticmethod
    def uniforminterval(intervalcount):
        allintervals = []
        step = float(1) / intervalcount
        previousleft = None
        previousright = 0
        currentleft = None
        currentright = None
        for intervalindex in range(0, intervalcount):
            interval = []
            currentleft = previousright
            currentright = currentleft + step
            if intervalindex == intervalcount - 1:
                currentright = 1 + metainfo.top.SMALL_VALUE
            previousleft = currentleft
            previousright = currentright
            allintervals.append([currentleft, currentright])
        return allintervals

    @staticmethod
    def entropy(probability):
        if type(probability) == np.float64 or type(probability) == np.float32 or type(probability) == float or type(probability) == int:
            if math.isinf(probability) == True:
                return probability
            else:
                if probability <= 0 or probability >= 1:
                    return 0
                else:
                    return 0 - (probability * math.log(probability, 2) + (1 - probability) * math.log((1 - probability), 2))
        else:
            if type(probability) == list:
                entropyoflist = []
                for eachprobability in probability:
                    entropyoflist.append(runtime.entropy(eachprobability))
                return entropyoflist
            else:
                return None

    @staticmethod
    def isnan(x):
        return np.isnan(x) or math.isnan(x)

    @staticmethod
    def probabilitypolar(probability):
        if type(probability) == np.float64 or type(probability) == np.float32 or type(probability) == float or type(probability) == int:
            if probability >= 0.5:
                return 1
            else:
                if probability < 0.5:
                    return 0
                else:
                    if runtime.isnan(probability) == True:
                        return math.nan
        else:
            return None

    @staticmethod
    def weightresultcorrect(finalweight, inpair, tolabeljudge):
        from source import fg
        ruleweight = fg.pair.inferenceresult.NOT_AVAILABLE
        GMLweight = None
        ruleresult = fg.pair.inferenceresult.NOT_AVAILABLE
        finalresult = None
        if finalweight >= 0 and inpair.truthlabel == 1 or finalweight < 0 and inpair.truthlabel == 0:
            finalresult = True
        else:
            finalresult = False
        if len(inpair.rules) > 0:
            ruleweight = inpair.ruleweight(detailed=tolabeljudge)
            if tolabeljudge == True:
                for eachrule in ruleweight:
                    if inpair in eachrule.truth_correcting_correct:
                        if finalresult == True:
                            eachrule.actual_correcting[1][0] += 1
                        elif finalresult == False:
                            eachrule.actual_correcting[1][1] -= 1
                    elif inpair in eachrule.truth_correcting_misjudge:
                        if finalresult == True:
                            eachrule.actual_correcting[2][0] += 1
                        elif finalresult == False:
                            eachrule.actual_correcting[2][1] -= 1
                    eachrule.actual_correcting[0] = eachrule.actual_correcting[1][0] + eachrule.actual_correcting[2][1]
                ruleweight = sum(ruleweight.values())
            GMLweight = finalweight - ruleweight
            if ruleweight >= 0 and inpair.truthlabel == 1 or ruleweight < 0 and inpair.truthlabel == 0:
                ruleresult = True
            else:
                ruleresult = False
        else:
            GMLweight = finalweight
        GMLresult = None
        if GMLweight >= 0 and inpair.truthlabel == 1 or GMLweight < 0 and inpair.truthlabel == 0:
            GMLresult = True
        else:
            GMLresult = False
        inferenceresult = None
        if ruleresult == fg.pair.inferenceresult.NOT_AVAILABLE:
            if finalresult == True:
                inferenceresult = fg.pair.inferenceresult.GMLONLY_RIGHT
            else:
                inferenceresult = fg.pair.inferenceresult.GMLONLY_WRONG
        else:
            if GMLresult == True and ruleresult == True:
                inferenceresult = fg.pair.inferenceresult.BOTH_RIGHT
            elif GMLresult == False and ruleresult == False:
                inferenceresult = fg.pair.inferenceresult.BOTH_WRONG
            elif finalresult == True:
                if GMLresult == False and ruleresult == True:
                    inferenceresult = fg.pair.inferenceresult.RULE_CORRECT
                elif GMLresult == True and ruleresult == False:
                    inferenceresult = fg.pair.inferenceresult.RULE_LEAN_MISJUDGE
            elif finalresult == False:
                if GMLresult == False and ruleresult == True:
                    inferenceresult = fg.pair.inferenceresult.RULE_LEAN_CORRECT
                elif GMLresult == True and ruleresult == False:
                    inferenceresult = fg.pair.inferenceresult.RULE_MISJUDGE
        return inferenceresult, ruleweight

    @staticmethod
    def sgmlresultcorrect(finalweight, inpair):
        from source import fg
        if inpair.ugmllabel == None:
            return fg.pair.inferenceresult.NOT_AVAILABLE
        else:
            finalresult = finalweight >= 0 and inpair.truthlabel == 1 or finalweight < 0 and inpair.truthlabel == 0
            ugmlresult = inpair.ugmllabel == inpair.truthlabel
            ruleresult = fg.pair.inferenceresult.NOT_AVAILABLE
            if len(inpair.rules) > 0:
                ruleweight = inpair.ruleweight(detailed=False)
                if ruleweight >= 0 and inpair.truthlabel == 1 or ruleweight < 0 and inpair.truthlabel == 0:
                    ruleresult = True
                else:
                    ruleresult = False
            inferenceresult = None
            if finalresult == True:
                if ugmlresult == True:
                    if inpair.truthlabel == 0:
                        inferenceresult = fg.pair.inferenceresult.USGML_BOTH_RIGHT_0
                    else:
                        inferenceresult = fg.pair.inferenceresult.USGML_BOTH_RIGHT_1
                else:
                    if inpair.truthlabel == 0:
                        if ruleresult == True:
                            inferenceresult = fg.pair.inferenceresult.SGML_RULE_CORRECT_0
                        elif ruleresult == None:
                            inferenceresult = fg.pair.inferenceresult.SGML_CORRECT_0
                        else:
                            inferenceresult = fg.pair.inferenceresult.SGML_CORRECT_0
                    else:
                        if ruleresult == True:
                            inferenceresult = fg.pair.inferenceresult.SGML_RULE_CORRECT_1
                        elif ruleresult == None:
                            inferenceresult = fg.pair.inferenceresult.SGML_CORRECT_1
                        else:
                            inferenceresult = fg.pair.inferenceresult.SGML_CORRECT_1
            else:
                if ugmlresult == True:
                    if inpair.truthlabel == 0:
                        if ruleresult == False:
                            inferenceresult = fg.pair.inferenceresult.SGML_RULE_MISJUDGE_0
                        elif ruleresult == None:
                            inferenceresult = fg.pair.inferenceresult.SGML_MISJUDGE_0
                        else:
                            inferenceresult = fg.pair.inferenceresult.SGML_MISJUDGE_0
                    else:
                        if ruleresult == False:
                            inferenceresult = fg.pair.inferenceresult.SGML_RULE_MISJUDGE_1
                        elif ruleresult == None:
                            inferenceresult = fg.pair.inferenceresult.SGML_MISJUDGE_1
                        else:
                            inferenceresult = fg.pair.inferenceresult.SGML_MISJUDGE_1
                else:
                    if inpair.truthlabel == 0:
                        inferenceresult = fg.pair.inferenceresult.USGML_BOTH_WRONG_0
                    else:
                        inferenceresult = fg.pair.inferenceresult.USGML_BOTH_WRONG_1
            return inferenceresult



    @staticmethod
    def sigmoid(weight):
        #i.e.  math.exp(weight)/(1+ math.exp(weight))
        #return float(1) / float(1 + math.exp((-1) * weight))
        return expit(weight)

    @staticmethod
    def weight2probabilityentropy(weight):
        probability = runtime.sigmoid(weight)
        entropy = runtime.entropy(probability)
        return weight, probability, entropy

    @staticmethod
    def sublist(thelist, theinds):
        sublist = []
        for eachind in theinds:
            sublist.append(thelist[eachind])
        return sublist

    @staticmethod
    def searchFile(searchpath, filenamepattern):
        def cmp_filename(a, b):
            if len(a) == len(b):
                return runtime.cmp(a, b)
            else:
                return runtime.cmp(len(a), len(b))
        matchedFile = []
        for root, dirs, files in os.walk(searchpath):
            for file in files:
                if re.match(filenamepattern, file):
                    fname = os.path.abspath(os.path.join(root, file))
                    matchedFile.append(fname)
        matchedFile.sort(key=functools.cmp_to_key(cmp_filename), reverse=False)
        return matchedFile

    class hashabledict(dict):
        def __key(self):
            return tuple((k, self[k]) for k in sorted(self))
        def __hash__(self):
            return hash(self.__key())
        def __eq__(self, other):
            return self.__key() == other.__key()

    class hashableset(set):
        def __hash__(self):
            return hash(frozenset(self))

    class hashablelist(list):
        def __hash__(self):
            return hash(frozenset(self))

    @staticmethod
    def completeset_redundant(values, polar):
        values = list(values)
        values.sort(reverse=False)
        if polar == 0:
            values = values[0: len(values) - 1]
        else:
            values = values[1: len(values)]
        return values

    @staticmethod
    def equaldistance_choice(thelist, choicenumber):
        if choicenumber == 1:
            return [thelist[int(len(thelist) / 2)]]
        else:
            start = 0
            stop = len(thelist) - 1
            choices = np.linspace(start, stop, num=choicenumber, endpoint=True, retstep=False, dtype=int)
            choicelist = list(set(np.array(thelist)[choices]))
            return choicelist

    @staticmethod
    def combine(form_values, all_form_values_values, polar, mutation_basepredicates, mutation):
        initindicator = runtime.hashabledict({str(metainfo.top.INDETERMINATE):metainfo.top.INDETERMINATE})
        previouscombinations = runtime.hashabledict({initindicator:all_form_values_values})
        combinations = None
        for eachform in form_values:
            combinations = runtime.hashabledict()
            eachformvalues = form_values[eachform]
            eachformvalues_completeset_redundant = runtime.completeset_redundant(eachformvalues.keys(), polar)
            mutation_rangevalue = None
            if mutation_basepredicates == None:
                mutation_rangevalue = [min(eachformvalues_completeset_redundant), max(eachformvalues_completeset_redundant)]
            else:
                for eachpredicate in mutation_basepredicates:
                    if eachpredicate.feature == eachform:
                        mutation_rangevalue = eachpredicate.mutationrange(mutation=mutation)
                        break
            for eachformvalue in eachformvalues.keys():
                if eachformvalue >= mutation_rangevalue[0] and eachformvalue <= mutation_rangevalue[1]:
                    if eachformvalue in eachformvalues_completeset_redundant:
                        eachformvalue_values = set()
                        if polar == 0:
                            for polaroperation_eachformvalue in eachformvalues:
                                if polaroperation_eachformvalue <= eachformvalue:
                                    eachformvalue_values.update(eachformvalues[polaroperation_eachformvalue])
                        else:
                            for polaroperation_eachformvalue in eachformvalues:
                                if polaroperation_eachformvalue >= eachformvalue:
                                    eachformvalue_values.update(eachformvalues[polaroperation_eachformvalue])
                        for eachpreviouscombination in previouscombinations:
                            eachextendcombination = runtime.hashabledict(eachpreviouscombination)
                            eachextendcombination[eachform] = eachformvalue
                            eachextendformvalue_values = set(previouscombinations[eachpreviouscombination]).intersection(eachformvalue_values)
                            combinations[eachextendcombination] = eachextendformvalue_values
                    else:
                        for eachpreviouscombination in previouscombinations:
                            eachextendcombination = runtime.hashabledict(eachpreviouscombination)
                            eachextendformvalue_values = set(previouscombinations[eachpreviouscombination])
                            combinations[eachextendcombination] = eachextendformvalue_values
            previouscombinations = combinations
        copycombinations = runtime.hashabledict(combinations)
        combinations = runtime.hashabledict()
        for eachcombination in copycombinations:
            thepairs = copycombinations[eachcombination]
            del eachcombination[str(metainfo.top.INDETERMINATE)]
            if len(eachcombination) > 0:
                combinations[eachcombination] = thepairs
        combinations_predicates = {}
        op = None
        if polar == 0:
            op = runtime.predicate.op.lesseq
        else:
            op = runtime.predicate.op.largereq
        for eachcombination in combinations:
            current_combinations_predicates = runtime.hashableset()
            for eachfeature in eachcombination:
                eachfeaturevalue = eachcombination[eachfeature]
                current_combinations_predicates.add(runtime.predicate(feature=eachfeature, op=op, value=eachfeaturevalue, valueex=None))
            combinations_predicates[current_combinations_predicates] = combinations[eachcombination]
        return combinations_predicates

    class predicate:

        class op:
            lesseq = '≤'
            largereq = '≥'
            eq = '='
            oppose = {lesseq:largereq, largereq:lesseq}

        def __init__(self, feature, op, value, valueex = None):
            self.feature = feature
            self.op = op
            self.value = value
            self.valueex = valueex
            if self.value == None:
                self.value = self.valueex
                self.valueex = None
            if self.valueex != None:
                if self.value == self.valueex:
                    self.op = runtime.predicate.op.eq
                    self.valueex = None
                else:
                    if self.op == runtime.predicate.op.largereq:
                        self.value = max(value, valueex)
                        self.valueex = min(value, valueex)
                    else:
                        self.value = min(value, valueex)
                        self.valueex = max(value, valueex)
            if runtime.isnumber(self.value):
                self.value = float(self.value)
            if runtime.isnumber(self.valueex):
                self.valueex = float(self.valueex)
            assert(self.value != metainfo.top.NONE_VALUE and self.valueex != metainfo.top.NONE_VALUE)

        def print(self):
            if self.valueex == None:
                description = str(self.feature) + ' ' + str(self.op) + ' ' + str(runtime.round(self.value))
            else:
                description = str(runtime.round(self.value)) + ' ' + str(self.op) + ' ' + str(self.feature) + ' ' + str(self.op) + ' ' + str(runtime.round(self.valueex))
            return description

        def singlebound(self, op):
            if self.valueex == None:
                return
            else:
                assert(op == runtime.predicate.op.lesseq or op == runtime.predicate.op.largereq)
                if self.op in runtime.predicate.op.lesseq:
                    if op == runtime.predicate.op.lesseq:
                        self.value = self.valueex
                        self.valueex = None
                    else:
                        self.op = runtime.predicate.op.oppose[self.op]
                        self.valueex = None
                else:
                    if op == runtime.predicate.op.largereq:
                        self.value = self.valueex
                        self.valueex = None
                    else:
                        self.op = runtime.predicate.op.oppose[self.op]
                        self.valueex = None

        def isconform(self, featurevalue):
            # __init__ requires assert(self.value != metainfo.top.NONE_VALUE and self.valueex != metainfo.top.NONE_VALUE)
            if featurevalue == metainfo.top.NONE_VALUE:
                return False
            if self.valueex == None:
                if self.op == runtime.predicate.op.eq:
                    return featurevalue == self.value
                elif self.op == runtime.predicate.op.lesseq:
                    return featurevalue <= self.value
                elif self.op == runtime.predicate.op.largereq:
                    return featurevalue >= self.value
            else:
                if self.op == runtime.predicate.op.lesseq:
                    return self.value <= featurevalue and featurevalue <= self.valueex
                elif self.op == runtime.predicate.op.largereq:
                    return self.value >= featurevalue and featurevalue >= self.valueex

        @staticmethod
        def combine(predicates_list):
            refer_features = {}
            for each_predicates in predicates_list:
                feature = each_predicates.feature
                op = each_predicates.op
                value = each_predicates.value
                valueex = each_predicates.valueex
                if feature not in refer_features:
                    refer_features[feature] = [None, None]
                bound = refer_features[feature]
                if op == runtime.predicate.op.eq:
                    if bound[0] == None or bound[0] <= value:
                        bound[0] = value
                    if bound[1] == None or bound[1] >= value:
                        bound[1] = value
                else:
                    if op == runtime.predicate.op.largereq:
                        if valueex != None:
                            if bound[0] == None or bound[0] <= valueex:
                                bound[0] = valueex
                            if bound[1] == None or bound[1] >= value:
                                bound[1] = value
                        else:
                            if bound[0] == None or bound[0] <= value:
                                bound[0] = value
                    else:
                        if valueex != None:
                            if bound[0] == None or bound[0] <= value:
                                bound[0] = value
                            if bound[1] == None or bound[1] >= valueex:
                                bound[1] = valueex
                        else:
                            if bound[1] == None or bound[1] >= value:
                                bound[1] = value
            predicates_list = []
            for each_feature in refer_features:
                newpredicate = None
                bound = refer_features[each_feature]
                if bound[0] != None and bound[1] != None:
                    if bound[0] < bound[1]:
                        if op == runtime.predicate.op.lesseq:
                            newpredicate = runtime.predicate(each_feature, runtime.predicate.op.lesseq, bound[0], bound[1])
                        else:
                            newpredicate = runtime.predicate(each_feature, runtime.predicate.op.largereq, bound[1], bound[0])
                        newpredicate.singlebound(op)
                    elif bound[0] == bound[1]:
                        newpredicate = runtime.predicate(each_feature, runtime.predicate.op.eq, bound[0], None)
                    else:
                        return None
                elif bound[0] != None and bound[1] == None:
                    if op == runtime.predicate.op.largereq:
                        newpredicate = runtime.predicate(each_feature, runtime.predicate.op.largereq, bound[0], None)
                        newpredicate.singlebound(runtime.predicate.op.largereq)
                elif bound[0] == None and bound[1] != None:
                    if op == runtime.predicate.op.lesseq:
                        newpredicate = runtime.predicate(each_feature, runtime.predicate.op.lesseq, bound[1], None)
                        newpredicate.singlebound(runtime.predicate.op.lesseq)
                if newpredicate == None:
                    predicates_list = None
                    break
                predicates_list.append(newpredicate)
            return predicates_list

        @staticmethod
        def issmallerthan(predicates_1, predicates_2):
            features_1 = set([eachpredicate.feature for eachpredicate in predicates_1])
            features_2 = set([eachpredicate.feature for eachpredicate in predicates_2])
            if len(features_1.intersection(features_2)) == len(features_2):
                predicates_12 = list(predicates_1) + list(predicates_2)
                predicates = runtime.predicate.combine(predicates_12)
                if predicates != None and set(predicates) == set(predicates_1):
                    return True
                else:
                    return False
            else:
                return False

        def mutationrange(self, mutation):
            thebounds = None
            if self.valueex != None:
                thebounds = [min(self.value, self.valueex), max(self.value, self.valueex)]
                if mutation != None:
                    thebounds = [thebounds[0] - mutation[0], thebounds[1] + mutation[1]]
            elif mutation == None:
                thebounds = [self.value, self.value]
            else:
                thebounds = [self.value - mutation[0], self.value + mutation[1]]
            return thebounds


        def __eq__(self, another):
            if type(self) == type(another) and self.feature == another.feature and self.op == another.op and self.value == another.value and self.valueex == another.valueex:
                return True
            else:
                return False

        def __hash__(self):
            return hash(self.print())

    class skyline:

        @staticmethod
        def nonevalueprocess(local_conform_map, diff_metafeature_indexes, polar, monotone_nonevalue_transformer):
            needprocess_notallsame = []
            for eachfeatureindex in range(0, local_conform_map.shape[1]):
                itemset = list(set(local_conform_map[:, eachfeatureindex]))
                if len(itemset) == 1 and (runtime.isnan(itemset[0])):
                    needprocess_notallsame.append(eachfeatureindex)
            # SIM: NONE_VALUE = Not relevant is safety.
            # DIFF for polar 0: != NONE_VALUE safety, for polar 1: != NONE_VALUE is skyline to verify to polar 0 rule, i.e. = NONE_VALUE is safety.
            # WaiCmp uses all unknown values in pair as NONE_VALUE, therefore can be trusted as safety.
            for eachindex in range(0, local_conform_map.shape[1]):
                npwhere_NONE_VALUE = np.where(local_conform_map[:, eachindex] == metainfo.top.NONE_VALUE)[0].tolist()
                npwhere_NONE_VALUE = [npwhere_NONE_VALUE, eachindex]
                if polar == 0:
                    if eachindex not in diff_metafeature_indexes:
                        local_conform_map[tuple(npwhere_NONE_VALUE)] = monotone_nonevalue_transformer[eachindex].min  # no sim: safety
                    else:
                        local_conform_map[tuple(npwhere_NONE_VALUE)] = monotone_nonevalue_transformer[eachindex].min  # no diff: skyline or always safety.
                else:
                    if eachindex not in diff_metafeature_indexes:
                        local_conform_map[tuple(npwhere_NONE_VALUE)] = monotone_nonevalue_transformer[eachindex].max  # no sim: safety
                    else:
                        local_conform_map[tuple(npwhere_NONE_VALUE)] = monotone_nonevalue_transformer[eachindex].max  # no diff: safety

        @staticmethod
        def isinside(base, aim, polar):
            if polar == 0:
                return np.all(aim <= base)
            else:
                return np.all(aim >= base)

        @staticmethod
        def distance(q, skylines, smallerprefer, w):
            def cost(q, p, w):
                return np.dot(w, np.fabs(q - p))
            def SP(q, SKY):
                dim = SKY.shape[1]
                m = SKY.shape[0]
                P = set()
                if dim == 1:
                    for i in range(1, m + 1):
                        P.add(runtime.hashablelist(SKY[i - 1, :]))
                elif dim == 2:
                    # sort points in SKY in the ascending order on dimension D1;
                    sortindexes = np.argsort(SKY[:, 0])
                    s = [None] * (m + 1)
                    for i in range(1, m + 1):
                        s[i] = SKY[sortindexes[i - 1], :]
                    for i in range(1, m + 2):
                        if i == 1:
                            P.add(runtime.hashablelist([q[0], s[1][1]]))
                        elif i == m + 1:
                            P.add(runtime.hashablelist([s[m][0], q[1]]))
                        else:
                            P.add(runtime.hashablelist([s[i - 1][0], s[i][1]]))
                else:
                    diversityvalues = [None] * dim
                    diversityvalues_len = [None] * dim
                    for eachdim in range(0, dim):
                        diversityvalues[eachdim] = list(set(SKY[:, eachdim]))
                        diversityvalues_len[eachdim] = len(diversityvalues[eachdim])
                    k = np.argmin(diversityvalues_len)
                    diversityvalues[k].sort(reverse=True)
                    l = diversityvalues_len[k]
                    S = [None] * (l + 2)
                    Sk_value = [None] * (l + 2)
                    SKY_k = SKY[:, k]
                    for i in range(1, l + 1):
                        Sk_value[i] = diversityvalues[k][i - 1]
                        _Si = SKY[np.where(SKY_k == Sk_value[i])]
                        S[i] = set()
                        _Si.tolist()
                        for each in _Si:
                            S[i].add(runtime.hashablelist(each))
                    S[l + 1] = set()
                    S_lp1_specialcase = runtime.hashablelist(q)
                    S[l + 1].add(S_lp1_specialcase)
                    Sk_value[l + 1] = S_lp1_specialcase[k]
                    Pi = [None] * (l + 2)
                    projectindexes = list(range(0, dim))
                    projectindexes.remove(k)
                    p = q[projectindexes]
                    Pi[1] = set()
                    Pi[1].add(runtime.hashablelist(p))
                    P = set()
                    P_adding = []
                    for eachpp in Pi[1]:
                        P_adding.append(eachpp)
                    recover_k = [Sk_value[1]] * len(P_adding)
                    P_adding = np.insert(P_adding, k, recover_k, axis=1)
                    for eachadding in P_adding:
                        P.add(runtime.hashablelist(eachadding))
                    proj_q = q[projectindexes]
                    SS = set()
                    for i in range(2, l + 2):
                        SS.update(S[i - 1])
                        _SS = np.array(list(SS), ndmin=2)
                        df = pd.DataFrame(_SS, columns=list(range(dim)))
                        mask = paretoset(df, sense=['max'] * dim, distinct=True, use_numba=True)
                        SS = set()
                        for eachmaskindex in range(0, len(mask)):
                            if mask[eachmaskindex] == 1:
                                SS.add(runtime.hashablelist(_SS[eachmaskindex, :]))
                        proj_SS= np.array(_SS)[:, projectindexes]
                        Pi[i] = SP(proj_q, proj_SS)
                        P_adding = []
                        for eachpp in Pi[i - 1]:
                            # larger i smaller Si[k], distance cost prefers smaller and local optimal points are not dominated as Margins.
                            if eachpp not in Pi[i]:
                                P_adding.append(eachpp)
                        # Not in condition, not always not empty.
                        if len(P_adding) > 0:
                            recover_k = [Sk_value[i]] * len(P_adding)
                            P_adding = np.insert(P_adding, k, recover_k, axis=1)
                            for each in P_adding:
                                P.add(runtime.hashablelist(each))
                return P
            if skylines.shape[0] == 0:
                return 0
            else:
                skylines = np.unique(skylines, axis=0)
                q = np.array(q)
                if smallerprefer == True:
                    q = q * (-1)
                    skylines = skylines * (-1)
                qp = SP(q=q, SKY=skylines)
                mincost = math.inf
                for eachqp in qp:
                    eachqp = np.array(eachqp)
                    thiscost = cost(q, eachqp, w)
                    if thiscost < mincost:
                        mincost = thiscost
                return mincost

    class indexes_transformer:

        def __init__(self, sub_indexes):
            self.subindexes = []
            for each_sub_index in sub_indexes:
                self.subindexes.append(np.array(each_sub_index))

        def fullindexes(self, subindexes):
            fullindexes = []
            for eachsubindex_index in range(0, len(subindexes)):
                fullindexes.append(self.subindexes[eachsubindex_index][subindexes[eachsubindex_index]])
            return fullindexes

        @staticmethod
        def combine(indexes_lists):
            thecombine = []
            for each_axis_index in range(0, len(indexes_lists[0])):
                thecombine.append([])
            for each_index_list in indexes_lists:
                for each_axis_index in range(0, len(each_index_list)):
                    thecombine[each_axis_index] += list(each_index_list[each_axis_index])
            return thecombine

    op_index_colon = slice(None)
    type_None = type(None)

    @staticmethod
    def isNone(object):
        if object is None:
            return True
        else:
            return False

    @staticmethod
    def isArrayEmpty(array):
        return np.array(array).size == 0

    class forest:

        def __init__(self, balance, mainmap_knowledgeupdating, labelindex, splitters, polar, weight, probability, confidence_coefficient, premilinary_condition_predicates, nondirectional_map = None, conform_map = None, roottrees = None):
            # map columns: [ UNITFEATURE TABLES, GML LABEL, PROBE LABEL ] ~ [ features (specified splitters range), labels (specified map index, to indicate both labels and Unlabeled) ]
            # main map for knowledge updating
            self.balance = balance
            self.map = np.array(mainmap_knowledgeupdating)
            self.labelindex = labelindex
            self.splitters = splitters
            self.splitter_count = len(self.splitters)
            self.polar = polar
            self.raw_rule_approveprobability = None
            if type(metainfo.paras.raw_rule_approveprobability) == list:
                self.raw_rule_approveprobability = metainfo.paras.raw_rule_approveprobability[self.polar]
            else:
                self.raw_rule_approveprobability = metainfo.paras.raw_rule_approveprobability
            self.weight = weight
            self.probability = probability
            self.confidence_coefficient = confidence_coefficient
            self.premilinary_condition_predicates = premilinary_condition_predicates
            self.nondirectional_map = nondirectional_map
            self.conform_map = conform_map
            self.trees = [None] * self.splitter_count
            self.rules = [None] * self.splitter_count
            self.roottrees = roottrees
            if self.roottrees == None:
                self.roottrees = range(0, self.splitter_count)
            self.generatetrees()

        def generatetrees(self):
            for each_splitter_index in self.roottrees:
                self.trees[each_splitter_index] = runtime.forest.tree(self, each_splitter_index)
                self.rules[each_splitter_index] = self.trees[each_splitter_index].rules
                self.rules[each_splitter_index].sort(key=lambda x:x.criterion, reverse=True)

        class table:

            def __init__(self, fullmap, submap_indtrans, splitterindex, labelindex):
                self.splitter_map = np.array(fullmap[submap_indtrans.subindexes[0], splitterindex]).reshape(-1, 1)
                self.label_map = np.array(fullmap[submap_indtrans.subindexes[0], labelindex]).reshape(-1, 1)
                self.labeled_indtrans = np.where(np.logical_and(np.logical_or(self.label_map == 0, self.label_map == 1), self.splitter_map != metainfo.top.NONE_VALUE))
                self.unlabeled_indtrans = np.where(np.logical_and(self.label_map == metainfo.top.INDETERMINATE, self.splitter_map != metainfo.top.NONE_VALUE))
                self.labeled_indtrans = runtime.indexes_transformer(submap_indtrans.fullindexes(self.labeled_indtrans))
                self.unlabeled_indtrans = runtime.indexes_transformer(submap_indtrans.fullindexes(self.unlabeled_indtrans))
                self.splitter_map_labeled = np.array(fullmap[self.labeled_indtrans.subindexes[0], splitterindex]).reshape(-1, 1)
                self.splitter_map_unlabeled = np.array(fullmap[self.unlabeled_indtrans.subindexes[0], splitterindex]).reshape(-1, 1)
                self.label_map_labeled = np.array(fullmap[self.labeled_indtrans.subindexes[0], labelindex]).reshape(-1, 1)
                self.label_map_unlabeled = np.array(fullmap[self.unlabeled_indtrans.subindexes[0], labelindex]).reshape(-1, 1)

        class tree:

            def __init__(self, forest, root_splitter_index):
                self.forest = forest
                self.root_splitter_index = root_splitter_index
                self.map = self.forest.map
                self.nondirectional_map = self.forest.nondirectional_map
                self.conform_map = self.forest.conform_map
                self.labelindex = self.forest.labelindex
                self.polar = self.forest.polar
                self.rules = []
                self.split(depth_splitters=[], splitter_index=self.root_splitter_index, pre_predicates=[], pre_mainmap_indtrans=runtime.op_index_colon, pre_nondirectional_indtrans=runtime.op_index_colon, pre_conform_indtrans=runtime.op_index_colon)

            class rule:
                def __init__(self, predicates, premilinary_condition_predicates, polareffective, criterion, weight, stat_probability, confidence_coefficient, labeled_indexes, unlabeled_indexes, map_indtrans, nondirectional_labeled_indexes, nondirectional_indtrans, conform_labeled_indexes, conform_unlabeled_indexes, conform_indtrans):
                    self.predicates = None
                    if premilinary_condition_predicates != None:
                        self.predicates = runtime.predicate.combine(premilinary_condition_predicates + predicates)
                    else:
                        self.predicates = predicates
                    self.predicates = runtime.hashableset(self.predicates)
                    self.predicatedisplays = ''
                    self.gml = None
                    self.criterion = criterion
                    self.polareffective = polareffective
                    if self.polareffective == True:
                        assert(self.criterion >= 0)
                        self.weight = weight
                        self.stat_probability = stat_probability
                        self.confidence_coefficient = confidence_coefficient
                        self.polar = runtime.probabilitypolar(self.stat_probability)
                    self.labeled_indexes = labeled_indexes
                    self.unlabeled_indexes = unlabeled_indexes
                    self.map_indtrans = map_indtrans
                    self.nondirectional_labeled_indexes = nondirectional_labeled_indexes
                    self.nondirectional_indtrans = nondirectional_indtrans
                    self.conform_labeled_indexes = conform_labeled_indexes
                    self.conform_unlabeled_indexes = conform_unlabeled_indexes
                    self.conform_indtrans = conform_indtrans
                    self.formfeature = runtime.hashableset()
                    for each_predicate in self.predicates:
                        self.formfeature.add(each_predicate.feature)
                    self.resolution = len(self.labeled_indexes) + len(self.unlabeled_indexes)
                    if self.nondirectional_labeled_indexes != None:
                        self.resolution += len(self.nondirectional_labeled_indexes)
                    if self.conform_labeled_indexes != None and self.conform_unlabeled_indexes != None:
                        self.resolution += (len(self.conform_labeled_indexes) + len(self.conform_unlabeled_indexes))

            def split(self, depth_splitters, splitter_index, pre_predicates, pre_mainmap_indtrans, pre_nondirectional_indtrans, pre_conform_indtrans):
                depth_splitters.append(splitter_index)
                if pre_mainmap_indtrans == runtime.op_index_colon:
                    pre_mainmap_indtrans = runtime.indexes_transformer(tuple([list(range(0, self.map.shape[0])), [0] * self.map.shape[0]]))
                mainmap_table = runtime.forest.table(fullmap=self.map, submap_indtrans=pre_mainmap_indtrans, splitterindex=splitter_index, labelindex=self.labelindex)
                pre_labeled_splittermap = mainmap_table.splitter_map_labeled
                pre_unlabeled_splittermap = mainmap_table.splitter_map_unlabeled
                pre_labeled_labelmap = mainmap_table.label_map_labeled
                pre_unlabeled_labelmap = mainmap_table.label_map_unlabeled
                pre_labeled_indtrans = mainmap_table.labeled_indtrans
                pre_unlabeled_indtrans = mainmap_table.unlabeled_indtrans
                nondirectional_pre_labeled_splittermap = None
                nondirectional_pre_labeled_labelmap = None
                nondirectional_pre_labeled_indtrans = None
                if runtime.isNone(self.nondirectional_map) == False and pre_nondirectional_indtrans != None:
                    if pre_nondirectional_indtrans == runtime.op_index_colon:
                        pre_nondirectional_indtrans = runtime.indexes_transformer(tuple([list(range(0, self.nondirectional_map.shape[0])), [0] * self.nondirectional_map.shape[0]]))
                    if runtime.isArrayEmpty(pre_nondirectional_indtrans.subindexes) == False:
                        nondirectionalmap_table = runtime.forest.table(fullmap=self.nondirectional_map, submap_indtrans=pre_nondirectional_indtrans, splitterindex=splitter_index, labelindex=self.labelindex)
                        nondirectional_pre_labeled_splittermap = nondirectionalmap_table.splitter_map_labeled
                        nondirectional_pre_labeled_labelmap = nondirectionalmap_table.label_map_labeled
                        nondirectional_pre_labeled_indtrans = nondirectionalmap_table.labeled_indtrans
                conform_pre_labeled_splittermap = None
                conform_pre_unlabeled_splittermap = None
                conform_pre_labeled_labelmap = None
                conform_pre_unlabeled_labelmap = None
                conform_pre_labeled_indtrans = None
                conform_pre_unlabeled_indtrans = None
                if runtime.isNone(self.conform_map) == False and pre_conform_indtrans != None:
                    if pre_conform_indtrans == runtime.op_index_colon:
                        pre_conform_indtrans = runtime.indexes_transformer(tuple([list(range(0, self.conform_map.shape[0])), [0] * self.conform_map.shape[0]]))
                    if runtime.isArrayEmpty(pre_conform_indtrans.subindexes) == False:
                        conformmap_table = runtime.forest.table(fullmap=self.conform_map, submap_indtrans=pre_conform_indtrans, splitterindex=splitter_index, labelindex=self.labelindex)
                        conform_pre_labeled_splittermap = conformmap_table.splitter_map_labeled
                        conform_pre_unlabeled_splittermap = conformmap_table.splitter_map_unlabeled
                        conform_pre_labeled_labelmap = conformmap_table.label_map_labeled
                        conform_pre_unlabeled_labelmap = conformmap_table.label_map_unlabeled
                        conform_pre_labeled_indtrans = conformmap_table.labeled_indtrans
                        conform_pre_unlabeled_indtrans = conformmap_table.unlabeled_indtrans
                splitvalues = list(set(pre_labeled_splittermap.ravel()))
                splitvalues.sort(reverse=False)
                previoussplitvalue = (-1) * math.inf
                for eachsplitvalue in list(splitvalues):
                    if eachsplitvalue - previoussplitvalue < metainfo.paras.tree_split_significantdelta:
                        splitvalues.remove(eachsplitvalue)
                    else:
                        previoussplitvalue = eachsplitvalue
                assert(metainfo.top.NONE_VALUE not in splitvalues)
                # Avoid complete or sub-complete redundant selection predicates.
                splitvalues = runtime.completeset_redundant(splitvalues, self.polar)
                optimal_criterion = (-1) * math.inf
                optimal_rule_split = None
                if self.polar == 1:
                    splitvalues.sort(reverse=True)
                for eachsplitvalue in splitvalues:
                    currentrule = self.criterion(splitter_index, eachsplitvalue, pre_labeled_splittermap, pre_unlabeled_splittermap, pre_labeled_labelmap, pre_unlabeled_labelmap, pre_labeled_indtrans, pre_unlabeled_indtrans, pre_predicates, nondirectional_pre_labeled_splittermap, nondirectional_pre_labeled_labelmap, nondirectional_pre_labeled_indtrans, conform_pre_labeled_splittermap, conform_pre_unlabeled_splittermap, conform_pre_labeled_labelmap, conform_pre_unlabeled_labelmap, conform_pre_labeled_indtrans, conform_pre_unlabeled_indtrans)
                    if currentrule.criterion >= optimal_criterion:
                        optimal_rule_split = currentrule
                    if currentrule.polareffective == False:
                        break
                if optimal_rule_split != None:
                    if optimal_rule_split.polareffective == True:
                        self.rules.append(optimal_rule_split)
                    if (metainfo.paras.tree_tridepth_exhausted == True or optimal_rule_split.polareffective == False) and len(depth_splitters) < metainfo.paras.tree_maxdepth:
                        for each_next_splitter in range(0, self.forest.splitter_count):
                            if each_next_splitter not in depth_splitters:
                                self.split(depth_splitters=list(depth_splitters), splitter_index=each_next_splitter, pre_predicates=optimal_rule_split.predicates, pre_mainmap_indtrans=optimal_rule_split.map_indtrans, pre_nondirectional_indtrans=optimal_rule_split.nondirectional_indtrans, pre_conform_indtrans=optimal_rule_split.conform_indtrans)

            def criterion(self, splitter_index, splitvalue, pre_labeled_splittermap, pre_unlabeled_splittermap, pre_labeled_labelmap, pre_unlabeled_labelmap, pre_labeled_indtrans, pre_unlabeled_indtrans, pre_predicates, nondirectional_pre_labeled_splittermap, nondirectional_pre_labeled_labelmap, nondirectional_pre_labeled_indtrans, conform_pre_labeled_splittermap, conform_pre_unlabeled_splittermap, conform_pre_labeled_labelmap, conform_pre_unlabeled_labelmap, conform_pre_labeled_indtrans, conform_pre_unlabeled_indtrans):
                from source.rule import confidence
                labeled_indexes = None
                unlabeled_indexes = None
                nondirectional_labeled_indexes = None
                conform_labeled_indexes = None
                conform_unlabeled_indexes = None
                if self.polar == 0:
                    labeled_indexes = np.where(pre_labeled_splittermap <= splitvalue)
                    unlabeled_indexes = np.where(pre_unlabeled_splittermap <= splitvalue)
                    if runtime.isNone(nondirectional_pre_labeled_splittermap) == False:
                        nondirectional_labeled_indexes = np.where(nondirectional_pre_labeled_splittermap <= splitvalue)
                    if runtime.isNone(conform_pre_labeled_splittermap) == False:
                        conform_labeled_indexes = np.where(conform_pre_labeled_splittermap <= splitvalue)
                        conform_unlabeled_indexes = np.where(conform_pre_unlabeled_splittermap <= splitvalue)
                else:
                    labeled_indexes = np.where(pre_labeled_splittermap >= splitvalue)
                    unlabeled_indexes = np.where(pre_unlabeled_splittermap >= splitvalue)
                    if runtime.isNone(nondirectional_pre_labeled_splittermap) == False:
                        nondirectional_labeled_indexes = np.where(nondirectional_pre_labeled_splittermap >= splitvalue)
                    if runtime.isNone(conform_pre_labeled_splittermap) == False:
                        conform_labeled_indexes = np.where(conform_pre_labeled_splittermap >= splitvalue)
                        conform_unlabeled_indexes = np.where(conform_pre_unlabeled_splittermap >= splitvalue)
                labeled_labels = pre_labeled_labelmap[labeled_indexes]
                unlabeled_labels = pre_unlabeled_labelmap[unlabeled_indexes]
                nondirectional_labeled_labels = []
                if runtime.isNone(nondirectional_pre_labeled_splittermap) == False:
                    nondirectional_labeled_labels = nondirectional_pre_labeled_labelmap[nondirectional_labeled_indexes]
                labeled_indexes = pre_labeled_indtrans.fullindexes(labeled_indexes)
                unlabeled_indexes = pre_unlabeled_indtrans.fullindexes(unlabeled_indexes)
                map_indexes = runtime.indexes_transformer.combine([labeled_indexes, unlabeled_indexes])
                assert(set(map_indexes[1]) == set([0]) or set(map_indexes[1]) == set())
                map_indexes[0].sort(reverse=False)
                map_indtrans = runtime.indexes_transformer(map_indexes)
                nondirectional_indtrans = None
                if runtime.isNone(nondirectional_pre_labeled_splittermap) == False:
                    nondirectional_labeled_indexes = nondirectional_pre_labeled_indtrans.fullindexes(nondirectional_labeled_indexes)
                    nondirectional_indexes = nondirectional_labeled_indexes
                    nondirectional_indtrans = runtime.indexes_transformer(nondirectional_indexes)
                conform_indtrans = None
                if runtime.isNone(conform_pre_labeled_splittermap) == False:
                    conform_labeled_indexes = conform_pre_labeled_indtrans.fullindexes(conform_labeled_indexes)
                    conform_unlabeled_indexes = conform_pre_unlabeled_indtrans.fullindexes(conform_unlabeled_indexes)
                    conform_indexes = runtime.indexes_transformer.combine([conform_labeled_indexes, conform_unlabeled_indexes])
                    assert(set(conform_indexes[1]) == set([0]) or set(conform_indexes[1]) == set())
                    conform_indexes[0].sort(reverse=False)
                    conform_indtrans = runtime.indexes_transformer(conform_indexes)
                label1count = sum(labeled_labels) + sum(nondirectional_labeled_labels)
                label0count = len(labeled_labels) + len(nondirectional_labeled_labels) - label1count
                probability = self.forest.probability(balance=self.forest.balance, label1count=label1count, label0count=label0count)
                polareffective_skylinecoverage = None
                if self.polar == 0:
                    polareffective_skylinecoverage = len(labeled_labels) - sum(labeled_labels)
                else:
                    polareffective_skylinecoverage = sum(labeled_labels)
                rulearea = confidence.subarea(polar=self.polar, func_probability=self.forest.probability, subarea_pairs=None, label_pairs=None)
                rulearea.labeledcount = label0count + label1count
                rulearea.totalcount = label0count + label1count + len(unlabeled_indexes[0])
                confidence_coefficient = self.forest.confidence_coefficient(func_probability=self.forest.probability, subareas=rulearea)
                polareffective_weight = self.forest.weight(confidence_coefficient=1, balanceprobability=probability, polar=self.polar)
                polareffective = runtime.probabilitypolar(probability) == self.polar and confidence.effective(polareffective_weight, effectiveprobability=self.forest.raw_rule_approveprobability, effectiveweight=None)
                #weight = self.forest.weight(confidence_coefficient=confidence_coefficient, balanceprobability=probability, polar=self.polar)
                weight = polareffective_weight
                criterion = (-1) * math.inf
                if weight != None:
                    criterion = math.fabs(weight)
                op = None
                if self.polar == 0:
                    op = runtime.predicate.op.lesseq
                else:
                    op = runtime.predicate.op.largereq
                predicates = list(pre_predicates)
                split_predicate = runtime.predicate(feature=self.forest.splitters[splitter_index], op=op, value=splitvalue)
                predicates.append(split_predicate)
                predicates = runtime.predicate.combine(predicates)
                therule = runtime.forest.tree.rule(predicates=predicates, premilinary_condition_predicates=self.forest.premilinary_condition_predicates, polareffective=polareffective, criterion=criterion, weight=weight, stat_probability=probability, confidence_coefficient=confidence_coefficient, labeled_indexes=labeled_indexes, unlabeled_indexes=unlabeled_indexes, map_indtrans=map_indtrans, nondirectional_labeled_indexes=nondirectional_labeled_indexes, nondirectional_indtrans=nondirectional_indtrans, conform_labeled_indexes=conform_labeled_indexes, conform_unlabeled_indexes=conform_unlabeled_indexes, conform_indtrans=conform_indtrans)
                return therule

    @staticmethod
    def asyncraise(tid, exctype = SystemExit):
        try:
            """raises the exception, performs cleanup if needed"""
            tid = tid.ident
            tid = ctypes.c_long(tid)
            if not inspect.isclass(exctype):
                exctype = type(exctype)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
            if res == 0:
                raise ValueError("invalid thread id")
            elif res != 1:
                # """if it returns a number greater than one, you're in trouble,
                # and you should call it again with exc=NULL to revert the effect"""
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                raise SystemError("PyThreadState_SetAsyncExc failed")
        except Exception as e:
            runtime.console('Error > asyncraise tid: ' + str(tid), repr(e), runtime.console.styles.EXCEPTION)
            traceback.print_exc()

    class countdown:

        def __init__(self, t, info, prompt = None, period = 1):
            self.t = max(0, t)
            self.info = str(info)
            self.thread = None
            self.prompt = prompt
            self.period = period

        def perform(self):
            while self.t >= 0:
                mins, secs = divmod(self.t, 60)
                timer = runtime.console.color.BACKGROUND + '\r' + self.info + ' in {:02d}:{:02d}'.format(mins, secs) + runtime.console.color.END
                if self.prompt != None:
                    timer += (' >> ' + str(self.prompt) + ' :> ')
                sys.stdout.write(timer)
                sys.stdout.flush()
                time.sleep(self.period)
                self.t -= self.period

        def async_perform(self):
            self.thread = Thread(target=self.perform, args=())
            self.thread.start()

        def exit(self):
            self.t = (-1) * math.inf
            runtime.asyncraise(self.thread)


    class Thread_ReturnValue():

        def __init__(self):
            self.result = []

    @staticmethod
    def awaitinput(default, t, info, prompt = None, period = 1):
        def input_ReturnValue(Thread_ReturnValue):
            Thread_ReturnValue.result.append(input())
        def skipinput(inputthread,Thread_ReturnValue,default):
            runtime.asyncraise(inputthread)
            Thread_ReturnValue.result.append(default)
            Thread_ReturnValue.result.append(default)
        theinput = None
        Thread_ReturnValue = runtime.Thread_ReturnValue()
        inputthread = Thread(target=input_ReturnValue, args=(Thread_ReturnValue,))
        timer = Timer(t, skipinput, args=(inputthread,Thread_ReturnValue,default))
        timer.start()
        cd = runtime.countdown(t, info, prompt, period)
        cd.async_perform()
        inputthread.start()
        inputthread.join(t + 1)
        timer.cancel()
        cd.exit()
        if len(Thread_ReturnValue.result) == 1:
            if len(Thread_ReturnValue.result[0]) > 0:
                print('awaitinput =?', Thread_ReturnValue.result[0])
                cert = input('awaitinput :> ')
                if len(cert) != 0:
                    Thread_ReturnValue.result[0] = cert
            else:
                Thread_ReturnValue.result[0] = default
        else:
            print()
        theinput = Thread_ReturnValue.result[0]
        print('awaitinput :=', theinput)
        return theinput

    @staticmethod
    def digit(number):
        thedigit = 0
        while number != 0:
            number = int(number/10)
            thedigit += 1
        return thedigit

    class nlp:
        # Auti Token Sparse problem fundamentally, instead of using Training-Test's Seeming Trick.

        def __init__(self, regulartokens, attributed_sentences):
            iter = 10
            workercount = cpu_count() / 2
            if workercount % 2 == 1:
                workercount *= 2
            workercount -= 2
            if workercount > 12:
                workercount = 12
            elif workercount < 2:
                workercount = 2
            workercount = int(workercount)
            self.wordlist = list(regulartokens)
            runtime.console('SGML > w2groups anti-sparse', str(len(self.wordlist)) + ' sentences processing, # multi-threads = ' + str(workercount) + ' / ' + str(cpu_count()) + ' ...', runtime.console.styles.INFO)
            self.wordvectors = []
            if attributed_sentences == None:
                self.model = None
            else:
                self.model = fasttext.FastText(attributed_sentences, min_count=1, workers=workercount, iter=iter)
                for each_regulartoken in self.wordlist:
                    self.wordvectors.append(self.model.wv[each_regulartoken])
            conv_test_exp_multiplier = runtime.digit(len(self.wordvectors)) - 1
            conv_test = math.pow(10, conv_test_exp_multiplier) * 1e-9
            conv_test_print = -9 + conv_test_exp_multiplier
            runtime.console('SGML > w2groups anti-sparse',  'nltk :: KMeansClusterer clustering conv_threshold = 1e' + str(conv_test_print) + ' ... take times ...', runtime.console.styles.INFO)
            self.wordcluster = KMeansClusterer(metainfo.paras.nlpw2vgroups, distance=nltk.cluster.util.cosine_distance, repeats=iter, conv_test=conv_test, avoid_empty_clusters=True)
            self.assigned_clusters = self.wordcluster.cluster(self.wordvectors, assign_clusters=True)
            self.w2groups = {}
            for eachindex in range(0, len(self.wordlist)):
                eachword = self.wordlist[eachindex]
                self.w2groups[eachword] = self.assigned_clusters[eachindex]

    @staticmethod
    def confidentialsample(priorp, error, n_, N, P=None):
        if P == None:
            if n_ == N:
                return 1
            else:
                n = None
                if N != None:
                    n = int(float(N - 1)/(N - n_) * n_)
                else:
                    n = n_
                t = math.sqrt((float(n) * math.pow(error, 2))/(priorp * (1 - priorp)))
                P = stats.norm.cdf(t)
                return 1 - (1 - P) * 2
        else:
            t = stats.norm.ppf(1 - (1 - P)/2)
            n = int(math.pow(t,2) * priorp * (1 - priorp)/math.pow(error, 2))
            n_ = math.ceil(float(n)/(1 + float(1 + n)/N))
            return n_

    @staticmethod
    def issorted(iter, reverse):
        if reverse == False:
            return all([iter[i] <= iter[i + 1] for i in range(len(iter) - 1)])
        else:
            return all([iter[i] >= iter[i + 1] for i in range(len(iter) - 1)])

    @staticmethod
    @jit
    def recombination_jit(iter1, iter2, lenlimit, len1, len2):
        therecombination_jit = []
        for eachlenlimit1 in len1:
            for eachlenlimit2 in len2:
                combination1 = list()
                combination2 = list()
                for eachcombination1 in itertools.combinations(iter1, eachlenlimit1):
                    combination1.append(set(eachcombination1))
                for eachcombination2 in itertools.combinations(iter2, eachlenlimit2):
                    combination2.append(set(eachcombination2))
                for eachproductset in itertools.product(combination1, combination2):
                    thisrecombination = set(eachproductset[0])
                    thisrecombination.update(eachproductset[1])
                    if len(thisrecombination) <= lenlimit and thisrecombination.issubset(iter1) == False and thisrecombination.issubset(iter2) == False:
                        therecombination_jit.append(thisrecombination)
        return therecombination_jit

    @staticmethod
    def recombination(iter1, iter2, lenlimit, semisub, processed_cache):
        if (iter1, iter2) in processed_cache or (iter2, iter1) in processed_cache:
            return None
        len1 = None
        len2 = None
        if semisub == True:
            len1 = [max(int(len(iter1) / 2), 1)]
            len2 = [max(int(len(iter2) / 2), 1)]
        else:
            len1 = range(1, len(iter1) + 1)
            len2 = range(1, len(iter2) + 1)
        numba_coded = (range(0, len(iter1)), range(len(iter1), len(iter1) + len(iter2)))
        therecombination_jit = None
        if numba_coded in processed_cache:
            therecombination_jit = processed_cache[numba_coded]
        else:
            therecombination_jit = runtime.recombination_jit(range(0, len(iter1)), range(len(iter1), len(iter1) + len(iter2)), lenlimit, len1, len2)
            processed_cache[numba_coded] = therecombination_jit
        processed_cache[(iter1, iter2)] = metainfo.top.AUTO
        iter = list(iter1)
        iter += list(iter2)
        therecombination = set()
        for eachcombination_jit in therecombination_jit:
            currentcombination = []
            for eachindex in eachcombination_jit:
                currentcombination.append(iter[eachindex])
            currentcombination = runtime.hashableset(runtime.predicate.combine(currentcombination))
            if runtime.predicate.issmallerthan(currentcombination, iter1) == False and runtime.predicate.issmallerthan(currentcombination, iter2) == False:
                therecombination.add(currentcombination)
        return therecombination

    @staticmethod
    def display(header, space, dictcontent=None):
        if dictcontent == None:
            displaystr = ''
            for eachvindex in range(0, len(header)):
                currentv = header[eachvindex]
                currentv = str(currentv)
                displaystr += currentv
                displaystr += ' ' * (space[eachvindex] - len(currentv))
            return displaystr
        else:
            if type(dictcontent) != list:
                dictcontent = [dictcontent] * len(header)
            displaydict = {}
            for eachvindex in range(0, len(header)):
                currentv = header[eachvindex]
                currentv = str(currentv)
                displaydict[currentv] = str(dictcontent[eachvindex])
            return displaydict

    @staticmethod
    def dellist(thelist, delindexes):
        return [thelist[i] for i in range(0, len(thelist)) if i not in delindexes]

    @staticmethod
    def fitprobability(labellist, polar, balance1_multiplier):
        labellist = np.array(labellist, dtype=np.int64)
        labellist = np.fabs(1 - polar - labellist)
        labellist = labellist.astype(np.int64)
        totalmass = None
        if balance1_multiplier != None:
            balancer = None
            if polar == 0:
                balancer = float(1) / balance1_multiplier
            else:
                balancer = balance1_multiplier
            bincount = np.bincount(labellist)
            labellist = labellist.astype(np.float64)
            labellist *= balancer
            totalmass = balancer * bincount[1] + bincount[0]
        else:
            totalmass = len(labellist)
        fitness = float(np.sum(labellist)) / totalmass
        return fitness

    @staticmethod
    def setcompare(set_1, set_2, method):
        if method == 'distinct':
            return len(set_1 - set_2)
        elif method == 'jaccard':
            intersection = set_1.intersection(set_2)
            union = set_1.union(set_2)
            return float(len(intersection)) / len(union)

    @staticmethod
    def sortbased(aim, score, reverse):
        if reverse == False:
            return np.array(aim)[np.argsort(score)], np.array(score)[np.argsort(score)]
        else:
            return np.array(aim)[np.argsort(np.array(score) * (-1))], np.array(score)[np.argsort(np.array(score) * (-1))]

    class minmax:
        def __init__(self, min, max):
            self.min = min
            self.max = max

    @staticmethod
    def cmp(x, y):
        if x < y:
            return -1
        elif x > y:
            return 1
        else:
            return 0

    @staticmethod
    def cmp_reverse(x, y):
        if x < y:
            return 1
        elif x > y:
            return -1
        else:
            return 0

    @staticmethod
    def sidesort(sortaim):
        i = np.array(range(0, len(sortaim)))
        mid = 0.5 * (i[0] + i[-1])
        x = - np.fabs(i - mid)
        xs = np.argsort(x)
        return xs, np.array(sortaim)[xs].tolist()