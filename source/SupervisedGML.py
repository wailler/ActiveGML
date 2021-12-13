import os
import warnings
import pandas as pd
import re
import sys
import datetime
import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from collections import Counter
import math
from scipy.sparse import *
from scipy.stats import t
import heapq
import csv
from copy import deepcopy, copy
import collections
import functools
import torch.nn as nn
import torch.optim as optim
from enum import Flag, auto

from source import fg
from source.rule import discretizefeature, genetic, rule, confidence
from source import metainfo
from source.runtime import runtime

class SGML:

    class probeGML:

        class probe:

            def __init__(self, pid, labeltype, label, probability, weight):
                self.pid = pid
                self.labeltype = labeltype
                self.label = label
                self.probability = probability
                self.weight = weight

        class rule:

            def __init__(self, therule):
                self.conform_pids = []
                for conform_pair in therule.conform_pairs:
                    self.conform_pids.append(conform_pair.pid)
                self.predicates = therule.predicates
                self.weight = therule.weight
                self.probability = therule.probability
                self.polar = therule.polar

        @staticmethod
        def foreround(proberule, gml):
            conform_pairs = []
            for conform_pid in proberule.conform_pids:
                conform_pairs.append(gml.pairs[conform_pid])
            inheritrule = genetic.inherit(gml=gml, predicates=proberule.predicates, polar=proberule.polar)
            isnew, thegenetic = genetic.find(gml, rule_or_inherit=inheritrule)
            rule.inherit(inherit_type=rule.inherit.types.FOREROUND, target_rule=inheritrule, flow1=None, flow2=None)
            currentcandidategenetic = genetic.candidategenetic(thegenetic, proberule.polar)
            therule = rule(candidategenetic=currentcandidategenetic, predicates=proberule.predicates, conform_pairs=conform_pairs, weight=proberule.weight, probability=proberule.probability, gml=gml, polar=proberule.polar)
            therule.toskyline()
            therule.toapprove()
            therule.toevolution()
            therule.toprocessed()
            therule.candidategenetic = None

        def __init__(self, gml):
            self.next = False
            self.active_round = None
            if metainfo.paras.supervised == True:
                self.next = gml.active_round < metainfo.paras.active_rounds
                self.active_round = gml.active_round
            self.probes = []
            for eachpair in gml.pairs:
                probability, weight = eachpair.withprobe_get_probability()
                assert(probability != None)
                self.probes.append(SGML.probeGML.probe(eachpair.pid, eachpair.labeltype, eachpair.label, probability, weight))
            self.approved_rules = []
            gml.approved_rules.sort(key=lambda x:x.truth_correcting[0], reverse=True)
            for eachapproved_rule in gml.approved_rules:
                self.approved_rules.append(SGML.probeGML.rule(eachapproved_rule))

    class cache:

        class globalcache:
            recombination = {}

        def __init__(self):
            self.skylinecache = {}

    def init_pairid(self, rowdata, data1, data2):
        id1, id2 = rowdata.split(',')
        id1 = id1.replace(id1, data1.split('.')[0] + '_'+ id1)
        id = id1
        if data2 is not None:
            id2 = id2.replace(id2, data2.split('.')[0] + '_' + id2)
            id = id + ',' + id2
        return id

    def record(self):
        if runtime.pickledump(self, 'records', 'e') == False:
            noreferids = set()
            for pairindex in range(0, len(self.rawpairs)):
                elem = self.rawpairs.values[pairindex]
                idlist = elem[0].split(",", 1)
                e1id = idlist[0]
                e2id = idlist[1]
                self.referids.add(e1id)
                self.referids.add(e2id)
            alldatavalues = [self.data1.values]
            if type(self.data2) == pd.core.frame.DataFrame:
                alldatavalues.append(self.data2.values)
            elif self.data.name == 'songs':
                data1_ids = self.data1.values[:, 0]
                referids_list = list(self.referids)
                referids_index = np.where(np.in1d(data1_ids, referids_list) == True)
                alldatavalues = [self.data1.values[referids_index]]
            totaldatavalues = 0
            for eachdatavalue in alldatavalues:
                totaldatavalues += len(eachdatavalue)
            currentdatavalues = 0
            for eachdatavalue in alldatavalues:
                for elem in eachdatavalue:
                    currentdatavalues += 1
                    runtime.consoleprogress(currentdatavalues, totaldatavalues, 'init records')
                    if elem[0] not in self.referids:
                        noreferids.add(elem[0])
                    recordtext = [None] * (len(self.RecordAttributes) + 2)
                    recordtext[0] = ''
                    occurrange = None
                    if metainfo.paras.nlpw2vgroups is not None and 'attroccur' in metainfo.paras.nlpw2vgroups:
                        occurrange = []
                        for eachattrindex in range(1, len(self.RecordAttributes)):
                            if not (self.data.infer_keytoken.idfrange[eachattrindex] == 0 or type(self.data.infer_keytoken.idfrange[eachattrindex]) == list and 0 in self.data.infer_keytoken.idfrange[eachattrindex] and set(self.data.infer_keytoken.idfrange[eachattrindex]) == 0):
                                occurrange.append(eachattrindex)
                    else:
                        occurrange = range(1, len(self.RecordAttributes))
                    for eachtextindex in range(1, len(self.RecordAttributes)):
                        recordtext[eachtextindex] = runtime.process(elem[eachtextindex], tostring=True)
                        if eachtextindex in occurrange and recordtext[eachtextindex] != None:
                            if len(recordtext[0]) > 0:
                                recordtext[0] += ' '
                            recordtext[0] += recordtext[eachtextindex]
                    recordtext[-1] = set(recordtext[0].split())
                    recordtext[-2] = set(recordtext[-1])
                    self.origintokens.update(recordtext[-1])
                    self.records[elem[0]] = recordtext
            runtime.pickledump(self, 'records', 'w')
        else:
            runtime.pickledump(self, 'records', 'r')
            for eachrecordid in self.records:
                recordtext = self.records[eachrecordid]
                self.origintokens.update(recordtext[-1])
            for pairindex in range(0, len(self.rawpairs)):
                elem = self.rawpairs.values[pairindex]
                idlist = elem[0].split(",", 1)
                e1id = idlist[0]
                e2id = idlist[1]
                self.referids.add(e1id)
                self.referids.add(e2id)

    def init_token(self):
        # [records attributes, -2: Infer Key Tokens, -1: All Regular Tokens]
        def regulize_token(data):
            token = data[0]
            if metainfo.paras.nlpw2vgroups is not None and 'ex' in metainfo.paras.nlpw2vgroups:
                return re.sub(runtime.regularpattern.notunicodepattern_ex, '', token)
            else:
                return re.sub(runtime.regularpattern.notunicodepattern, '', token)
        if runtime.pickledump(self, 'regularingtokens', 'e') == False:
            tokens = self.origintokens
            token_list = list(tokens)
            token_df = pd.DataFrame(token_list)
            token_df.columns = ['id']
            token_df['regular'] = token_df.apply(func=regulize_token, axis=1)
            token_diff = token_df[token_df['id'] != token_df['regular']]
            merge = pd.merge(token_diff, token_diff, on='regular')
            regularied = merge[merge['id_x'] != merge['id_y']]
            regularied = regularied[['id_x', 'regular']]
            keys = regularied['id_x'].tolist()
            values = regularied['regular'].tolist()
            self.regularingtokens = dict(zip(keys, values))
            runtime.pickledump(self, 'regularingtokens', 'w')
            runtime.console('GML > Preprocess', 'regularingtokens pickles dumped', runtime.console.styles.REPORT)
        else:
            runtime.pickledump(self, 'regularingtokens', 'r')

        for eachtoken in self.origintokens:
            if eachtoken not in self.regularingtokens:
                self.regularingtokens[eachtoken] = eachtoken

        if self.data.infer_keytoken.patternrestrict is not None:
            for eachrecordid in self.records:
                recordtext = self.records[eachrecordid]
                copy_recordtext = set(recordtext[-1])
                recordtext[-1].clear()
                for eachtoken in copy_recordtext:
                    recordtext[-1].add(self.regularingtokens[eachtoken])
            self.idf = {}
            for eachrecordid in self.records:
                recordtext = self.records[eachrecordid]
                for eachtoken in recordtext[-1]:
                    if eachtoken in self.idf:
                        self.idf[eachtoken] += 1
                    else:
                        self.idf[eachtoken] = 1
            runtime.regularpattern.grouppatterns_idf_para = self.idf

        runtime.key_token_mining(gml = self)
        self.infer_keytoken[0] = set()
        for target_attribute_index in range(1, len(self.infer_keytoken)):
            self.infer_keytoken[0].update(self.infer_keytoken[target_attribute_index])
        origininferkeytokens = set(self.infer_keytoken[0])

        self.inferregularingtokens = {}
        for eachrecordid in self.referids:
            recordtext = self.records[eachrecordid]
            for eachtoken in recordtext[-1]:
                regular_eachtoken = None
                if self.data.infer_keytoken.patternrestrict is not None:
                    # Already regular for recordtext[-1]
                    regular_eachtoken = eachtoken
                else:
                    regular_eachtoken = self.regularingtokens[eachtoken]
                if eachtoken in origininferkeytokens:
                    if eachtoken not in self.inferregularingtokens:
                        self.inferregularingtokens[eachtoken] = regular_eachtoken
                    else:
                        assert(self.inferregularingtokens[eachtoken] == regular_eachtoken)

        for eachrecordid in self.referids:
            recordtext = self.records[eachrecordid]
            copy_recordtext = set(recordtext[-1])
            if self.data.infer_keytoken.patternrestrict is None:
                # Not yet regular for recordtext[-1]
                recordtext[-1].clear()
                for eachtoken in copy_recordtext:
                    recordtext[-1].add(self.regularingtokens[eachtoken])
            recordtext[-2].clear()
            # Obtain key token in THIS record.
            for eachtoken in copy_recordtext:
                if eachtoken in self.inferregularingtokens:
                    recordtext[-2].add(self.inferregularingtokens[eachtoken])

    def weightedsimilarity(self):
        attributesdiversity = {}
        for eachattributeindex in range(1, len(self.RecordAttributes)):
            attributesdiversity[eachattributeindex] = set()
        for recordid in self.records:
            therecord = self.records[recordid]
            for eachattributeindex in range(1, len(self.RecordAttributes)):
                attributesdiversity[eachattributeindex].add(therecord[eachattributeindex])
        for eachattributeindex in range(1, len(self.RecordAttributes)):
            attributesdiversity[eachattributeindex] = len(attributesdiversity[eachattributeindex])
        basicmetricsdiversity = {}
        basicmetrics = []
        for eachbasicmetricindex in self.basicmetric_columns_indexes:
            if self.raw_basicmetric_columns[eachbasicmetricindex] in self.metafeatures:
                thebasicmetric = fg.metafeature.find(self, self.raw_basicmetric_columns[eachbasicmetricindex])
                basicmetrics.append(thebasicmetric)
                basicmetricsdiversity[thebasicmetric] = attributesdiversity[thebasicmetric.attributeindex]
        allnononevalue_basicmetricsdiversitysum = sum(basicmetricsdiversity.values())
        allnononevalue_basicmetricweights = {}
        existnonvalue_basicmetricsdiversity = dict(basicmetricsdiversity)
        currentexistnonevalue_basicmetricsdiversitysum = None
        currentexistnononevalue_basicmetricweights = {}
        for eachreferredbasicmetric in basicmetrics:
            allnononevalue_basicmetricweights[eachreferredbasicmetric] = basicmetricsdiversity[eachreferredbasicmetric]/allnononevalue_basicmetricsdiversitysum
        for eachpair in self.pairs:
            eachpair.similarity = 0
            allnononevalue = True
            for eachreferredbasicmetric in basicmetrics:
                if eachreferredbasicmetric not in eachpair.metafeatures:
                    allnononevalue = False
                    break
            if allnononevalue == True:
                for eachreferredbasicmetric in basicmetrics:
                    eachpair.similarity += allnononevalue_basicmetricweights[eachreferredbasicmetric] * eachpair.metafeatures[eachreferredbasicmetric]
            else:
                for eachreferredbasicmetric in basicmetrics:
                    if eachreferredbasicmetric not in eachpair.metafeatures:
                        del existnonvalue_basicmetricsdiversity[eachreferredbasicmetric]
                currentexistnonevalue_basicmetricsdiversitysum = sum(existnonvalue_basicmetricsdiversity.values())
                currentexistnononevalue_basicmetricweights.clear()
                for eachreferredbasicmetric in existnonvalue_basicmetricsdiversity:
                    currentexistnononevalue_basicmetricweights[eachreferredbasicmetric] = existnonvalue_basicmetricsdiversity[eachreferredbasicmetric]/currentexistnonevalue_basicmetricsdiversitysum
                for eachreferredbasicmetric in existnonvalue_basicmetricsdiversity:
                    eachpair.similarity += currentexistnononevalue_basicmetricweights[eachreferredbasicmetric] * eachpair.metafeatures[eachreferredbasicmetric]
                currentexistnononevalue_basicmetricweights.clear()
                existnonvalue_basicmetricsdiversity = dict(basicmetricsdiversity)
        for eachpair in self.pairs:
            for eachmetafeature in eachpair.metafeatures:
                if eachpair.metafeatures[eachmetafeature] == metainfo.top.WAITING:
                    eachpair.metafeatures[eachmetafeature] = eachpair.similarity
        for eachfid in self.metafeatures:
            eachmetafeature = self.metafeatures[eachfid]
            eachmetafeature.valuebound = [math.inf, -math.inf]
            for currentmetafeatureeachpair in eachmetafeature.pairs:
                value = currentmetafeatureeachpair.metafeatures[eachmetafeature]
                if value < eachmetafeature.valuebound[0]:
                    eachmetafeature.valuebound[0] = value
                if value > eachmetafeature.valuebound[1]:
                    eachmetafeature.valuebound[1] = value

    def __init__(self, version, data, probegml = None):
        self.version = version
        self.starttime = datetime.datetime.fromtimestamp(time.time())
        self.endtime = None
        self.data = data
        self.datapath = metainfo.top.datapath + data.name + metainfo.top.pathindicator
        self.processpath = metainfo.top.processpath + data.name + metainfo.top.pathindicator
        self.data1 = pd.read_csv(self.datapath + data.data1path, dtype={data.idname: str}, encoding='ISO-8859-1')
        self.data1[data.idname] = self.data1[data.idname].apply(lambda x: str(x).replace(str(x), data.data1path.split('.')[0] + '_' + str(x)))
        if data.data2path != data.data1path:
            self.data2 = pd.read_csv(self.datapath + data.data2path, dtype={data.idname: str}, encoding='ISO-8859-1')
            self.data2[data.idname] = self.data2[data.idname].apply(lambda x: str(x).replace(str(x), data.data2path.split('.')[0] + '_' + str(x)))
        else:
            self.data2 = None
        self.RecordAttributes = self.data1.columns.tolist()
        assert(self.RecordAttributes.index('id') == 0)
        self.RecordAttributes[metainfo.top.ALL_ATTRIBUTES_INDEX] = metainfo.top.ALL_ATTRIBUTES
        self.rawpairs = pd.read_csv(self.datapath + data.pairpath)
        self.rawpairs[data.idname] = self.rawpairs[data.idname].apply(func=self.init_pairid, data1=data.data1path, data2=data.data2path)
        self.records = {}
        self.referids = set()
        self.origintokens = set()
        self.evidenceinterval = runtime.uniforminterval(metainfo.paras.evidenceintervalcount)
        self.infer_keytoken = [None] * len(self.RecordAttributes)
        self.regularingtokens = None
        self.inferregularingtokens = None
        self.preprocesscached_keytokens = True
        self.idf = None
        self.record()
        self.init_token()
        self.w2groups = None
        self.groups2w = None
        if runtime.isNone(self.data.infer_keytoken.patternrestrict) == False:
            self.w2groups = {}
            self.groups2w = {}
            if runtime.regularpattern.ispattern(metainfo.paras.nlpw2vgroups, runtime.regularpattern.numberpattern, runtime.regularpattern.matchway.contain, True) == False:
                if runtime.pickledump(self, 'w2groups', 'e') == False or self.preprocesscached_keytokens == False:
                    if len(self.inferregularingtokens) > 0:
                        assert(self.data.infer_keytoken.patternrestrict != None)
                        pattern, matchway = runtime.regularpattern.index(self.data.infer_keytoken.patternrestrict)
                        for eachtoken in set(self.inferregularingtokens.values()):
                            groupname = runtime.regularpattern.ispattern(eachtoken, pattern, matchway, False)
                            if groupname != False:
                                groupname = 'w2group_' + groupname
                                self.w2groups[eachtoken] = groupname
                        if metainfo.runningflags.refresh_cache == True:
                            runtime.pickledump(self, 'w2groups', 'w')
                else:
                    runtime.pickledump(self, 'w2groups', 'r')
            else:
                metainfo.paras.nlpw2vgroups = runtime.regularpattern.ispattern(metainfo.paras.nlpw2vgroups, runtime.regularpattern.numberpattern, runtime.regularpattern.matchway.contain, False)
                metainfo.paras.nlpw2vgroups = int(metainfo.paras.nlpw2vgroups[0])
                if runtime.pickledump(self, 'w2groups', 'e') == False or self.preprocesscached_keytokens == False:
                    attributed_sentences = []
                    for eachrecord in self.records.values():
                        for eachtextindex in range(1, len(self.RecordAttributes)):
                            thisidfrange = None
                            thisfreqrange = None
                            if type(self.data.infer_keytoken.idfrange) == list:
                                thisidfrange = self.data.infer_keytoken.idfrange[eachtextindex] != 0 and self.data.infer_keytoken.idfrange[eachtextindex] != None and self.data.infer_keytoken.idfrange[eachtextindex] != [0, 0]
                            else:
                                thisidfrange = self.data.infer_keytoken.idfrange != 0 and self.data.infer_keytoken.idfrange != None
                            if type(self.data.infer_keytoken.freqrange) == list:
                                thisfreqrange = self.data.infer_keytoken.freqrange[eachtextindex] != 0 and self.data.infer_keytoken.freqrange[eachtextindex] != None and self.data.infer_keytoken.freqrange[eachtextindex] != [0, 0]
                            else:
                                thisfreqrange = self.data.infer_keytoken.freqrange != 0 and self.data.infer_keytoken.freqrange != None
                            thistexteffective = thisidfrange or thisfreqrange
                            if thistexteffective == True:
                                current_attributed_sentence = eachrecord[eachtextindex].split()
                                for eachtokenindex in range(0, len(current_attributed_sentence)):
                                    currenttoken = current_attributed_sentence[eachtokenindex]
                                    assert(currenttoken in self.regularingtokens)
                                    current_attributed_sentence[eachtokenindex] = self.regularingtokens[currenttoken]
                                if len(current_attributed_sentence) > 0:
                                    attributed_sentences.append(current_attributed_sentence)
                    assert(len(attributed_sentences) > 0)
                    if len(self.inferregularingtokens) > 0:
                        nlp = runtime.nlp(set(self.regularingtokens.values()), attributed_sentences)
                        for eachtoken in set(self.inferregularingtokens.values()):
                            groupname = 'w2group_' + str(nlp.w2groups[eachtoken])
                            self.w2groups[eachtoken] = groupname
                        if metainfo.runningflags.refresh_cache == True:
                            runtime.pickledump(self, 'w2groups', 'w')
                else:
                    runtime.pickledump(self, 'w2groups', 'r')
            if len(self.w2groups) > 0:
                for eachtoken in self.w2groups:
                    groupname = self.w2groups[eachtoken]
                    if groupname not in self.groups2w:
                        self.groups2w[groupname] = []
                    self.groups2w[groupname].append(eachtoken)
                print_groups2w = deepcopy(self.groups2w)
                print_groupscount = 10
                for eachgroup in print_groups2w:
                    ori_print_groups2w_eachgroup = print_groups2w[eachgroup]
                    print_groups2w[eachgroup] = ori_print_groups2w_eachgroup[0:min(10, len(ori_print_groups2w_eachgroup))]
                    print_groupscount -= 1
                    if print_groupscount == 0:
                        break
                runtime.console('SGML > w2groups - anti-sparse', print_groups2w, runtime.console.styles.REPORT)
            else:
                metainfo.paras.nlpw2vgroups = None

        self.metafeatures = {}
        self.pairs = []
        self.truth_1_pairs = set()
        self.test_truth_1_pairs = set()
        self.current_test_truelabel_1_pairs = set()
        self.current_test_label_1_pairs = set()
        self.current_test_truth_1_pairs = set()
        self.test_unlabeledpairs = set()
        self.recall = metainfo.top.NOT_AVAILABLE
        self.precision = metainfo.top.NOT_AVAILABLE
        self.raw_basicmetric_columns = self.rawpairs.columns.tolist()
        self.basicmetric_columns_indexes = []
        self.diff_columns_indexes = []
        self.monotone_nonevalue_transformer = {}

        for each_rawbasicmetric_columns_index in range(2, len(self.raw_basicmetric_columns)):
            metafeaturetype, abbr, attributename, attributeindex, function, parameter = fg.metafeature.rfid(self, self.raw_basicmetric_columns[each_rawbasicmetric_columns_index])
            if metafeaturetype == fg.metafeature.types.BASICMETRIC:
                self.basicmetric_columns_indexes.append(each_rawbasicmetric_columns_index)
            elif metafeaturetype == fg.metafeature.types.DIFF:
                self.diff_columns_indexes.append(each_rawbasicmetric_columns_index)

        for pairindex in range(0, len(self.rawpairs)):
            fg.pair(self, pid = pairindex)
            runtime.consoleprogress(pairindex + 1, len(self.rawpairs), 'init pairs')
        runtime.console('GML > Dataset - ' + self.data.name, 'M ' + str(len(self.truth_1_pairs)) + ' / ' + str(len(self.pairs)) + ', U/M = ' + str(runtime.round(float(len(self.pairs) - len(self.truth_1_pairs)) / len(self.truth_1_pairs))), runtime.console.styles.INFO)

        self.unlabeledpairs = set(self.pairs)

        for eachmetafeature in list(self.metafeatures.values()):
            eachmetafeature.tonormalize()
        self.weightedsimilarity()
        self.metafeatureslist = []

        if self.data.necessary_attribute[1] != None:
            self.data.necessary_attribute[1] = runtime.sublist(self.raw_basicmetric_columns, self.data.necessary_attribute[1])

        if runtime.isNone(self.data.infer_keytoken.patternrestrict) == False:
            samegroup_count = 0
            diffgroup_count = 0
            same_correct = {}
            same_wrong = {}
            diff_correct = {}
            diff_wrong = {}
            bilateral_fids = set()
            for eachfid in self.metafeatures:
                if self.metafeatures[eachfid].type in fg.metafeature.types.bilateral_same:
                    bilateral_fids.add(eachfid)
                    same_correct[eachfid] = 0
                    same_wrong[eachfid] = 0
                    samegroup_count += 1
                elif self.metafeatures[eachfid].type in fg.metafeature.types.bilateral_diff:
                    bilateral_fids.add(eachfid)
                    diff_correct[eachfid] = 0
                    diff_wrong[eachfid] = 0
                    diffgroup_count += 1
            for eachpair in self.pairs:
                if eachpair.truthlabel == 0:
                    for eachmetafeature in eachpair.metafeatures:
                        if eachmetafeature.type in fg.metafeature.types.bilateral_same:
                            same_wrong[eachmetafeature.fid] += 1
                        elif eachmetafeature.type in fg.metafeature.types.bilateral_diff:
                            diff_correct[eachmetafeature.fid] += 1
                else:
                    for eachmetafeature in eachpair.metafeatures:
                        if eachmetafeature.type in fg.metafeature.types.bilateral_same:
                            same_correct[eachmetafeature.fid] += 1
                        elif eachmetafeature.type in fg.metafeature.types.bilateral_diff:
                            diff_wrong[eachmetafeature.fid] += 1
            group_info = {}
            group_info['#samegruop'] = samegroup_count
            group_info['#diffgroup'] = diffgroup_count
            for eachbilateral_fid in bilateral_fids:
                if self.metafeatures[eachbilateral_fid].type in fg.metafeature.types.bilateral_same:
                    group_info[eachbilateral_fid] = [same_correct[eachbilateral_fid], same_wrong[eachbilateral_fid]]
                elif self.metafeatures[eachbilateral_fid].type in fg.metafeature.types.bilateral_diff:
                    group_info[eachbilateral_fid] = [diff_correct[eachbilateral_fid], diff_wrong[eachbilateral_fid]]

            runtime.console('GML > w2groups - ' + str(metainfo.paras.nlpw2vgroups) + ' groups', group_info, runtime.console.styles.OUTLOOK)

        self.data_matrix = None
        self.scalableinference_evidentialsupport = None
        self.scalableinference_approximateentropy = None
        self.scalableinference_inferencecache = None
        self.scalableinference_updatecache_absdirtycount = 0

        self.cache = SGML.cache()

        self.trainingpoolpairs = self.trainingpool()
        self.probegml = None
        self.active_round = None
        self.humancostallowance = None
        self.humancostallowance_thisround = None
        self.discretizefeature = None
        self.humanlabeled_indexes = None
        self.genetics = None
        self.new_genetics_0 = None
        self.new_genetics_1 = None
        self.certified_rules = None
        self.certified_rules_0 = None
        self.certified_rules_1 = None
        self.certified_rules_criterion = None
        self.certified_rules_rawprobability = None
        self.approved_rules = []
        self.processed_recombination_predicates = None
        self.processed_rules_predicates = None
        self.dominate_degenerated_rules_predicates = None
        self.dominate_approved_rules_predicates = None
        self.balance1_multiplier = None
        self.balance1_total = None
        self.recoeffective_map_PF = None
        self.rule_verify = None

        self.ugml_recall = None
        self.ugml_precision = None
        self.ugml_f1 = None
        self.probe_recall = None
        self.probe_precision = None
        self.probe_f1 = None

        self.easyaccuracy = None

        if metainfo.paras.supervised == True:
            if runtime.pickledump(self, 'probegml', 'e') == True:
                runtime.pickledump(self, 'probegml', 'r')
                ugml_truelabel_1 = 0
                ugml_label_1 = 0
                ugml_truth_1 = 0
                for eachpid in range(0, len(self.pairs)):
                    thepair = self.pairs[eachpid]
                    theprobe = self.probegml.probes[eachpid]
                    thepair.ugmllabel = theprobe.label
                    if thepair.pairtype == fg.pair.pairtypes.TESTSET:
                        if thepair.truthlabel == 1:
                            ugml_truth_1 += 1
                            if thepair.ugmllabel == 1:
                                ugml_truelabel_1 += 1
                        if thepair.ugmllabel == 1:
                            ugml_label_1 += 1
                self.ugml_recall = runtime.round(float(ugml_truelabel_1) / ugml_truth_1)
                self.ugml_precision = runtime.round(float(ugml_truelabel_1) / ugml_label_1)
                self.ugml_f1 = runtime.round(float(2) * self.ugml_recall * self.ugml_precision / (self.ugml_recall + self.ugml_precision))

            self.probegml = None
            self.humanlabeled_indexes = []
            if probegml == None and runtime.pickledump(self, 'probegml', 'e') == True:
                runtime.pickledump(self, 'probegml', 'r')
                self.probegml.active_round = 0
            else:
                self.probegml = probegml
                if self.probegml == None:
                    runtime.console('ERROR > probegml', 'probegml must be prerequisite when run Supervised SGML ', runtime.console.styles.EXCEPTION)
                    sys.breakpointhook()
            self.active_round = self.probegml.active_round + 1
            if self.probegml != None:
                probe_activeround = self.probegml.active_round
                probe_truelabel_1 = 0
                probe_label_1 = 0
                probe_truth_1 = 0
                for eachpid in range(0, len(self.pairs)):
                    thepair = self.pairs[eachpid]
                    theprobe = self.probegml.probes[eachpid]
                    thepair.probe.label = theprobe.label
                    thepair.probe.weight = theprobe.weight
                    thepair.probe.probability = theprobe.probability
                    if thepair.pairtype == fg.pair.pairtypes.TESTSET:
                        if thepair.truthlabel == 1:
                            probe_truth_1 += 1
                            if thepair.probe.label == 1:
                                probe_truelabel_1 += 1
                        if thepair.probe.label == 1:
                            probe_label_1 += 1
                self.probe_recall = runtime.round(float(probe_truelabel_1)/probe_truth_1)
                self.probe_precision = runtime.round(float(probe_truelabel_1)/probe_label_1)
                self.probe_f1 = runtime.round(float(2) * self.probe_recall * self.probe_precision / (self.probe_recall + self.probe_precision))
                runtime.console('SGML > ForeRound # ' + str(probe_activeround), 'recall = ' + str(self.probe_recall) + ' , precision = ' + str(self.probe_precision) + ' , f1 = ' + str(self.probe_f1), runtime.console.styles.REPORT)
            self.humancostallowance = int(len(self.pairs) * metainfo.paras.humanproportion)
            self.genetics = {}
            self.new_genetics_0 = {}
            self.new_genetics_1 = {}
            self.certified_rules = {}
            self.certified_rules_0 = {}
            self.certified_rules_1 = {}
            self.certified_rules_criterion = {}
            self.certified_rules_resolution = {}
            self.certified_rules_rawprobability = {}
            self.rule_inherit = []
            self.processed_recombination_predicates = SGML.cache.globalcache.recombination
            self.processed_rules_predicates = {}
            self.dominate_degenerated_rules_predicates = set()
            self.dominate_approved_rules_predicates = set()
            self.GlobalBalance_probability(balance=True, label1count=None, label0count=None)
            discretizefeature(gml = self)
            self.rule_verify = {}
            verifyresult = rule.verifyresult.__dict__
            for each in verifyresult:
                if each[0:2] != '__':
                    self.rule_verify[verifyresult[each]] = 0
            if self.active_round >= 2:
                for eachprobe in self.probegml.probes:
                    thelabelpair = self.pairs[eachprobe.pid]
                    assert(eachprobe.pid == thelabelpair.pid)
                    if eachprobe.labeltype == fg.pair.labeltypes.HUMAN:
                        assert(thelabelpair in self.trainingpoolpairs and thelabelpair.pairtype == fg.pair.pairtypes.TRAININGPOOL)
                        thelabelpair.tolabel(fg.pair.labeltypes.HUMAN)
                if metainfo.paras.forerule_inherit == True:
                    for eachprobe_approved_rule in self.probegml.approved_rules:
                        SGML.probeGML.foreround(eachprobe_approved_rule, gml=self)
            self.humancostallowance_thisround = int(self.humancostallowance / (metainfo.paras.active_rounds - self.active_round + 1))
            if metainfo.paras.skyline_verify_steplimit > self.humancostallowance_thisround:
                metainfo.paras.skyline_verify_steplimit = self.humancostallowance_thisround
            self.GlobalBalance_probability(balance=True, label1count=None, label0count=None)
        else:
            metainfo.paras.humanproportion = 0
            self.humancostallowance = 0
            self.humancostallowance_thisround = 0

        self.results = None
        self.supervised_results = {}
        self.supervised_results['active_round'] = self.active_round
        self.mislabeledinfo = []

        inferenceresults = fg.pair.inferenceresult.__dict__
        for each_inferenceresult in inferenceresults:
            if each_inferenceresult[0:2] != '__':
                self.supervised_results[str(inferenceresults[each_inferenceresult])] = 0

    def init_scalableinference(self):
        for eachmetafeature in list(self.metafeatures.values()):
            if eachmetafeature.for_inference == True:
                self.metafeatureslist.append(eachmetafeature)
        data = []
        row = []
        col = []
        for fid_indexinlist, currentmetafeature in enumerate(self.metafeatureslist):
            currentmetafeature.fid_index = fid_indexinlist
        for pid, currentpair in enumerate(self.pairs):
            assert(currentpair.pid == pid)
            for eachmetafeature in currentpair.metafeatures:
                if eachmetafeature.for_inference == True:
                    fid = eachmetafeature.fid_index
                    value = currentpair.metafeatures[eachmetafeature]
                    data.append(value + metainfo.top.SMALL_VALUE)
                    row.append(pid)
                    col.append(fid)
        self.data_matrix = csr_matrix((data, (row, col)), shape=(len(self.pairs), len(self.metafeatureslist)))
        self.scalableinference_evidentialsupport = []
        self.scalableinference_approximateentropy = []
        self.scalableinference_inferencecache = {}
        self.scalableinference_updatecache_absdirtycount = 0
        self.recoeffective()

    def recoeffective(self):
        if metainfo.method.Rule_Balance == True:
            recoeffective_map_PF = np.zeros((len(self.pairs), len(self.metafeatureslist)), dtype=np.float64)
            for eachfactorindex in range(0, len(self.metafeatureslist)):
                themetafeature = self.metafeatureslist[eachfactorindex]
                if themetafeature.for_inference == True:
                    if themetafeature.ruletype == True:
                        if themetafeature.type == fg.metafeature.types.RULE_0:
                            for eachmetafeaturepair in themetafeature.pairs:
                                recoeffective_map_PF[eachmetafeaturepair.pid, eachfactorindex] = (-1)
                        else:
                            for eachmetafeaturepair in themetafeature.pairs:
                                recoeffective_map_PF[eachmetafeaturepair.pid, eachfactorindex] = 1
                    else:
                        for eachmetafeaturepair in themetafeature.pairs:
                            recoeffective_map_PF[eachmetafeaturepair.pid, eachfactorindex] = np.NaN
            self.recoeffective_map_PF = []
            for eachpairindex in range(0, len(self.pairs)):
                currentpairmap = recoeffective_map_PF[eachpairindex, :]
                rule0_npwhere = np.where(np.logical_and(np.isnan(currentpairmap) == False, currentpairmap < 0))
                rule1_npwhere = np.where(np.logical_and(np.isnan(currentpairmap) == False, currentpairmap > 0))
                ruletype_npwhere = np.where(np.logical_and(np.isnan(currentpairmap) == False, np.logical_or(currentpairmap < 0, currentpairmap > 0)))
                otherinfertype_npwhere = np.where(np.isnan(currentpairmap) == True)
                currentpairmap[rule0_npwhere] *= (-1)
                if len(ruletype_npwhere[0]) == 0:
                    currentpairmap[otherinfertype_npwhere] = 1
                else:
                    all_infer_count = len(ruletype_npwhere[0]) + len(otherinfertype_npwhere[0])
                    ruletype_coefficient_sum = metainfo.paras.ruletype_coefficient * float(all_infer_count)
                    otherinfertype_coefficient = metainfo.paras.otherinfertype_coefficient * float(all_infer_count)
                    if metainfo.method.Rule_Coefficient_Balance == True:
                        if len(rule0_npwhere[0]) > 0 and len(rule1_npwhere[0]) > 0:
                            currentpairmap[rule0_npwhere] = (ruletype_coefficient_sum * 0.5 / len(rule0_npwhere[0]))
                            currentpairmap[rule1_npwhere] = (ruletype_coefficient_sum * 0.5 / len(rule1_npwhere[0]))
                        elif len(rule0_npwhere[0]) > 0 and len(rule1_npwhere[0]) == 0:
                            currentpairmap[rule0_npwhere] = (ruletype_coefficient_sum / len(ruletype_npwhere[0]))
                        elif len(rule0_npwhere[0]) == 0 and len(rule1_npwhere[0]) > 0:
                            currentpairmap[rule1_npwhere] = (ruletype_coefficient_sum / len(ruletype_npwhere[0]))
                    else:
                        currentpairmap[ruletype_npwhere] = (ruletype_coefficient_sum / len(ruletype_npwhere[0]))
                    currentpairmap[otherinfertype_npwhere] = otherinfertype_coefficient / (len(otherinfertype_npwhere[0]))
                self.recoeffective_map_PF.append(currentpairmap)
            self.recoeffective_map_PF = np.array(self.recoeffective_map_PF)
        else:
            self.recoeffective_map_PF = np.ones((len(self.pairs), len(self.metafeatureslist)), dtype=np.float64)
        for eachfactorindex in range(0, len(self.metafeatureslist)):
            themetafeature = self.metafeatureslist[eachfactorindex]
            if themetafeature.for_inference == True:
                themetafeature.coeffective = self.recoeffective_map_PF[:, eachfactorindex]
            else:
                themetafeature.coeffective = np.array([0] * len(self.pairs))

    def del_certified(self, predicate):
        if predicate in self.certified_rules_criterion:
            self.certified_rules_criterion[predicate] = metainfo.top.NOT_AVAILABLE
            self.certified_rules_resolution[predicate] = metainfo.top.NOT_AVAILABLE

    def to_processed_genetics(self):
        processed_new_genetics = collections.namedtuple('processed_new_genetics', ['processed_genetics_0', 'new_genetics_0', 'processed_genetics_1', 'new_genetics_1'])
        processed_genetics_0 = {}
        new_genetics_0 = dict(self.new_genetics_0)
        processed_genetics_1 = {}
        new_genetics_1 = dict(self.new_genetics_1)
        for each_exist_genetic in self.genetics.values():
            if each_exist_genetic.polar0 != None:
                processed_genetics_0[each_exist_genetic] = set(each_exist_genetic.polar0.keys())
            if each_exist_genetic.polar1 != None:
                processed_genetics_1[each_exist_genetic] = set(each_exist_genetic.polar1.keys())
        for each_exist_genetic in self.certified_rules_0:
            processed_genetics_0[each_exist_genetic] = set(self.certified_rules_0[each_exist_genetic])
        for each_exist_genetic in self.certified_rules_1:
            processed_genetics_1[each_exist_genetic] = set(self.certified_rules_1[each_exist_genetic])
        for each_new_genetic in new_genetics_0:
            if each_new_genetic in processed_genetics_0:
                exist_genetics_0_thisgenetic_predicates = processed_genetics_0[each_new_genetic]
                new_genetics_0_thisgenetic_predicates = new_genetics_0[each_new_genetic]
                for each_predicate in new_genetics_0_thisgenetic_predicates:
                    if each_predicate in exist_genetics_0_thisgenetic_predicates:
                        exist_genetics_0_thisgenetic_predicates.remove(each_predicate)
                if len(exist_genetics_0_thisgenetic_predicates) == 0:
                    del processed_genetics_0[each_new_genetic]
        for each_new_genetic in new_genetics_1:
            if each_new_genetic in processed_genetics_1:
                exist_genetics_1_thisgenetic_predicates = processed_genetics_1[each_new_genetic]
                new_genetics_1_thisgenetic_predicates = new_genetics_1[each_new_genetic]
                for each_predicate in new_genetics_1_thisgenetic_predicates:
                    if each_predicate in exist_genetics_1_thisgenetic_predicates:
                        exist_genetics_1_thisgenetic_predicates.remove(each_predicate)
                if len(exist_genetics_1_thisgenetic_predicates) == 0:
                    del processed_genetics_1[each_new_genetic]
        certified_checklist = [processed_genetics_0, new_genetics_0, processed_genetics_1, new_genetics_1]
        def cmp_recombination_cutcount(a, b):
            if a[1] == b[1]:
                return runtime.cmp_reverse(self.certified_rules_resolution[a[0]], self.certified_rules_resolution[b[0]])
            else:
                return runtime.cmp_reverse(a[1], b[1])
        certified_rules_criterion_sortlist = []
        for each_item in self.certified_rules_criterion.items():
            if each_item[1] != metainfo.top.NOT_AVAILABLE and self.certified_rules_resolution[each_item[0]] != metainfo.top.NOT_AVAILABLE:
                certified_rules_criterion_sortlist.append(each_item)
        if metainfo.paras.genetics_recombination_cutcount != None:
            certified_rules_criterion_sortlist.sort(key=functools.cmp_to_key(cmp_recombination_cutcount), reverse=False)
            for each_item in certified_rules_criterion_sortlist:
                assert (each_item[1] != metainfo.top.WAITING and self.certified_rules_resolution[each_item[0]] != metainfo.top.WAITING)
            certified_rules_criterion_sortlist_0 = []
            certified_rules_criterion_sortlist_1 = []
            certified_rules_criterion_0s = set()
            for eachset in self.certified_rules_0.values():
                certified_rules_criterion_0s = certified_rules_criterion_0s.union(eachset)
            certified_rules_criterion_1s = set()
            for eachset in self.certified_rules_1.values():
                certified_rules_criterion_1s = certified_rules_criterion_1s.union(eachset)
            for certified_rules_criterion_sortlist_index in range(0, len(certified_rules_criterion_sortlist)):
                current_certified_rule = certified_rules_criterion_sortlist[certified_rules_criterion_sortlist_index]
                assert(current_certified_rule[0] in certified_rules_criterion_0s or current_certified_rule[0] in certified_rules_criterion_1s)
                if len(certified_rules_criterion_sortlist_0) < metainfo.paras.genetics_recombination_cutcount and current_certified_rule in certified_rules_criterion_0s:
                    certified_rules_criterion_sortlist_0.append(current_certified_rule)
                if len(certified_rules_criterion_sortlist_1) < metainfo.paras.genetics_recombination_cutcount and current_certified_rule in certified_rules_criterion_1s:
                    certified_rules_criterion_sortlist_1.append(current_certified_rule)
            certified_rules_criterion_sortlist = certified_rules_criterion_sortlist_0 + certified_rules_criterion_sortlist_1
        certified_predicates = [eachsortlist[0] for eachsortlist in certified_rules_criterion_sortlist]
        for each_check in certified_checklist:
            delkeys = []
            for each_genetic in each_check:
                if each_genetic in self.certified_rules:
                    this_certified = self.certified_rules[each_genetic]
                    this_genetic_predicates = each_check[each_genetic]
                    delelems = set()
                    for each_genetic_predicates in this_genetic_predicates:
                        if each_genetic_predicates not in this_certified or each_genetic_predicates not in certified_predicates:
                            delelems.add(each_genetic_predicates)
                    for each_delelem in delelems:
                        this_genetic_predicates.remove(each_delelem)
                else:
                    each_check[each_genetic].clear()
                if len(each_check[each_genetic]) == 0:
                    delkeys.append(each_genetic)
            for each_delkey in delkeys:
                del each_check[each_delkey]
        self.new_genetics_0.clear()
        self.new_genetics_1.clear()
        the_processed_new_genetics = processed_new_genetics(processed_genetics_0=processed_genetics_0, new_genetics_0=new_genetics_0, processed_genetics_1=processed_genetics_1, new_genetics_1=new_genetics_1)
        return the_processed_new_genetics

    def GlobalBalance_probability(self, balance, label1count, label0count = None):
        # Global balance, NOT for Local or Subarea.
        label01count = None
        if balance == True:
            if label1count != None:
                if label0count == None:
                    label01count = self.balance1_total
                else:
                    label01count = label0count + self.balance1_multiplier * label1count
                return self.balance1_multiplier * label1count / label01count
            else:
                probe0count = 0
                probe1count = 0
                for eachpair in self.pairs:
                    if eachpair.probe.label == 0:
                        probe0count += 1
                    else:
                        probe1count += 1
                self.balance1_multiplier = float(probe0count)/probe1count
                self.balance1_total = probe0count + self.balance1_multiplier * probe1count
        else:
            # Unbalance regards no polar
            if label0count == None:
                label01count = len(self.pairs)
            else:
                label01count = label0count + label1count
            return float(label1count) / label01count

    def probe(self):
        self.probegml = SGML.probeGML(gml=self)
        if metainfo.paras.supervised == False and metainfo.runningflags.refresh_cache == True:
            runtime.pickledump(self, 'probegml', 'w')

    def clustering(self):
        pairs = self.rawpairs.values
        metafeaturefilterindexlist = self.basicmetric_columns_indexes
        x_input = np.array(pairs)[:, metafeaturefilterindexlist].astype(np.float32)
        y_label = np.array(pairs)[:, 1].astype(np.float32)
        random_state = 170
        km_model = KMeans(n_clusters=2, random_state=random_state).fit(x_input)
        y_pred = km_model.labels_
        cnt = Counter(y_pred)
        smallgroup = min(cnt[0], cnt[1])
        biggroup = max(cnt[0], cnt[1])
        cc = km_model.cluster_centers_
        all_point_distances = euclidean_distances(cc, x_input)
        # [
        #  [all points' distances with the centroid of cluster0],
        #  [all points' distances with the centroid of cluster1],
        #  ...
        # ]
        minority = min(cnt.values())
        label_remap = dict()
        for k, v in cnt.items():
            if v == minority:
                label_remap[k] = 1
                runtime.console.print(0, runtime.console.styles.REPORT, [1], "Clustering > Label 1 Count: ", v)
            else:
                label_remap[k] = -1
                runtime.console.print(0, runtime.console.styles.REPORT, [1], "Clustering > Label 0 Count: ", v)
        _id_2_cluster_label = dict()
        label0id2probability = {}
        label1id2probability = {}
        pair_ids = np.array(pairs)[:, 0].astype(str)
        for i in range(0, len(pair_ids)):
            _assigned_label = label_remap.get(y_pred[i])
            _id_2_cluster_label[pair_ids[i]] = _assigned_label
            _denominator = 0
            for elem in all_point_distances:
                _denominator += elem[i]
            _numerator = all_point_distances[y_pred[i]][i]
            # The probability of being the member of predicted cluster
            # Larger distance, smaller probability
            _cluster_pro = 1.0 - 1.0 * _numerator / _denominator
            # In our setting, we only care the probability of being match
            if _assigned_label == 1:
                _match_pro = _cluster_pro
                label1id2probability[pair_ids[i]] = _match_pro
            else:
                _match_pro = 1 - _cluster_pro
                label0id2probability[pair_ids[i]] = _match_pro
        label1id2probabilitylist = sorted(label1id2probability.items(), key=lambda x: x[1], reverse=True)
        label0id2probabilitylist = sorted(label0id2probability.items(), key=lambda x: x[1], reverse=False)
        return label0id2probabilitylist, label1id2probabilitylist

    def EasyInstanceLabeling(self):
        label0id2probabilitylist, label1id2probabilitylist = self.clustering()
        sortedsimilarity_pairs = sorted(self.pairs, key=lambda x: x.similarity, reverse=False)
        truelabel1count = 0
        ori_basicmetric_columns_indexes = list(self.basicmetric_columns_indexes)
        basicmetric_columns = runtime.sublist(self.raw_basicmetric_columns, self.basicmetric_columns_indexes)
        for index in range(len(sortedsimilarity_pairs) - 1, len(sortedsimilarity_pairs) - 1 - len(label1id2probabilitylist), -1):
            thepair = sortedsimilarity_pairs[index]
            for eachbasicmetricfid in basicmetric_columns:
                thisbasicmetricmetafeature = self.metafeatures[eachbasicmetricfid]
                if thisbasicmetricmetafeature in thepair.metafeatures:
                    currentbasicmetricx = thepair.metafeatures[thisbasicmetricmetafeature]
                    thisbasicmetricmetafeature.label1pairxs[thepair] = currentbasicmetricx
            if thepair.truthlabel == 1:
                truelabel1count += 1
        for index in range(len(sortedsimilarity_pairs) - 1 - len(label1id2probabilitylist), -1, -1):
            thepair = sortedsimilarity_pairs[index]
            for eachbasicmetricfid in basicmetric_columns:
                thisbasicmetricmetafeature = self.metafeatures[eachbasicmetricfid]
                if thisbasicmetricmetafeature in thepair.metafeatures:
                    currentbasicmetricx = thepair.metafeatures[thisbasicmetricmetafeature]
                    thisbasicmetricmetafeature.label0pairxs[thepair] = currentbasicmetricx
        self.basicmetric_columns_indexes.clear()
        for eachbasicmetricfid in basicmetric_columns:
            thisbasicmetric = self.metafeatures[eachbasicmetricfid]
            if thisbasicmetric.update(updateop = fg.metafeature.updateopes.PROBE) == True:
                self.basicmetric_columns_indexes.append(self.raw_basicmetric_columns.index(eachbasicmetricfid))
        self.basicmetric_columns_indexes.sort()
        monotonyeffective_basicmetric_fids = [self.raw_basicmetric_columns[index] for index in self.basicmetric_columns_indexes]

        if len(self.basicmetric_columns_indexes) < len(ori_basicmetric_columns_indexes):
            runtime.console('GML > Influence Modeling - Basic Metrics Monotony Revised; This revising is NOT RESPONSIBLE !', monotonyeffective_basicmetric_fids, runtime.console.styles.EXCEPTION)
            self.weightedsimilarity()
            label0id2probabilitylist, label1id2probabilitylist = self.clustering()
            sortedsimilarity_pairs = sorted(self.pairs, key=lambda x: x.similarity, reverse=False)
            truelabel1count = 0
            for index in range(len(sortedsimilarity_pairs) - 1, len(sortedsimilarity_pairs) - 1 - len(label1id2probabilitylist), -1):
                thepair = sortedsimilarity_pairs[index]
                if thepair.truthlabel == 1:
                    truelabel1count += 1

        runtime.console('vs > CLUSTERING-based SIMILARITY RULE', 'recall = ' + str(runtime.round(float(truelabel1count) / len(self.truth_1_pairs))) + ' , precision = ' + str(runtime.round(float(truelabel1count) / len(label1id2probabilitylist))) + ' , f1 = ' + str(runtime.round(float(truelabel1count) * 2 / (len(self.truth_1_pairs) + len(label1id2probabilitylist)))), runtime.console.styles.INFO)

        if metainfo.paras.easyproportion > 0:
            human0count = 0
            human1count = 0
            if metainfo.paras.supervised == True and self.humanlabeled_indexes != None:
                for eachhumanlabeledpairindex in self.humanlabeled_indexes:
                    if self.pairs[eachhumanlabeledpairindex].truthlabel == 0:
                        human0count += 1
                    else:
                        human1count += 1
            labeled0count = human0count
            labeled1count = human1count
            easy0count = int(len(label0id2probabilitylist) * (metainfo.paras.easyproportion))
            easy1count = int(len(label1id2probabilitylist) * (metainfo.paras.easyproportion))
            if easy0count < labeled0count:
                easy0count = labeled0count
            if easy1count < labeled1count:
                easy1count = labeled1count
            truelabel1count = human1count
            truelabel0count = human0count
            currenteasypair = None
            index = len(sortedsimilarity_pairs) - 1
            while labeled1count < easy1count:
                currenteasypair = sortedsimilarity_pairs[index]
                if currenteasypair.ishumanlabeled() == False:
                    currenteasypair.tolabel(fg.pair.labeltypes.EASY, 1)
                    labeled1count += 1
                    correct = 0
                    if currenteasypair.truthlabel == currenteasypair.label:
                        correct = 1
                    truelabel1count += correct
                index -= 1
            index = 0
            while labeled0count < easy0count:
                currenteasypair = sortedsimilarity_pairs[index]
                if currenteasypair.ishumanlabeled() == False:
                    currenteasypair.tolabel(fg.pair.labeltypes.EASY, 0)
                    labeled0count += 1
                    correct = 0
                    if currenteasypair.truthlabel == currenteasypair.label:
                        correct = 1
                    truelabel0count += correct
                index += 1
            self.easyaccuracy = [runtime.round(float(truelabel0count) / easy0count), runtime.round(float(truelabel1count) / easy1count)]
            runtime.console('GML > Easy Instances Labeling', 'easy label 0 #' + str(easy0count) + ', accuracy: ' + str(self.easyaccuracy[0]), runtime.console.styles.INFO)
            runtime.console(None, 'easy label 1 #' + str(easy1count) + ', accuracy: ' + str(self.easyaccuracy[1]), runtime.console.styles.INFO)
        return

    def EvidentialSupport(self):
        self.scalableinference_evidentialsupport.clear()
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        polarenforce = []
        delta = metainfo.paras.regressiondelta
        zero_confidence = []
        coeffective = []
        for i, f in enumerate(self.metafeatureslist):
            if f.for_inference == True and f.regression.regression is not None and f.regression.monotonyeffective == True:
                coefs.append(f.regression.regression.coef_[0][0])
                intercept.append(f.regression.regression.intercept_[0])
                zero_confidence.append(1)
            else:
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)
            coeffective.append(f.coeffective)
            Ns.append(f.regression.N if f.regression.N > f.regression.effectivetrainingcount else np.NaN)
            residuals.append(f.regression.residual if f.regression.residual is not None else np.NaN)
            meanX.append(f.regression.meanX if f.regression.meanX is not None else np.NaN)
            variance.append(f.regression.variance if f.regression.variance is not None else np.NaN)
            polarenforce.append(f.regression.polarenforce if f.regression.polarenforce is not None else np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance, polarenforce = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], np.array(variance)[col], np.array(polarenforce)[col]
        coeffective = np.array(coeffective).T

        assert(variance.all() != np.NaN and variance.all() != None and variance.all() >= 0)
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(np.logical_and(residuals > 0, variance > 0))] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(np.logical_and(residuals > 0, variance > 0))]
        confidence = confidence * zero_confidence
        evidentialsupport = (1 + confidence) / 2
        for i, r in enumerate(row):
            self.pairs[r].metafeature_evidentialsupport[self.metafeatureslist[col[i]]] = evidentialsupport[i]
        predict = data * coefs + intercept
        espredict = predict * evidentialsupport
        espredict[np.where(polarenforce == 0)] = np.minimum(espredict, 0)[np.where(polarenforce == 0)]
        espredict[np.where(polarenforce == 1)] = np.maximum(espredict, 0)[np.where(polarenforce == 1)]
        espredict = espredict * zero_confidence
        assert len(np.where(evidentialsupport < 0)[0]) == 0
        assert len(np.where((1 - evidentialsupport) < 0)[0]) == 0
        loges = np.log(evidentialsupport)
        logunes = np.log(1 - evidentialsupport)
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.pairs), len(self.metafeatureslist)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.pairs), len(self.metafeatureslist)))
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        regressionweight = csr_matrix((espredict, (row, col)), shape=(len(self.pairs), len(self.metafeatureslist)))
        sum_oparray = np.array([1] * len(self.metafeatureslist)).reshape(-1, 1)
        approximateweights = regressionweight.multiply(coeffective).dot(sum_oparray).reshape(-1)

        for eachpair in self.pairs:
            i = eachpair.pid
            withruleevidentialsupport = p_es[i]
            withruleunevidentialsupport = p_unes[i]
            eachpair.evidentialsupport = withruleevidentialsupport / (withruleevidentialsupport + withruleunevidentialsupport)
            sumsigruleweight = None
            if metainfo.method.Rule_LearnableWeight == False:
                sumsigruleweight = eachpair.ruleweight(detailed=False)
            else:
                sumsigruleweight = 0
            approximateweight = approximateweights[i] + sumsigruleweight
            eachpair.approximateweight, eachpair.approximateprobability, eachpair.approximateentropy = runtime.weight2probabilityentropy(approximateweight)
            if eachpair in self.unlabeledpairs:
                self.scalableinference_evidentialsupport.append([eachpair, eachpair.evidentialsupport])
        if len(self.scalableinference_evidentialsupport) > fg.metainfo.paras.evidentialsupport_topm:
            self.scalableinference_evidentialsupport = heapq.nlargest(metainfo.paras.evidentialsupport_topm, self.scalableinference_evidentialsupport, key=lambda x: x[1])

    def InfluenceModeling(self, initregression):
        if initregression == False:
            self.scalableinference_approximateentropy.clear()
            evidentialpairs = self.scalableinference_evidentialsupport[0: min(metainfo.paras.evidentialsupport_topm, len(self.scalableinference_evidentialsupport))]
            for eachevidentialpair in evidentialpairs:
                self.scalableinference_approximateentropy.append([eachevidentialpair[0], eachevidentialpair[0].approximateentropy])
            if len(self.scalableinference_approximateentropy) > metainfo.paras.approximateentropy_lowestk:
                self.scalableinference_approximateentropy = heapq.nsmallest(metainfo.paras.approximateentropy_lowestk, self.scalableinference_approximateentropy, key=lambda x: x[1])
                self.scalableinference_approximateentropy = self.scalableinference_approximateentropy[0: min(metainfo.paras.approximateentropy_lowestk, len(self.scalableinference_approximateentropy))]
            self.scalableinference_approximateentropy.sort(key=lambda x: x[1], reverse=False)
        else:
            for eachmetafeature in list(self.metafeatures.values()):
                eachmetafeature.regressionmodeling()

    def ScalableInference(self):
        self.init_scalableinference()
        self.scalableinference_updatecache_absdirtycount = 0
        while len(self.unlabeledpairs) > 0:
            if self.scalableinference_updatecache_absdirtycount % metainfo.paras.updatecache_abscapacity == 0:
                self.scalableinference_inferencecache.clear()
                self.EvidentialSupport()
                f1 = self.f1()
                runtime.console('GML > Process ' + str(runtime.round(float(len(self.pairs) - len(self.unlabeledpairs)) * 100 / len(self.pairs))) + ' %', f1, runtime.console.styles.PERIOD)
                runtime.console(None, self.supervised_results, runtime.console.styles.PERIOD)
            self.InfluenceModeling(initregression = False)
            for the_inferingpair_approximatedentropy in self.scalableinference_approximateentropy:
                theinferingpair = the_inferingpair_approximatedentropy[0]
                theapproximatedentropy = the_inferingpair_approximatedentropy[1]
                if theapproximatedentropy <= metainfo.paras.approximateentropy_lowthreshold:
                    theinferingpair.tolabel(labeltype = fg.pair.labeltypes.APPROXIMATE, labelpara = None)
                    self.tolabelupdate(theinferingpair)
                else:
                    if theinferingpair not in self.scalableinference_inferencecache:
                        theinfering = fg.factorgraphinference(theinferingpair)
                        theinfering.performinference()
                        self.scalableinference_inferencecache[theinferingpair] = theinferingpair.entropy
            if len(self.scalableinference_inferencecache) > 0:
                scalableinference_inferencecache_sortlist = sorted(self.scalableinference_inferencecache.items(), key=lambda x: x[1], reverse=False)
                tolabelpair = scalableinference_inferencecache_sortlist[0][0]
                tolabelpair.tolabel(labeltype = fg.pair.labeltypes.INFERENCE, labelpara = None)
                del self.scalableinference_inferencecache[tolabelpair]
                scalableinference_inferencecache_sortlist.clear()
                self.tolabelupdate(tolabelpair)

    def tolabelupdate(self, tolabelpair):
        scalableinference_evidentialsupport_delpairindex = None
        for eachindex in range(0, len(self.scalableinference_evidentialsupport)):
            if self.scalableinference_evidentialsupport[eachindex][0] == tolabelpair:
                scalableinference_evidentialsupport_delpairindex = eachindex
                break
        assert(scalableinference_evidentialsupport_delpairindex != None)
        del self.scalableinference_evidentialsupport[scalableinference_evidentialsupport_delpairindex]

    def trainingpool(self):
        trainingpoolpairs = set()
        validationpoolpairs = set()
        if metainfo.paras.supervised == True:
            trainingpoolselectors = None
            trainingpoolpathprefix = 'trainingpool-'
            validationpathprefix = 'validationpool-'
            trainingpoolselectorcolumnname = (trainingpoolpathprefix + 'pid')
            validationpoolselectorcolumnname = (validationpathprefix + 'pid')
            newtrainingpool = False
            if metainfo.paras.trainingpoolpath == None or metainfo.paras.trainingpoolpath == metainfo.top.AUTO or os.path.exists(metainfo.paras.trainingpoolpath) == False:
                trainingpooldf = None
                trainingpoolpath = trainingpoolpathprefix + runtime.regularpattern.numberpattern + '.csv'
                trainingpoolexistpaths = runtime.searchFile(self.processpath, trainingpoolpath)
                trainingpoolexistpathsindexes = {0:'new'}
                for eachindex in range(0, len(trainingpoolexistpaths)):
                    if metainfo.runningflags.refresh_trainingpool == True:
                        os.remove(trainingpoolexistpaths[eachindex])
                    else:
                        currentexistpath = trainingpoolexistpaths[eachindex][trainingpoolexistpaths[eachindex].rfind(metainfo.top.pathindicator) + 1:]
                        trainingpoolexistpathsindexes[eachindex + 1] = currentexistpath
                runtime.console.print(0, runtime.console.styles.INFO, [0], 'Select a training pool:')
                runtime.console(None, trainingpoolexistpathsindexes, runtime.console.styles.INFO)
                if metainfo.runningflags.refresh_trainingpool == True:
                    trainingpoolexistpaths = runtime.searchFile(self.processpath, trainingpoolpath)
                    assert(len(trainingpoolexistpaths) == 0)
                selector = 0
                if metainfo.paras.trainingpoolpath != metainfo.top.AUTO:
                    selector = int(runtime.awaitinput(0, 10, 'PLEASE ENTER', 'training index', 5))
                if selector == 0:
                    trainingvalidation_combineproportion = metainfo.paras.trainingpoolproportion + metainfo.paras.validationpoolproportion
                    trainingpoolselectors = np.random.choice(range(0, len(self.pairs)), int(trainingvalidation_combineproportion * len(self.pairs)), replace=False)
                    trainingpooldf = pd.DataFrame(columns=[trainingpoolselectorcolumnname])
                    for eachtrainingpoolselector in trainingpoolselectors:
                        trainingpooldf.loc[len(trainingpooldf)] = eachtrainingpoolselector
                        trainingpoolpair = self.pairs[eachtrainingpoolselector]
                        trainingpoolpairs.add(trainingpoolpair)
                    nowtrainingpoolpath = None
                    nowtrainingpoolindicator = 1
                    while(True):
                        nowtrainingpoolpath = self.processpath + trainingpoolpathprefix + str(nowtrainingpoolindicator) + '.csv'
                        if os.path.exists(nowtrainingpoolpath):
                            nowtrainingpoolindicator += 1
                        else:
                            break
                    metainfo.paras.trainingpoolpath = nowtrainingpoolpath
                    trainingpooldf.to_csv(nowtrainingpoolpath, index=False)
                    runtime.console.print(0, runtime.console.styles.INFO, [2], 'SGML > new Training Pool:', trainingpoolpathprefix, nowtrainingpoolindicator, '.csv')
                    newtrainingpool = True
                else:
                    metainfo.paras.trainingpoolpath = self.processpath + trainingpoolexistpathsindexes[selector]
                    trainingpooldf = pd.read_csv(metainfo.paras.trainingpoolpath, dtype={trainingpoolselectorcolumnname:int})
                    trainingpoolselectors = trainingpooldf[trainingpoolselectorcolumnname].tolist()
                    for eachtrainingpoolselector in trainingpoolselectors:
                        trainingpoolpair = self.pairs[eachtrainingpoolselector]
                        trainingpoolpairs.add(trainingpoolpair)
            else:
                trainingpooldf = pd.read_csv(metainfo.paras.trainingpoolpath, dtype={trainingpoolselectorcolumnname: int})
                trainingpoolselectors = trainingpooldf[trainingpoolselectorcolumnname].tolist()
                for eachtrainingpoolselector in trainingpoolselectors:
                    trainingpoolpair = self.pairs[eachtrainingpoolselector]
                    trainingpoolpairs.add(trainingpoolpair)
            if metainfo.paras.validationpoolproportion > 0:
                validationpoolselectors = None
                validationpooldf = None
                validationpoolpath = metainfo.paras.trainingpoolpath.replace(trainingpoolpathprefix, validationpathprefix)
                nowvalidationpoolindicator = validationpoolpath[validationpoolpath.rindex('-') + 1: validationpoolpath.rindex('.')]
                if newtrainingpool == True or os.path.exists(validationpoolpath) == False:
                    if os.path.exists(validationpoolpath) == True:
                        os.remove(validationpoolpath)
                    validationpoolselectors = np.random.choice(trainingpoolselectors, int(metainfo.paras.validationpoolproportion * len(self.pairs)), replace=False)
                    validationpooldf = pd.DataFrame(columns=[validationpoolselectorcolumnname])
                    for eachvalidationpoolselector in validationpoolselectors:
                        validationpooldf.loc[len(validationpooldf)] = eachvalidationpoolselector
                        validationpoolpair = self.pairs[eachvalidationpoolselector]
                        validationpoolpairs.add(validationpoolpair)
                    validationpooldf.to_csv(validationpoolpath, index=False)
                    runtime.console.print(0, runtime.console.styles.INFO, [2], 'SGML > new Validation Pool:', validationpathprefix, nowvalidationpoolindicator, '.csv')
                else:
                    validationpooldf = pd.read_csv(validationpoolpath, dtype={trainingpoolselectorcolumnname: int})
                    validationpoolselectors = validationpooldf[validationpoolselectorcolumnname].tolist()
                    for eachvalidationpoolselector in validationpoolselectors:
                        validationpoolpair = self.pairs[eachvalidationpoolselector]
                        validationpoolpairs.add(validationpoolpair)
                assert(set(validationpoolselectors).intersection(set(trainingpoolselectors)) == set(validationpoolselectors))
                for eachtrainingpoolpair in validationpoolpairs:
                    eachtrainingpoolpair.pairtype = fg.pair.pairtypes.VALIDATIONPOOL
        trainingpoolpairs = trainingpoolpairs - validationpoolpairs
        for eachtrainingpoolpair in trainingpoolpairs:
            eachtrainingpoolpair.pairtype = fg.pair.pairtypes.TRAININGPOOL
        testpairs = set(self.pairs) - trainingpoolpairs - validationpoolpairs
        for eachtestpair in testpairs:
            eachtestpair.pairtype = fg.pair.pairtypes.TESTSET
            self.test_unlabeledpairs.add(eachtestpair)
            if eachtestpair.truthlabel == 1:
                self.test_truth_1_pairs.add(eachtestpair)
        if metainfo.paras.supervised == True:
            assert(round(float(len(trainingpoolpairs)) / len(self.pairs) - metainfo.paras.trainingpoolproportion, 2) == 0)
            assert(round(float(len(validationpoolpairs)) / len(self.pairs) - metainfo.paras.validationpoolproportion, 2) == 0)
            assert(round(float(len(testpairs)) / len(self.pairs) - (1 - metainfo.paras.trainingpoolproportion - metainfo.paras.validationpoolproportion), 2) == 0)
        return trainingpoolpairs

    def f1(self):
        f1 = {}
        if len(self.current_test_truth_1_pairs) > 0:
            self.recall = float(len(self.current_test_truelabel_1_pairs)) / len(self.current_test_truth_1_pairs)
        if len(self.current_test_label_1_pairs) > 0:
            self.precision = float(len(self.current_test_truelabel_1_pairs)) / len(self.current_test_label_1_pairs)
        f1['round'] = self.active_round
        if len(self.current_test_truth_1_pairs) > 0 and len(self.current_test_label_1_pairs) > 0:
            f1['F1'] = runtime.round(float(2) * self.recall * self.precision / (self.recall + self.precision))
        else:
            f1['F1'] = metainfo.top.NOT_AVAILABLE
        f1['recall'] = runtime.round(self.recall)
        f1['precision'] = runtime.round(self.precision)
        f1['# A, T unlabeled'] = str(len(self.unlabeledpairs)) + ', ' + str(len(self.test_unlabeledpairs))
        f1['# T unlabeled0'] = len(self.test_unlabeledpairs) - (len(self.test_truth_1_pairs) - len(self.current_test_truth_1_pairs))
        f1['# T unlabeled1'] = len(self.test_truth_1_pairs) - len(self.current_test_truth_1_pairs)
        return f1

    def saveresult(self):
        if os.path.exists(metainfo.top.resultspath) == False:
            os.mkdir(metainfo.top.resultspath)
        csvfilepath = metainfo.top.resultspath + self.data.name + '_' + self.version + '.csv'
        csvfile = open(csvfilepath, 'a', newline='')
        self.endtime = datetime.datetime.fromtimestamp(time.time())
        self.results = self.f1()
        if metainfo.paras.supervised == True and self.active_round == metainfo.paras.active_rounds:
            self.results['ugml_promote'] = runtime.round(self.results['F1'] - self.ugml_f1)
            self.results['ugml_F1'] = runtime.round(self.ugml_f1)
            self.results['ugml_recall'] = runtime.round(self.ugml_recall)
            self.results['ugml_precision'] = runtime.round(self.ugml_precision)
        else:
            self.results['ugml_promote'] = metainfo.top.NOT_AVAILABLE
            self.results['ugml_F1'] = metainfo.top.NOT_AVAILABLE
            self.results['ugml_recall'] = metainfo.top.NOT_AVAILABLE
            self.results['ugml_precision'] = metainfo.top.NOT_AVAILABLE
        if metainfo.paras.supervised == True:
            self.results['prev_promote'] = runtime.round(self.results['F1'] - self.probe_f1)
        else:
            self.results['prev_promote'] = metainfo.top.NOT_AVAILABLE
        self.results['% humancost'] = runtime.round(metainfo.paras.humanproportion - float(self.humancostallowance) / len(self.pairs))
        assert(self.results['# T unlabeled0'] == 0 and self.results['# T unlabeled1'] == 0)
        del self.results['# A, T unlabeled']
        del self.results['# T unlabeled0']
        del self.results['# T unlabeled1']
        runtime.console('GML > round ' + str(self.active_round) + ' FINISH >>>', self.results, runtime.console.styles.PERIOD)
        self.results['LUID'] = self.version + '(' + str(self.starttime) + ')'
        self.results['Time'] = str(self.endtime - self.starttime)
        self.results['easyacc'] = str(self.easyaccuracy)
        for each_supervised_result in self.supervised_results:
            self.results[each_supervised_result] = str(self.supervised_results[each_supervised_result])
        GMLparas = metainfo.paras.__dict__
        for eachpara in GMLparas:
            if eachpara[0:2] != '__':
                if runtime.isnumber(GMLparas[eachpara]) and GMLparas[eachpara] == math.e:
                    self.results[str(eachpara)] = 'math.e'
                else:
                    self.results[str(eachpara)] = str(GMLparas[eachpara])
        methods = metainfo.method.__dict__
        for eachmethod in methods:
            if eachmethod[0:2] != '__':
                if runtime.isnumber(methods[eachmethod]) and methods[eachmethod] == math.e:
                    self.results[str(eachmethod)] = 'math.e'
                else:
                    self.results[str(eachmethod)] = str(methods[eachmethod])
        if metainfo.top.OBSOLETE and metainfo.runningflags.Show_Detail == True and metainfo.paras.supervised == True:
            self.results['map_unitfeatures'] = self.discretizefeature.map_unitfeatures
            self.results['monotone_nonevalue_transformer'] = self.monotone_nonevalue_transformer
        if metainfo.paras.supervised == True:
            self.results['trainingpoolpath'] = self.results['trainingpoolpath'][self.results['trainingpoolpath'].rfind(metainfo.top.pathindicator) + 1:]
            runtime.console(None, 'on ' + self.results['trainingpoolpath'], runtime.console.styles.INFO)
        del self.results['para_adapt']
        del self.results['para_adapt_formula']
        del self.results['eachdataset']
        dictcontent = ''
        csv_rule_header = 'rules'
        result_rules_header = ['polar', 'raw certified', 'v horizon prob', 'truth correcting', 'actual correcting', 'L pred', 'predicates']
        result_rules_space  = [7      , 12             , 12              , 25                , 25                 , 7       , 50          ]
        result_rule_display = runtime.display(result_rules_header, result_rules_space, dictcontent=dictcontent)
        if dictcontent == False:
            self.results[csv_rule_header] = result_rule_display
        else:
            for eachkey in result_rule_display:
                result_rule_display[eachkey] = ''
            self.results.update(result_rule_display)
        csvwriter = csv.DictWriter(csvfile, fieldnames=list(self.results.keys()))
        csvwriter.writeheader()
        csvwriter.writerow(self.results)
        blankresults = dict(self.results)
        for eachresult in blankresults:
            blankresults[eachresult] = ''
        for eachrule in self.approved_rules:
            current_result_rule_results = dict(blankresults)
            current_result_rule = []
            current_result_rule.append(eachrule.polar)                                               # polar
            current_result_rule.append(eachrule.raw_certified)                                       # raw certified
            current_result_rule.append(eachrule.probability)                                         # verified probability
            current_result_rule.append(eachrule.truth_correcting)                                    # truth correcting
            current_result_rule.append(eachrule.actual_correcting)                                   # actual_correcting
            current_result_rule.append(len(eachrule.predicates))                                     # L pred
            current_result_rule.append(eachrule.predicatedisplays)                                   # predicates
            current_result_rule_display = runtime.display(result_rules_header, result_rules_space, dictcontent=current_result_rule)
            if dictcontent == False:
                current_result_rule_results[csv_rule_header] = current_result_rule_display
            else:
                current_result_rule_results.update(current_result_rule_display)
            csvwriter.writerow(current_result_rule_results)
        csvfile.close()
        if len(self.mislabeledinfo) > 0:
            csvfilepath = metainfo.top.resultspath + self.data.name + '_' + self.version + '_mislabeled.csv'
            existed = os.path.exists(csvfilepath)
            csvfile = open(csvfilepath, 'a', newline='')
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(self.mislabeledinfo[0].keys()))
            if existed == False:
                csvwriter.writeheader()
            for eachmislabeledinfo in self.mislabeledinfo:
                csvwriter.writerow(eachmislabeledinfo)
        csvfile.close()
        # rule.inherit.settle('C:\\Users\\TFU\\Desktop\\example_' + self.data.name + '.csv', self.rule_inherit)

    def active_ruleselect(gml):
        #gml.active_rulegenerate(probe=True)
        gml.active_recombination()
        candidaterules = gml.active_rulegenerate(probe=False)
        if len(candidaterules) == 0:
            return candidaterules
        def cmp_active_rulereward(a, b):
            if a.expectation_roi == b.expectation_roi:
                return runtime.cmp_reverse(a.resolution, b.resolution)
            else:
                return runtime.cmp_reverse(a.expectation_roi, b.expectation_roi)
        candidaterules.sort(key=functools.cmp_to_key(cmp_active_rulereward), reverse=False)
        delindexes = []
        for eachindex in range(0, len(candidaterules)):
            eachcandidaterule = candidaterules[eachindex]
            if eachcandidaterule.require_new_verification_allowance == 0:
                eachcandidaterule.verify()
                eachcandidaterule.reportresult()
                delindexes.append(eachindex)
        oricandidaterules = list(candidaterules)
        candidaterules = runtime.dellist(candidaterules, delindexes)
        if len(candidaterules) > 0:
            selectedverifyingrule = candidaterules[0]
            selectedverifyingrule.verify()
            selectedverifyingrule.reportresult()
            runtime.console(None, gml.rule_verify, runtime.console.styles.REPORT)
        return oricandidaterules

    # Correcting Reward suffers from the quality of Skyline, therefore Genetics Guidance is also important to help to alleviate.
    def active_rulegenerate(gml, probe):
        candidaterules = []
        candidategenetics = []
        for eachgenetic in gml.genetics.values():
            for eachpolar in [0, 1]:
                currentcandidategenetic = genetic.candidategenetic(eachgenetic, eachpolar)
                if currentcandidategenetic.effective == True:
                    candidategenetics.append(currentcandidategenetic)
        processed = 0
        for eachindex in range(0, len(candidategenetics)):
            if probe == False:
                runtime.consoleprogress(processed, len(candidategenetics), 'generating candidate rules # ' + str(len(candidaterules)))
            currentcandidategenetic = candidategenetics[eachindex]
            currentcandidategenetic.torule(probe)
            for eachcandidaterule in currentcandidategenetic.candidaterules:
                candidaterules.append(eachcandidaterule)
            processed += 1
        if probe == False:
            runtime.consoleprogress(processed, len(candidategenetics), 'generating candidate rules # ' + str(len(candidaterules)))
        return candidaterules

    def active_recombination(self):
        the_processed_new_genetics = self.to_processed_genetics()
        processed_genetics_0s = the_processed_new_genetics.processed_genetics_0
        new_genetics_0s = the_processed_new_genetics.new_genetics_0
        processed_genetics_1s = the_processed_new_genetics.processed_genetics_1
        new_genetics_1s = the_processed_new_genetics.new_genetics_1
        new_recombination_genetics_0_count = 0
        new_recombination_genetics_1_count = 0
        processedcount = len(processed_genetics_0s) + len(processed_genetics_1s)
        newcount = len(new_genetics_0s) + len(new_genetics_1s)
        runtime.console.print(0, runtime.console.styles.INFO, [1, 3], 'genetics recombination : #', processedcount, ' , +', newcount)
        donetimes = 0
        handled_genetics = set()
        for new_genetics_0 in new_genetics_0s:
            handled_genetics.add(new_genetics_0)
            new_genetics_0_values = new_genetics_0s[new_genetics_0]
            for processed_genetics_0 in processed_genetics_0s:
                processed_genetics_0_values = processed_genetics_0s[processed_genetics_0]
                for each_new_genetics_0_value in new_genetics_0_values:
                    for each_processed_genetics_0_value in processed_genetics_0_values:
                        new_recombination_genetics_0s = runtime.recombination(each_new_genetics_0_value, each_processed_genetics_0_value, metainfo.paras.tree_maxdepth, semisub=True, processed_cache=self.processed_recombination_predicates)
                        if new_recombination_genetics_0s != None:
                            for new_recombination_genetics_0 in new_recombination_genetics_0s:
                                inherit = genetic.inherit(self, new_recombination_genetics_0, polar=0)
                                isnew, thegenetic = genetic.find(gml=self, rule_or_inherit=inherit)
                                rule.inherit(inherit_type=rule.inherit.types.RECOMBINATION, target_rule=inherit, flow1=each_new_genetics_0_value, flow2=each_processed_genetics_0_value)
                                new_recombination_genetics_0_count += isnew
            for new_genetics_0_another in new_genetics_0s:
                if new_genetics_0_another not in handled_genetics:
                    new_genetics_0_another_values = new_genetics_0s[new_genetics_0_another]
                    for each_new_genetics_0_value in new_genetics_0_values:
                        for each_new_genetics_0_another_value in new_genetics_0_another_values:
                            new_recombination_genetics_0s = runtime.recombination(each_new_genetics_0_value, each_new_genetics_0_another_value, metainfo.paras.tree_maxdepth, semisub=True, processed_cache=self.processed_recombination_predicates)
                            if new_recombination_genetics_0s != None:
                                for new_recombination_genetics_0 in new_recombination_genetics_0s:
                                    inherit = genetic.inherit(self, new_recombination_genetics_0, polar=0)
                                    isnew, thegenetic = genetic.find(gml=self, rule_or_inherit=inherit)
                                    rule.inherit(inherit_type=rule.inherit.types.RECOMBINATION, target_rule=inherit, flow1=each_new_genetics_0_value, flow2=each_new_genetics_0_another_value)
                                    new_recombination_genetics_0_count += isnew
            donetimes += 1
        handled_genetics.clear()
        for new_genetics_1 in new_genetics_1s:
            handled_genetics.add(new_genetics_1)
            new_genetics_1_values = new_genetics_1s[new_genetics_1]
            for processed_genetics_1 in processed_genetics_1s:
                processed_genetics_1_values = processed_genetics_1s[processed_genetics_1]
                for each_new_genetics_1_value in new_genetics_1_values:
                    for each_processed_genetics_1_value in processed_genetics_1_values:
                        new_recombination_genetics_1s = runtime.recombination(each_new_genetics_1_value, each_processed_genetics_1_value, metainfo.paras.tree_maxdepth, semisub=True, processed_cache=self.processed_recombination_predicates)
                        if new_recombination_genetics_1s != None:
                            for new_recombination_genetics_1 in new_recombination_genetics_1s:
                                inherit = genetic.inherit(self, new_recombination_genetics_1, polar=1)
                                isnew, thegenetic = genetic.find(gml=self, rule_or_inherit=inherit)
                                rule.inherit(inherit_type=rule.inherit.types.RECOMBINATION, target_rule=inherit, flow1=each_new_genetics_1_value, flow2=each_processed_genetics_1_value)
                                new_recombination_genetics_1_count += isnew
            for new_genetics_1_another in new_genetics_1s:
                if new_genetics_1_another not in handled_genetics:
                    new_genetics_1_another_values = new_genetics_1s[new_genetics_1_another]
                    for each_new_genetics_1_value in new_genetics_1_values:
                        for each_new_genetics_1_another_value in new_genetics_1_another_values:
                            new_recombination_genetics_1s = runtime.recombination(each_new_genetics_1_value, each_new_genetics_1_another_value, metainfo.paras.tree_maxdepth, semisub=True, processed_cache=self.processed_recombination_predicates)
                            if new_recombination_genetics_1s != None:
                                for new_recombination_genetics_1 in new_recombination_genetics_1s:
                                    inherit = genetic.inherit(self, new_recombination_genetics_1, polar=1)
                                    isnew, thegenetic = genetic.find(gml=self, rule_or_inherit=inherit)
                                    rule.inherit(inherit_type=rule.inherit.types.RECOMBINATION, target_rule=inherit, flow1=each_new_genetics_1_value, flow2=each_new_genetics_1_another_value)
                                    new_recombination_genetics_1_count += isnew
            donetimes += 1
        info = {}
        info['# new combine 0'] = new_recombination_genetics_0_count
        info['# new combine 1'] = new_recombination_genetics_1_count
        runtime.console('SGML > recombination', info, runtime.console.styles.STRESS)

    def active_geneticinit(self):
        init_rules_0 = []
        init_rules_1 = []
        mainmap = self.discretizefeature.discretize_map
        add0count = 0
        add1count = 0
        runtime.console('SGML > active_geneticinit', None, runtime.console.styles.INFO)
        for polar in [0, 1]:
            for roottree in range(0, len(self.discretizefeature.map_unitfeatures)):
                roottrees = [roottree]
                genetics_forest = runtime.forest(balance=False, mainmap_knowledgeupdating=mainmap, labelindex=-1,
                                                 splitters=self.discretizefeature.map_unitfeatures, polar=polar,
                                                 weight=confidence.weight, probability=self.GlobalBalance_probability,
                                                 confidence_coefficient=confidence.confidence_coefficient,
                                                 premilinary_condition_predicates=None,
                                                 nondirectional_map=None, conform_map=None,
                                                 roottrees=roottrees)
                if polar == 0:
                    for eachroottree in roottrees:
                        add0count += len(genetics_forest.rules[eachroottree])
                        for each_rule in genetics_forest.rules[eachroottree]:
                            init_rules_0.append(each_rule)
                    init_rules_0.sort(key=lambda x: x.criterion, reverse=True)
                else:
                    for eachroottree in roottrees:
                        add1count += len(genetics_forest.rules[eachroottree])
                        for each_rule in genetics_forest.rules[eachroottree]:
                            init_rules_1.append(each_rule)
                    init_rules_1.sort(key=lambda x: x.criterion, reverse=True)
                if polar == 0:
                    runtime.consoleprogress(roottree + 1, len(self.discretizefeature.map_unitfeatures), 'generating forest polar 0 : # ' + str(add0count))
                else:
                    runtime.consoleprogress(roottree + 1, len(self.discretizefeature.map_unitfeatures), 'generating forest polar 1 : # ' + str(add1count))
        init_rules = init_rules_0 + init_rules_1
        totalrules = len(init_rules)
        add0count = 0
        add1count = 0
        if len(init_rules) > 0:
            processed = 0
            for each_rule in init_rules:
                isnew, this_genetic = genetic.find(gml=self, rule_or_inherit=each_rule)
                each_rule.predicatedisplays = ''
                for eachpredicate in each_rule.predicates:
                    assert (eachpredicate.valueex == None)
                    currentpredicatedescribe = eachpredicate.print()
                    if len(each_rule.predicatedisplays) > 0:
                        each_rule.predicatedisplays += ' & '
                    each_rule.predicatedisplays += currentpredicatedescribe
                each_rule.gml = self
                rule.inherit(inherit_type=rule.inherit.types.INIT, target_rule=each_rule, flow1=each_rule.stat_probability, flow2=None)
                processed += 1
                runtime.consoleprogress(processed, len(init_rules), 'generating genetics ' + str(processed) + ' | ' + str(totalrules))
                if each_rule.polar == 0:
                    add0count += isnew
                else:
                    add1count += isnew
        runtime.console.print(1, runtime.console.styles.INFO, [1, 3], 'generating # genetics =', add0count, ' & ', add1count)

    def activelearning(self):
        if metainfo.paras.supervised == True:
            self.active_geneticinit()
            total_humancostallowance_thisround = self.humancostallowance_thisround
            newverification_1moreround_toapprove0require = None
            while self.humancostallowance_thisround > 0 or newverification_1moreround_toapprove0require == True:
                humancostallowance_previoussubround = self.humancostallowance_thisround
                runtime.console('SGML > active_ruleselect', str(self.humancostallowance_thisround) + ' lefted, ' + str(total_humancostallowance_thisround) + ' total.', runtime.console.styles.REPORT)
                candidaterules = self.active_ruleselect()
                if len(candidaterules) == 0:
                    if metainfo.paras.skyline_verify_steplimit_adjust != None and metainfo.paras.skyline_verify_steplimit <= self.humancostallowance_thisround:
                        previous_skyline_verify_steplimit = metainfo.paras.skyline_verify_steplimit
                        metainfo.paras.skyline_verify_steplimit += metainfo.paras.skyline_verify_steplimit_adjust
                        runtime.console('SGML > adjust steplimit', 'skyline_verify_steplimit : ' + str(previous_skyline_verify_steplimit) + '  ➡️' + str(metainfo.paras.skyline_verify_steplimit), runtime.console.styles.EXCEPTION)
                    else:
                        break
                if metainfo.paras.genetics_recombination_cutcount_adjust != None:
                    metainfo.paras.genetics_recombination_cutcount += metainfo.paras.genetics_recombination_cutcount_adjust
                newverification_1moreround_toapprove0require = ((humancostallowance_previoussubround - self.humancostallowance_thisround) > 0)
