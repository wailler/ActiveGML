from source import metainfo
from source.runtime import runtime
from source import fg
import numpy as np
import pandas as pd
import math
from sklearn import mixture
from paretoset import paretoset
from copy import deepcopy
from scipy.special import logit, expit
from numba import jit, njit, prange
from enum import Flag, auto
import collections
import functools
import sys
import csv

class discretizefeature:

    class discretizemetafeature:

        def __init__(self, discretizefeature, unitfeatureindex):
            self.discretizefeature = discretizefeature
            assert(self.discretizefeature.map_unitfeatures_nonevalueprocessed)
            self.unitfeature = self.discretizefeature.map_unitfeatures[unitfeatureindex]
            self.pairs = {}
            self.ranges = None
            self.description = None
            X = np.array(self.discretizefeature.raw_map[:, unitfeatureindex])
            # Only effective values handling.
            ef_X_indexes = list(np.where(X != metainfo.top.NONE_VALUE)[0])
            for_discretizing = self.unitfeature in runtime.sublist(self.discretizefeature.gml.raw_basicmetric_columns, self.discretizefeature.gml.basicmetric_columns_indexes) and self.discretizefeature.gml.metafeatures[self.unitfeature].normalize == False
            if for_discretizing == False:
                self.discretizefeature.map_unitfeature_category.append(self.unitfeature)
            valid = False
            retry = metainfo.top.MAX_RETRY
            initarea = metainfo.paras.discretize_splitcount_min
            splitarea = metainfo.paras.discretize_splitcount_max
            while valid == False:
                valid = True
                ef_X = X[ef_X_indexes]
                Y = None
                if for_discretizing == True:
                    ef_X = ef_X.reshape(-1, 1)
                    gmm = None
                    ic = math.inf
                    for n in range(initarea, splitarea + 1):
                        currentgmm = mixture.GaussianMixture(n_components=n, covariance_type='full', init_params='kmeans', random_state=None)
                        currentgmm.fit(ef_X)
                        currentic = currentgmm.bic(ef_X)
                        if currentgmm.converged_ == True:
                            if currentic < ic:
                                gmm = currentgmm
                                ic = currentic
                    if gmm == None:
                        runtime.console('SGML > discretize > GaussianMixture uncoveraged', str(self.unitfeature), runtime.console.styles.STRESS)
                        Y = [1] * self.discretizefeature.raw_map.shape[0]
                    else:
                        Y = gmm.predict(ef_X)
                else:
                    Y = np.array(ef_X)
                discretize = list(np.unique(Y))
                discretize.sort()
                pairs = {}
                ranges = {}
                sorting = {}
                description = {}
                for eachdiscretize in discretize:
                    pairs[eachdiscretize] = []
                for eachindex_of_ef_X_index in range(0, len(ef_X_indexes)):
                    currentpairindex = ef_X_indexes[eachindex_of_ef_X_index]
                    currentpairy = Y[eachindex_of_ef_X_index]
                    pairs[currentpairy].append(self.discretizefeature.gml.pairs[currentpairindex])
                for eachdiscretize in discretize:
                    pairs[eachdiscretize].sort(key=lambda x:self.discretizefeature.raw_map[x.pid, unitfeatureindex], reverse=False)
                    ranges[eachdiscretize] = [self.discretizefeature.raw_map[pairs[eachdiscretize][0].pid, unitfeatureindex], self.discretizefeature.raw_map[pairs[eachdiscretize][-1].pid, unitfeatureindex]]
                    assert(ranges[eachdiscretize][0] <= ranges[eachdiscretize][1])
                    if ranges[eachdiscretize][0] < ranges[eachdiscretize][1]:
                        description[eachdiscretize] = runtime.predicate(self.unitfeature, runtime.predicate.op.lesseq, ranges[eachdiscretize][0], ranges[eachdiscretize][1])
                    else:
                        description[eachdiscretize] = runtime.predicate(self.unitfeature, runtime.predicate.op.eq, ranges[eachdiscretize][1])
                    sorting[eachdiscretize] = sum(ranges[eachdiscretize])
                if for_discretizing == True:
                    self.ranges = {}
                    self.description = {}
                    sorteditemlist = sorted(sorting.items(), key=lambda x:x[1], reverse=False)
                    for eachsorteditemindex in range(0, len(sorteditemlist)):
                        eachsorteditem = sorteditemlist[eachsorteditemindex]
                        eachsorteditem_discretize = eachsorteditem[0]
                        # Design from 0.
                        self.pairs[eachsorteditemindex] = pairs[eachsorteditem_discretize]
                        for eachpair_currentdiscretize in self.pairs[eachsorteditemindex]:
                            self.discretizefeature.discretize_map[eachpair_currentdiscretize.pid, unitfeatureindex] = eachsorteditemindex
                        self.ranges[eachsorteditemindex] = ranges[eachsorteditem_discretize]
                        self.description[eachsorteditemindex] = description[eachsorteditem_discretize]
                    pairs.clear()
                    ranges.clear()
                    description.clear()
                    previousmaxvalue = (-1) * math.inf
                    for eachrange in self.ranges:
                        currentminvalue = self.ranges[eachrange][0]
                        if currentminvalue <= previousmaxvalue:
                            valid = False
                            #info = {'currentminvalue':runtime.round(currentminvalue), 'previousmaxvalue':runtime.round(previousmaxvalue)}
                            #runtime.console('WARNING ! SGML > discretize range error', info, runtime.console.styles.EXCEPTION)
                            break
                        previousmaxvalue = self.ranges[eachrange][1]
                    if valid == False:
                        retry -= 1
                        if retry == 0:
                            initarea = splitarea
                            splitarea += 1
                            retry = metainfo.top.MAX_RETRY
                        continue
                    assert(len(self.ranges) >= 2)
                    self.description[0].singlebound(runtime.predicate.op.lesseq)
                    self.description[len(sorteditemlist) - 1].singlebound(runtime.predicate.op.largereq)
                    self.discretizefeature.gml.monotone_nonevalue_transformer[unitfeatureindex] = runtime.minmax(min(self.pairs.keys()), max(self.pairs.keys()))
                else:
                    self.ranges = metainfo.top.NOT_AVAILABLE
                    self.description = metainfo.top.NOT_AVAILABLE
                    if self.unitfeature in self.discretizefeature.diff_columns:
                        # Diff feature is Reversed Monotone and clustering discretized is the most comfortable absolute value for distance computing.
                        # self.pairs already excepted none values, only effective values handling.
                        monotone_transformer = max(pairs.keys())
                        for eachsorteditemindex in pairs:
                            monotony_transformer_value = monotone_transformer - eachsorteditemindex
                            self.pairs[monotony_transformer_value] = pairs[eachsorteditemindex]
                            for eachpair_currentdiscretize in self.pairs[monotony_transformer_value]:
                                self.discretizefeature.discretize_map[eachpair_currentdiscretize.pid, unitfeatureindex] = monotony_transformer_value
                    else:
                        self.pairs = pairs
                    self.discretizefeature.gml.monotone_nonevalue_transformer[unitfeatureindex] = runtime.minmax(min(self.pairs.keys()), max(self.pairs.keys()))
            self.discretizefeature.features[self.unitfeature] = self

    def subarea(self):
        for eachpair_1 in self.gml.trainingpoolpairs:
            if eachpair_1 not in self.subareas_0:
                samesubarea = []
                for eachpair_2 in self.gml.trainingpoolpairs:
                    if eachpair_2 not in self.subareas_0:
                        if self.gml.discretizefeature.issamesubarea(eachpair_1.pid, eachpair_2.pid, columns=None, polar=0) == True:
                            samesubarea.append(eachpair_2)
                samesubarea.sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)
                self.subareas_0[eachpair_1] = samesubarea
                for eachsamesubareapair in samesubarea:
                    self.subareas_0[eachsamesubareapair] = samesubarea
        for eachpair_1 in self.gml.trainingpoolpairs:
            if eachpair_1 not in self.subareas_1:
                samesubarea = []
                for eachpair_2 in self.gml.trainingpoolpairs:
                    if eachpair_2 not in self.subareas_1:
                        if self.gml.discretizefeature.issamesubarea(eachpair_1.pid, eachpair_2.pid, columns=None, polar=1) == True:
                            samesubarea.append(eachpair_2)
                samesubarea.sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)
                self.subareas_1[eachpair_1] = samesubarea
                for eachsamesubareapair in samesubarea:
                    self.subareas_1[eachsamesubareapair] = samesubarea

    def map(self):
        # map columns: [ UNITFEATURE TABLES, GML LABEL, PROBE LABEL ]
        # Auti Token Sparse problem fundamentally, instead of using Training-Test's Seeming Trick.
        basicmetric_columns_indexes = self.gml.basicmetric_columns_indexes
        if metainfo.paras.genetics_basicmetric_selector != None:
            basicmetric_columns_indexes = list(set(self.gml.basicmetric_columns_indexes).intersection(metainfo.paras.genetics_basicmetric_selector))
            basicmetric_columns_indexes.sort()
        basicmetric_columns = runtime.sublist(self.gml.raw_basicmetric_columns, basicmetric_columns_indexes)
        runtime.console.print(0, runtime.console.styles.INFO, [0], 'MAP: SIM', basicmetric_columns)
        for eachbasicmetricfid in basicmetric_columns:
            self.map_unitfeatures.append(eachbasicmetricfid)
        runtime.console.print(0, runtime.console.styles.INFO, [0], 'MAP: DIFF', self.diff_columns)
        for eachdifffid in self.diff_columns:
            self.map_unitfeatures.append(eachdifffid)
        basicmetric_embed_w2group_column_names = []
        for eachmetafeature in self.gml.metafeatures.values():
            if eachmetafeature.type in fg.metafeature.types.bilateral:
                if eachmetafeature.function not in self.map_unitfeatures:
                    self.map_unitfeatures.append(eachmetafeature.function)
                    basicmetric_embed_w2group_column_names.append(eachmetafeature.function)
        basicmetric_embed_w2group = []
        for eachpair in self.gml.pairs:
            raw_map_currentpair = [metainfo.top.NONE_VALUE] * (len(self.map_unitfeatures) + 2)
            basicmetric_embed_w2group.append([metainfo.top.NONE_VALUE] * len(basicmetric_embed_w2group_column_names))
            for eachmetafeature_ineachpair in eachpair.metafeatures:
                if eachmetafeature_ineachpair.type == fg.metafeature.types.BASICMETRIC:
                    if eachmetafeature_ineachpair.fid in self.map_unitfeatures:
                        raw_map_metafeature_index = self.map_unitfeatures.index(eachmetafeature_ineachpair.fid)
                        metafeaturevalue = eachpair.metafeatures[eachmetafeature_ineachpair]
                        raw_map_currentpair[raw_map_metafeature_index] = metafeaturevalue
                elif eachmetafeature_ineachpair.type == fg.metafeature.types.DIFF:
                    if eachmetafeature_ineachpair.fid in self.map_unitfeatures:
                        raw_map_metafeature_index = self.map_unitfeatures.index(eachmetafeature_ineachpair.fid)
                        metafeaturevalue = eachpair.metafeatures[eachmetafeature_ineachpair]
                        raw_map_currentpair[raw_map_metafeature_index] = metafeaturevalue
                elif eachmetafeature_ineachpair.type in fg.metafeature.types.bilateral:
                    if eachmetafeature_ineachpair.function in self.map_unitfeatures:
                        raw_map_metafeature_index = self.map_unitfeatures.index(eachmetafeature_ineachpair.function)
                        if eachmetafeature_ineachpair.type in fg.metafeature.types.bilateral_same:
                            metafeaturevalue = 1
                        elif eachmetafeature_ineachpair.type in fg.metafeature.types.bilateral_diff:
                            metafeaturevalue = 0
                        raw_map_currentpair[raw_map_metafeature_index] = metafeaturevalue
                        basicmetric_embed_w2group[-1][basicmetric_embed_w2group_column_names.index(eachmetafeature_ineachpair.function)] = metafeaturevalue
            self.raw_map.append(raw_map_currentpair)
        self.raw_map = np.array(self.raw_map)
        if metainfo.runningflags.basicmetric_embed_w2group == True:
            basicmetric_embed_w2group = np.array(basicmetric_embed_w2group)
            basicmetric_embed_w2group_rawpairs = pd.read_csv(self.gml.datapath + self.gml.data.pairpath)
            for columnindex in range(0, len(basicmetric_embed_w2group_column_names)):
                basicmetric_embed_w2group_rawpairs[basicmetric_embed_w2group_column_names[columnindex]] = list(basicmetric_embed_w2group[:, columnindex])
            basicmetric_embed_w2group_rawpairs.to_csv(self.gml.datapath + 'svm_' + self.gml.data.pairpath, index= False)
            sys.exit(0)
        nonevalue_unitfeatures = []
        effectivevalue_unitfeatures = []
        for eachunitfeatureindex in range(0, len(self.map_unitfeatures)):
            if len(set(self.raw_map[:, eachunitfeatureindex].tolist())) == 1:
                nonevalue_unitfeatures.append(self.map_unitfeatures[eachunitfeatureindex])
            else:
                effectivevalue_unitfeatures.append(eachunitfeatureindex)
        effectivevalue_unitfeatures.append(len(self.map_unitfeatures))
        effectivevalue_unitfeatures.append(len(self.map_unitfeatures) + 1)
        self.raw_map = np.array(self.raw_map[:, effectivevalue_unitfeatures])
        self.discretize_map = np.array(self.raw_map)
        self.map_unitfeatures_nonevalueprocessed = True
        for each_nonevalue_unitfeature in nonevalue_unitfeatures:
            self.map_unitfeatures.remove(each_nonevalue_unitfeature)
        for eachunitfeatureindex in range(0, len(self.map_unitfeatures)):
            if self.map_unitfeatures[eachunitfeatureindex] in self.diff_columns:
                self.diff_metafeature_indexes_in_map_unitfeature.append(eachunitfeatureindex)

    def issamesubarea(self, index1, index2, columns, polar):
        discretize_map = None
        if polar == 0:
            discretize_map = self.discretize_map_for_skyline_nonevalueprocessed_0
        elif polar == 1:
            discretize_map = self.discretize_map_for_skyline_nonevalueprocessed_1
        elif polar == None:
            discretize_map = self.discretize_map
        if columns == None:
            return np.array_equal(discretize_map[index1, 0:len(self.map_unitfeatures)], discretize_map[index2, 0:len(self.map_unitfeatures)], equal_nan=False)
        else:
            return np.array_equal(discretize_map[index1, columns], discretize_map[index2, columns], equal_nan=False)

    def update(self, specifiedpair = None):
        specifiedpairs = None
        if specifiedpair == None:
            specifiedpairs = self.gml.pairs
        else:
            specifiedpairs = [specifiedpair]
        for eachpair in specifiedpairs:
            if eachpair.islabeled() == False:
                self.raw_map[eachpair.pid, -2] = metainfo.top.INDETERMINATE
                self.discretize_map[eachpair.pid, -2] = metainfo.top.INDETERMINATE
                if self.gml.probegml != None:
                    probe = self.gml.probegml.probes[eachpair.pid]
                    self.raw_map[eachpair.pid, -1] = probe.label
                    self.discretize_map[eachpair.pid, -1] = probe.label
                else:
                    self.raw_map[eachpair.pid, -1] = metainfo.top.INDETERMINATE
                    self.discretize_map[eachpair.pid, -1] = metainfo.top.INDETERMINATE
            else:
                self.raw_map[eachpair.pid, -2] = eachpair.label
                self.discretize_map[eachpair.pid, -2] = eachpair.label
                self.raw_map[eachpair.pid, -1] = eachpair.label
                self.discretize_map[eachpair.pid, -1] = eachpair.label
            if eachpair in self.gml.trainingpoolpairs:
                self.subareas_0[eachpair].sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)
                self.subareas_1[eachpair].sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)

    def __init__(self, gml):
        self.gml = gml
        diff_columns_indexes = self.gml.diff_columns_indexes
        if metainfo.paras.genetics_basicmetric_selector != None:
            diff_columns_indexes = list(set(self.gml.diff_columns_indexes).intersection(metainfo.paras.genetics_basicmetric_selector))
            diff_columns_indexes.sort()
        self.diff_columns = runtime.sublist(self.gml.raw_basicmetric_columns, diff_columns_indexes)
        gml.discretizefeature = self
        self.map_unitfeatures = []
        self.map_unitfeature_category = []
        self.map_unitfeatures_nonevalueprocessed = False
        self.diff_metafeature_indexes_in_map_unitfeature = []
        self.fixed_unitfeatures = []
        self.raw_map = []
        self.discretize_map = []
        self.discretize_map_for_skyline_nonevalueprocessed_0 = None
        self.discretize_map_for_skyline_nonevalueprocessed_1 = None
        self.skylinedistance_w = []
        self.features = {}
        self.subareas_0 = {}
        self.subareas_1 = {}
        self.map()
        self.discretizemetafeatures()
        self.skyline_nonevalueprocessed()
        self.subarea()
        self.update(None)

    def skyline_nonevalueprocessed(self):
        self.discretize_map_for_skyline_nonevalueprocessed_0 = np.array(self.discretize_map[:, 0:len(self.map_unitfeatures)])
        self.discretize_map_for_skyline_nonevalueprocessed_1 = np.array(self.discretize_map[:, 0:len(self.map_unitfeatures)])
        runtime.skyline.nonevalueprocess(self.discretize_map_for_skyline_nonevalueprocessed_0, self.gml.discretizefeature.diff_metafeature_indexes_in_map_unitfeature, 0, monotone_nonevalue_transformer=self.gml.monotone_nonevalue_transformer)
        runtime.skyline.nonevalueprocess(self.discretize_map_for_skyline_nonevalueprocessed_1, self.gml.discretizefeature.diff_metafeature_indexes_in_map_unitfeature, 1, monotone_nonevalue_transformer=self.gml.monotone_nonevalue_transformer)

    def discretizemetafeatures(self):
        progress = 0
        for eachindex in range(0, len(self.map_unitfeatures)):
            thisdiscretizefeature = discretizefeature.discretizemetafeature(self, eachindex)
            runtime.consoleprogress(progress, len(self.map_unitfeatures), 'discretize metafeatures')
            self.skylinedistance_w.append(float(1)/(max(thisdiscretizefeature.pairs.keys())))
            progress += 1
        self.skylinedistance_w = np.array(self.skylinedistance_w)
        runtime.consoleprogress(progress, len(self.map_unitfeatures), 'discretize metafeatures')

class genetic:

    # The performing of genetics aims to be exactly perfect, therefore the granularity of genetics are not subareas but the forms to be more precise.

    def __init__(self, gml, rule_or_inherit, mutation):
        self.gml = gml
        self.formfeature = rule_or_inherit.formfeature
        self.polar0 = None
        self.polar1 = None
        self.active(rule_or_inherit, mutation)
        self.gml.genetics[self.formfeature] = self

    def active(self, rule_or_inherit, mutation):
        certified_predicates = runtime.hashableset()
        polaractive = None
        new_genetics = None
        polar = rule_or_inherit.polar
        if rule_or_inherit.polar == 0:
            if self.polar0 is None:
                self.polar0 = {}
            polaractive = self.polar0
            new_genetics = self.gml.new_genetics_0
        else:
            if self.polar1 is None:
                self.polar1 = {}
            polaractive = self.polar1
            new_genetics = self.gml.new_genetics_1
        formfeature_values = {}
        for eachformmetafeature in self.formfeature:
            formfeature_values[eachformmetafeature] = self.gml.discretizefeature.features[eachformmetafeature].pairs
        rules_predicates_conforms = runtime.combine(form_values=formfeature_values, all_form_values_values=self.gml.pairs, polar=polar, mutation_basepredicates=rule_or_inherit.predicates, mutation=mutation)
        for each_rule_predicates in rules_predicates_conforms:
            if each_rule_predicates not in self.gml.processed_rules_predicates:
                degenerated = False
                for each_dominate_degenerated_predicates in self.gml.dominate_degenerated_rules_predicates:
                    if runtime.predicate.issmallerthan(each_dominate_degenerated_predicates, each_rule_predicates) == True:
                        degenerated = True
                        break
                if degenerated == True:
                    continue
                superiorapproved = False
                for each_dominate_approved_rules_predicates in self.gml.dominate_approved_rules_predicates:
                    if runtime.predicate.issmallerthan(each_rule_predicates, each_dominate_approved_rules_predicates) == True:
                        superiorapproved = True
                        break
                if superiorapproved == True:
                    continue
                conform_pairs = rules_predicates_conforms[each_rule_predicates]
                if len(conform_pairs) > 0:
                    conform_pairs = sorted(conform_pairs, key=lambda x: x.pid, reverse=False)
                    if len(each_rule_predicates) == len(self.formfeature):
                        polaractive[each_rule_predicates] = conform_pairs
                        if self not in new_genetics:
                            new_genetics[self] = set()
                        new_genetics[self].add(each_rule_predicates)
                        certified_predicates.add(each_rule_predicates)
                        if each_rule_predicates != rule_or_inherit.predicates:
                            assert(mutation is not None and mutation != 0)
                            formmutation_inheritrule = genetic.inherit(gml=self.gml, predicates=each_rule_predicates, polar=polar)
                            rule.inherit(inherit_type=rule.inherit.types.MUTATION, target_rule=formmutation_inheritrule, flow1=rule_or_inherit, flow2=None)
                    else:
                        # Mutation to another form.
                        formmutation_inheritrule = genetic.inherit(gml=self.gml, predicates=each_rule_predicates, polar=polar)
                        genetic.find(self.gml, rule_or_inherit=formmutation_inheritrule)
                        rule.inherit(inherit_type=rule.inherit.types.MUTATION, target_rule=formmutation_inheritrule, flow1=rule_or_inherit, flow2=None)
        return certified_predicates

    class candidategenetic:

        def __init__(self, thegenetic, polar):
            self.genetic = thegenetic
            self.polar = polar
            self.candidaterules = None
            self.effective = None
            if self.polar == 0:
                if self.genetic.polar0 != None:
                    self.effective = True
                else:
                    self.effective = False
            else:
                if self.genetic.polar1 != None:
                    self.effective = True
                else:
                    self.effective = False

        def torule(self, probe):
            rules_predicates_conforms = None
            if self.polar == 0:
                rules_predicates_conforms = self.genetic.polar0
            else:
                rules_predicates_conforms = self.genetic.polar1
            self.candidaterules = []
            for each_rule_predicates in rules_predicates_conforms:
                if each_rule_predicates not in self.genetic.gml.processed_rules_predicates:
                    conform_pairs = rules_predicates_conforms[each_rule_predicates]
                    rule(self, each_rule_predicates, conform_pairs, weight=None, gml=None, polar=None)
            for each_candidaterule in list(self.candidaterules):
                each_candidaterule.tocandidate(probe)
                if (runtime.isNone(each_candidaterule.expectation_roi) == True or each_candidaterule.expectation_roi == 0):
                    self.candidaterules.remove(each_candidaterule)

    class inherit:
        def __init__(self, gml, predicates, polar):
            self.gml = gml
            self.polar = polar
            self.predicates = runtime.hashableset(predicates)
            self.formfeature = runtime.hashableset()
            for each_predicate in self.predicates:
                self.formfeature.add(each_predicate.feature)
            self.formfeaturevalues = {}
            self.predicatedisplays = ''
            for eachpredicate in self.predicates:
                formfeatureindex = self.gml.discretizefeature.map_unitfeatures.index(eachpredicate.feature)
                assert (eachpredicate.valueex == None)
                self.formfeaturevalues[formfeatureindex] = eachpredicate.value
                currentpredicatedescribe = eachpredicate.print()
                if len(self.predicatedisplays) > 0:
                    self.predicatedisplays += ' & '
                self.predicatedisplays += currentpredicatedescribe
                # self.predicatedisplays += currentpredicatedescribe.replace(eachpredicate.feature, str(formfeatureindex))
            self.formfeatureindexes = list(self.formfeaturevalues.keys())
            self.formfeatureindexes.sort()

    @staticmethod
    def necessarycheck(gml, rule_or_inherit):
        if gml.data.necessary_attribute[0] == None and gml.data.necessary_attribute[1] == None:
            return True
        attribute_necessary = []
        if gml.data.necessary_attribute[0] != None:
            attribute_necessary = list(gml.data.necessary_attribute[0])
        raw_basicmetric_necessary = []
        if gml.data.necessary_attribute[1] != None:
            raw_basicmetric_necessary = list(gml.data.necessary_attribute[1])
        for eachpredicate in rule_or_inherit.predicates:
            thefeature_fid_or_function = eachpredicate.feature
            if len(attribute_necessary) > 0:
                fidsparse = fg.metafeature.rfid(gml, thefeature_fid_or_function)
                attributeindex = None
                if fidsparse != None:
                    type, abbr, attributename, attributeindex, function, parameter = fidsparse
                else:
                    attributeindex = metainfo.top.ALL_ATTRIBUTES_INDEX
                if attributeindex in attribute_necessary:
                    attribute_necessary.remove(attributeindex)
            if len(raw_basicmetric_necessary) > 0:
                if thefeature_fid_or_function in raw_basicmetric_necessary:
                    raw_basicmetric_necessary.remove(thefeature_fid_or_function)
            if len(attribute_necessary) == 0 and len(raw_basicmetric_necessary) == 0:
                break
        return len(attribute_necessary) == 0 and len(raw_basicmetric_necessary) == 0

    @staticmethod
    def find(gml, rule_or_inherit):
        thegenetic = None
        isnew = False
        formfeature = rule_or_inherit.formfeature
        polar = rule_or_inherit.polar
        if genetic.necessarycheck(gml, rule_or_inherit) == True:
            thegenetic = None
            if formfeature in gml.genetics:
                thegenetic = gml.genetics[formfeature]
                if polar == 0 and thegenetic.polar0 == None or polar == 1 and thegenetic.polar1 == None:
                    isnew = True
            else:
                thegenetic = genetic(gml, rule_or_inherit, mutation=None)
                isnew = True
            certified_predicates = thegenetic.active(rule_or_inherit, mutation=None)
            if type(rule_or_inherit) != genetic.inherit and rule_or_inherit.predicates in certified_predicates:
                gml.certified_rules_rawprobability[rule_or_inherit.predicates] = rule_or_inherit.stat_probability
        return isnew, thegenetic

    def formfid(self):
        theformfid = []
        for eachformfeature in self.formfeature:
            if type(eachformfeature) == fg.metafeature:
                theformfid.append(eachformfeature.fid)
            else:
                theformfid.append(str(eachformfeature))
        return theformfid

    def __eq__(self, another):
        if type(self) == type(another) and self.formfeature != None and self.formfeature == another.formfeature:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.formfeature)

class rule:

    class verifyresult:
        SUCCESS_0 = '0 √'
        FAIL_0 = '0 ×'
        SUCCESS_1 = '1 √'
        FAIL_1 = '1 ×'
        NO_CHANCE = 'NO CHANCE'

    class inherit:

        class types:
            MUTATION = 'MUTATION'
            RECOMBINATION = 'RECOMBINATION'
            INIT = 'INIT'
            REBOOT = 'REBOOT'
            FOREROUND = 'FOREROUND'

        def __init__(self, inherit_type, target_rule, flow1, flow2=None):
            self.type = inherit_type
            self.target_rule = target_rule
            self.predicates = target_rule.predicates
            self.rule = self.target_rule.predicatedisplays
            self.flow = None
            self.flowindex = None
            if self.type == rule.inherit.types.MUTATION:
                self.flow = flow1.predicatedisplays
                self.flowindex = rule.inherit.index(flow1, self.target_rule.gml.rule_inherit)
            elif self.type == rule.inherit.types.RECOMBINATION:
                self.flow = (flow1.predicatedisplays, flow2.predicatedisplays)
                self.flowindex = (rule.inherit.index(flow1, self.target_rule.gml.rule_inherit), rule.inherit.index(flow2, self.target_rule.gml.rule_inherit))
            elif self.type == rule.inherit.types.INIT:
                self.flow = runtime.round(flow1)
            elif self.type == rule.inherit.types.REBOOT:
                self.flow = flow1.predicatedisplays
                self.flowindex = rule.inherit.index(flow1, self.target_rule.gml.rule_inherit)
            elif self.type == rule.inherit.types.FOREROUND:
                self.flow = None
                self.flowindex = None
            self.target_rule.gml.rule_inherit.append(self)

        @staticmethod
        def update(update_rule, rule_inherit):
            updateindex = rule.inherit.index(update_rule, rule_inherit)
            rule_inherit[updateindex].target_rule = update_rule
            rule_inherit[updateindex].predicates = update_rule.predicates
            rule_inherit[updateindex].rule = update_rule.predicatedisplays
            return

        @staticmethod
        def index(flow, rule_inherit):
            for eachflowindex in range(0, len(rule_inherit)):
                if flow.predicates == rule_inherit[eachflowindex].predicates:
                    return eachflowindex
            return None

        @staticmethod
        def settle(settlecsvpath, rule_inherit):
            csvfile = open(settlecsvpath, 'a', newline='')
            settledict = {}
            for eachflowindex in range(0, len(rule_inherit)):
                currentinherit = rule_inherit[eachflowindex]
                approve = ''
                if type(currentinherit.target_rule) == rule:
                    if currentinherit.target_rule.approve == True:
                        approve = '√'
                    elif currentinherit.target_rule.approve == False:
                        approve = '×'
                    elif currentinherit.target_rule.approve == None:
                        approve = ''
                settledict['id'] = eachflowindex
                settledict['approve'] = approve
                settledict['type'] = currentinherit.type
                settledict['rule'] = currentinherit.rule
                settledict['flow'] = currentinherit.flow
                settledict['flowindex'] = currentinherit.flowindex
                csvwriter = csv.DictWriter(csvfile, list(settledict.keys()))
                if eachflowindex == 0:
                    csvwriter.writeheader()
                csvwriter.writerow(settledict)
            csvfile.close()
            return

    def __init__(self, candidategenetic, predicates, conform_pairs, weight=None, probability=None, gml=None, polar=None):
        self.predicates = runtime.hashableset(predicates)
        self.expectation_roi = None
        self.fitness = None
        self.probeprobability = None
        self.correcting = 0
        self.truth_correcting = [0, 0, 0]
        self.truth_correcting_correct = []
        self.truth_correcting_misjudge = []
        self.actual_correcting = [0, [0, 0], [0, 0]]
        self.raw_certified = metainfo.top.NOT_AVAILABLE
        self.conform_pairs = conform_pairs
        assert(len(self.conform_pairs) > 0)
        self.resolution = len(self.conform_pairs)
        self.conform_pair_indexes = [each.pid for each in self.conform_pairs]
        self.conform_pair_indexes.sort(reverse=False)
        self.indtrans_conform_pair_indexes = runtime.indexes_transformer(tuple([self.conform_pair_indexes,]))
        self.test_conform_pairs = []
        self.test_truth_correctproportion = 0
        self.complete_knowable_trainingpool = None
        self.confidence = None
        self.probability = None
        self.weight = weight
        self.probability = probability
        self.nonevalueprocess_map = None
        self.trainingpool = None
        self.skyline = None
        self.skyline_nonevalueprocess_map_unique = None
        self.skyline_trainingpool_nonevalueprocess_map_unique = None
        self.skyline_trainingpool = None
        self.skyline_verify_0 = None
        self.skyline_verify_1 = None
        self.skyline_support = None
        self.skyline_oppose = None
        self.skyline_verify_indexes = None
        self.skyline_verify_selector = None
        self.skylinedistance_w = None
        self.skyline_distances = None
        self.require_new_verification_allowance = None
        self.exceed = None
        self.subareas = None
        self.subareas_values = None
        self.subareas_polareffectives = None
        self.nondirectional_pair_indexes = None
        self.approve = None
        self.rulemetafeature = None
        self.probepolar_oppositerule = None
        self.evo_rules_offensive_evolution = None
        self.evo_voilate_rules_offensive_evolution = None
        self.evo_subarea_genetics = None
        self.evo_subarea_violate_genetics = None
        self.gml = None
        self.polar = None
        self.candidategenetic = candidategenetic
        if self.candidategenetic != None:
            self.gml = self.candidategenetic.genetic.gml
            self.polar = self.candidategenetic.polar
            if self.predicates in self.gml.certified_rules_rawprobability:
                self.raw_certified = runtime.round(self.gml.certified_rules_rawprobability[self.predicates])
        else:
            self.gml = gml
            self.polar = polar
        self.formfeaturevalues = {}
        self.predicatedisplays = ''
        for eachpredicate in self.predicates:
            formfeatureindex = self.gml.discretizefeature.map_unitfeatures.index(eachpredicate.feature)
            assert(eachpredicate.valueex == None)
            self.formfeaturevalues[formfeatureindex] = eachpredicate.value
            currentpredicatedescribe = eachpredicate.print()
            if len(self.predicatedisplays) > 0:
                self.predicatedisplays += ' & '
            self.predicatedisplays += currentpredicatedescribe
            # self.predicatedisplays += currentpredicatedescribe.replace(eachpredicate.feature, str(formfeatureindex))
        self.formfeatureindexes = list(self.formfeaturevalues.keys())
        self.formfeatureindexes.sort()
        processed_rule = collections.namedtuple('processed_rule', ['polar', 'conforms'])
        self.processed_rule = processed_rule(polar=self.polar, conforms=set(self.conform_pair_indexes))
        for each_conform_pair in self.conform_pairs:
            if each_conform_pair.pairtype == fg.pair.pairtypes.TESTSET:
                self.test_conform_pairs.append(each_conform_pair)
                if each_conform_pair.truthlabel == self.polar:
                    self.test_truth_correctproportion += 1
                truth_correcting = each_conform_pair.probe_correcting(self.polar, truth=True, newcoverageaware=None)
                self.truth_correcting[0] += truth_correcting
                if truth_correcting == 1:
                    self.truth_correcting[1] += 1
                    self.truth_correcting_correct.append(each_conform_pair)
                elif truth_correcting == -1:
                    self.truth_correcting[2] -= 1
                    self.truth_correcting_misjudge.append(each_conform_pair)
        if len(self.test_conform_pairs) > 0:
            self.test_truth_correctproportion /= len(self.test_conform_pairs)
            if self.weight == None:
                self.candidategenetic.candidaterules.append(self)
        else:
            self.gml.del_certified(self.predicates)

    def tocandidate(self, probe):
        assert(self in self.candidategenetic.candidaterules)
        self.toskyline()
        if len(self.skyline_trainingpool) > 0 and (metainfo.paras.rule_knowable == False or self.complete_knowable_trainingpool == True or probe == True):
            self.subarea_selector()
            if self.exceed == False or probe == True:
                self.analysis()
        else:
            self.gml.del_certified(self.predicates)
        if probe == False:
            self.toprocessed()

    def toprocessed(self):
        if self.approve != None or \
           self.skyline_trainingpool != None and len(self.skyline_trainingpool) == 0 or \
           metainfo.paras.skyline_verify_steplimit_adjust == None and self.exceed == True or \
           metainfo.paras.rule_knowable == True and self.complete_knowable_trainingpool == False:
            self.gml.processed_rules_predicates[self.predicates] = self.processed_rule
            if self.candidategenetic != None:
                rules_predicates_conforms = None
                if self.polar == 0:
                    rules_predicates_conforms = self.candidategenetic.genetic.polar0
                else:
                    rules_predicates_conforms = self.candidategenetic.genetic.polar1
                if self.predicates in rules_predicates_conforms:
                    del rules_predicates_conforms[self.predicates]
                    self.gml.del_certified(self.predicates)
            self.gml.rule_verify[rule.verifyresult.NO_CHANCE] += 1
        # have tocandidate()
        if (self.approve is not None) and self.require_new_verification_allowance != None and self.require_new_verification_allowance > 0:
            rule_verify_result = rule.verifyresult.NO_CHANCE
            if self.approve == True:
                if self.polar == 0:
                    rule_verify_result = rule.verifyresult.SUCCESS_0
                else:
                    rule_verify_result = rule.verifyresult.SUCCESS_1
            else:
                if self.polar == 0:
                    rule_verify_result = rule.verifyresult.FAIL_0
                else:
                    rule_verify_result = rule.verifyresult.FAIL_1
            self.gml.rule_verify[rule_verify_result] += 1
        rule.inherit.update(self, self.gml.rule_inherit)

    def print(self, specifiedpair = None):
        info = {}
        info['polar'] = self.polar
        info[metainfo.top.GROUND_TRUTH] = str(int(self.test_truth_correctproportion * 100)) + ' %, ' + str(self.truth_correcting)
        info['actual'] = str(self.actual_correcting)
        info['raw certified'] = self.raw_certified
        info['# T'] = len(self.test_conform_pairs)
        if self.approve == None:
            info['fitness'] = runtime.round(self.fitness)
            info['# Verify'] = len(self.skyline_verify_selector)
            info['# SArea'] = len(self.subareas_values)
        else:
            if metainfo.method.Rule_LearnableWeight == True:
                if specifiedpair != None:
                    info['sppair weight'] = runtime.round(specifiedpair.metaweight(self.rulemetafeature))
            else:
                info['weight'] = runtime.round(self.weight)
            if self.candidategenetic != None:
                info['fitness'] = runtime.round(self.fitness)
                info['VB probability'] = runtime.round(self.probability)
                info['VS probability'] = runtime.round(float(len(self.skyline_support)) / len(self.skyline_verify_selector))
                info['# Verify'] = len(self.skyline_verify_selector)
                info['# SArea'] = len(self.subareas_values)
                info['# support'] = len(self.skyline_support)
                info['# oppose'] = len(self.skyline_oppose)
                if self.probepolar_oppositerule != None:
                    info['# probeoppose'] = len(self.probepolar_oppositerule)
                else:
                    info['# probeoppose'] = 'superior or depressed'
            else:
                info['foreround'] = True
        info['predicate'] = self.predicatedisplays
        return info

    def analysis(self):
        # Basically, generating rules aims to correcting. The violate opposition just accelerates the genetic evolution.
        # Expectation of Correctly Correction: Utility * Fitness.
        # Utility: # of prospective correction pairs.
        # Fitness: Probability of being effective, independent of Utility, High Fitness Low Utility still leads to low expectation_roi.
        # Risk: Confess Skyline is NOT always being effective.
        if len(self.skyline_trainingpool) > 0:
            for eachsubareaindex in range(0, len(self.subareas_values)):
                self.subareas_values[eachsubareaindex] = confidence.subarea(polar=self.polar, func_probability=self.gml.GlobalBalance_probability, subarea_pairs=self.subareas_values[eachsubareaindex], label_pairs=self.skyline_verify_selector)
            #self.confidence = confidence.confidence_coefficient(func_probability=self.gml.GlobalBalance_probability, subareas=self.subareas_values)
            self.confidence = 1
            self.fitness = [thepair.withprobe_get_probability()[0] for thepair in self.skyline_verify_selector]
            self.fitness = np.array(self.fitness, dtype=np.float64)
            self.fitness = np.fabs(1 - self.polar - self.fitness)
            self.fitness = np.average(self.fitness)
            if self.predicates in self.gml.certified_rules_criterion:
                self.gml.certified_rules_resolution[self.predicates] = self.resolution
                self.gml.certified_rules_criterion[self.predicates] = self.fitness
            for each_conform_pair in self.conform_pairs:
                currentcorrecting = each_conform_pair.probe_correcting(self.candidategenetic.polar, truth=False, newcoverageaware=metainfo.method.RuleCandidate_NewCoverage)
                self.correcting += currentcorrecting
            correcting = self.correcting
            # rule approve already has weight polar judgement.
            self.expectation_roi = math.pow(self.fitness, metainfo.paras.rule_roi_fitness_coefficient) * math.pow(correcting, metainfo.paras.rule_roi_correcting_coefficient)

    def toskyline(self):
        sense = None
        if self.polar == 0:
            self.nonevalueprocess_map = self.gml.discretizefeature.discretize_map_for_skyline_nonevalueprocessed_0
            sense = 'max'
        else:
            self.nonevalueprocess_map = self.gml.discretizefeature.discretize_map_for_skyline_nonevalueprocessed_1
            sense = 'min'
        skylinecache = collections.namedtuple('skylinecache', ['skyline', 'skyline_nonevalueprocess_map_unique', 'skyline_trainingpool', 'trainingpool', 'complete_knowable_trainingpool', 'skyline_distances'])
        if self.predicates in self.gml.cache.skylinecache:
            theskylinecache = self.gml.cache.skylinecache[self.predicates]
            self.skyline = theskylinecache.skyline
            self.skyline_nonevalueprocess_map_unique = theskylinecache.skyline_nonevalueprocess_map_unique
            self.skyline_trainingpool = theskylinecache.skyline_trainingpool
            self.trainingpool = theskylinecache.trainingpool
            self.complete_knowable_trainingpool = theskylinecache.complete_knowable_trainingpool
            self.skyline_distances = theskylinecache.skyline_distances
        else:
            # Skyline of raw_map must be skyline of discretize_map, not vice versa.
            # Priority to Skyline of both raw_map and discretize_map indicated priority of raw_map skyline in the same subareas.
            local_conform_map = self.nonevalueprocess_map[self.conform_pair_indexes, :]
            map_unitfeatures = None
            if metainfo.method.Rule_LocalSkyline == True:
                local_conform_map = local_conform_map[:, self.formfeatureindexes]
                map_unitfeatures = np.array(self.gml.discretizefeature.map_unitfeatures)[self.formfeatureindexes].tolist()
            else:
                map_unitfeatures = self.gml.discretizefeature.map_unitfeatures
            df = pd.DataFrame(local_conform_map, columns=map_unitfeatures)
            mask = paretoset(df, sense=[sense] * local_conform_map.shape[1], distinct=False, use_numba=True)
            self.trainingpool = set()
            for eachconformpair in self.conform_pairs:
                if eachconformpair.pairtype == fg.pair.pairtypes.TRAININGPOOL:
                    self.trainingpool.add(eachconformpair)
            self.skyline = set()
            self.skyline_trainingpool = set()
            self.skyline_nonevalueprocess_map_unique = []
            self.skyline_trainingpool_nonevalueprocess_map_unique = []
            for eachmaskindex in range(0, len(mask)):
                if mask[eachmaskindex] == 1:
                    pairindex = self.conform_pair_indexes[eachmaskindex]
                    thepair = self.gml.pairs[pairindex]
                    themap = local_conform_map[eachmaskindex, :].tolist()
                    self.skyline.add(thepair)
                    self.skyline_nonevalueprocess_map_unique.append(themap)
                    if thepair.pairtype == fg.pair.pairtypes.TRAININGPOOL:
                        self.skyline_trainingpool.add(thepair)
                        self.skyline_trainingpool_nonevalueprocess_map_unique.append(themap)
            self.skyline_nonevalueprocess_map_unique = np.array(self.skyline_nonevalueprocess_map_unique)
            self.skyline_nonevalueprocess_map_unique = np.unique(self.skyline_nonevalueprocess_map_unique, axis=0)
            self.skyline_trainingpool_nonevalueprocess_map_unique = np.array(self.skyline_trainingpool_nonevalueprocess_map_unique)
            self.skyline_trainingpool_nonevalueprocess_map_unique = np.unique(self.skyline_trainingpool_nonevalueprocess_map_unique, axis=0)
            # count for unique subareas.
            self.complete_knowable_trainingpool = (self.skyline_nonevalueprocess_map_unique.shape[0] == self.skyline_trainingpool_nonevalueprocess_map_unique.shape[0])
            self.skyline_distances = {}
            theskylinecache = skylinecache(skyline=self.skyline, skyline_nonevalueprocess_map_unique=self.skyline_nonevalueprocess_map_unique, skyline_trainingpool=self.skyline_trainingpool, trainingpool=self.trainingpool, complete_knowable_trainingpool=self.complete_knowable_trainingpool, skyline_distances=self.skyline_distances)
            self.gml.cache.skylinecache[self.predicates] = theskylinecache
        if metainfo.method.Rule_LocalSkyline == True:
            self.skylinedistance_w = self.gml.discretizefeature.skylinedistance_w[self.formfeatureindexes]
        else:
            self.skylinedistance_w = self.gml.discretizefeature.skylinedistance_w

    def subarea_selector(self):
        self.skyline_verify_selector = set()
        for eachpair in self.trainingpool:
            if eachpair.ishumanlabeled() == True:
                self.skyline_verify_selector.add(eachpair)
        self.require_new_verification_allowance = 0
        self.subareas = {}
        self.subareas_values = []
        skyline_columns = None
        if metainfo.method.Rule_SkylineSubArea == False:
            assert(metainfo.method.Rule_LocalSkyline == True)
            assert(metainfo.paras.tree_maxdepth <= 2)
            samesubarea = list(self.skyline_trainingpool)
            samesubarea.sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)
            for eachpair in self.skyline_trainingpool:
                self.subareas[eachpair] = samesubarea
            self.subareas_values.append(samesubarea)
        else:
            # Better to uniformly verification on skyline subareas.
            # If short predicates <= 2, mostly 2 (local) subareas not affect a lot.
            # For predicates >= 3, difference (local) subareas no intersection must be all concerned, or it may become a verification for another form rule.
            if metainfo.method.Rule_LocalSkyline == True:
                skyline_columns = self.formfeatureindexes
                for eachpair_1 in self.skyline_trainingpool:
                    if eachpair_1 not in self.subareas:
                        samesubarea = []
                        for eachpair_2 in self.skyline_trainingpool:
                            if eachpair_2 not in self.subareas:
                                if self.gml.discretizefeature.issamesubarea(eachpair_1.pid, eachpair_2.pid, columns=skyline_columns, polar=self.polar) == True:
                                    samesubarea.append(eachpair_2)
                        samesubarea.sort(key=lambda x: runtime.entropy(x.withprobe_get_probability()[0]), reverse=True)
                        self.subareas[eachpair_1] = samesubarea
                        for eachsamesubareapair in samesubarea:
                            self.subareas[eachsamesubareapair] = samesubarea
                        self.subareas_values.append(samesubarea)
            else:
                skyline_columns = None
                for eachpair in self.skyline_trainingpool:
                    if eachpair not in self.subareas:
                        subarea_pairs = None
                        if self.polar == 0:
                            subarea_pairs = self.gml.discretizefeature.subareas_0[eachpair]
                        else:
                            subarea_pairs = self.gml.discretizefeature.subareas_1[eachpair]
                        subarea_pairs_raw_skyline = []
                        for eachpair_in_same_subarea_pairs in subarea_pairs:
                            # If use raw_map skyline, not all the same subarea pairs are in self.skyline_trainingpool.
                            if eachpair_in_same_subarea_pairs in self.skyline_trainingpool:
                                subarea_pairs_raw_skyline.append(eachpair_in_same_subarea_pairs)
                        for eachpair_in_same_subarea_pairs in subarea_pairs_raw_skyline:
                            self.subareas[eachpair_in_same_subarea_pairs] = subarea_pairs_raw_skyline
                        self.subareas_values.append(subarea_pairs_raw_skyline)
        allowance_limit = min(metainfo.paras.skyline_verify_steplimit, self.gml.humancostallowance_thisround)
        subareas_totalpairs = []
        for subarea_pairs_raw_skyline in self.subareas_values:
            subareas_totalpairs.append(len(subarea_pairs_raw_skyline))
        subareas_totalpairs = np.array(subareas_totalpairs)
        assert(sum(subareas_totalpairs)) == len(self.skyline_trainingpool)
        skyline_verify_selector = set(self.skyline_verify_selector)
        self.skyline_verify_selector = set()
        def put_PerSkylineSubarea_single(PerSkylineSubarea_single, skyline_verify_selector):
            subareas_alreadyinselector = []
            for subarea_pairs_raw_skyline in self.subareas_values:
                copy_samesubarea = set(subarea_pairs_raw_skyline[0:min(len(subarea_pairs_raw_skyline), PerSkylineSubarea_single)])
                alreadyinselector = copy_samesubarea.intersection(skyline_verify_selector)
                subareas_alreadyinselector.append(len(alreadyinselector))
            subareas_alreadyinselector = np.array(subareas_alreadyinselector)
            return subareas_alreadyinselector
        VerificationLimit_PerSkylineSubarea_single = 0
        VerificationLimit_PerSkylineSubarea = None
        subareas_require_new_verification = None
        total_selectors = None
        subareas_alreadyinselector = None
        if metainfo.method.Rule_SkylineSort == False:
            subareas_alreadyinselector = put_PerSkylineSubarea_single(math.inf, skyline_verify_selector)
        while True:
            VerificationLimit_PerSkylineSubarea_single += 1
            if metainfo.method.Rule_SkylineSort == True:
                subareas_alreadyinselector = put_PerSkylineSubarea_single(VerificationLimit_PerSkylineSubarea_single, skyline_verify_selector)
            VerificationLimit_PerSkylineSubarea = np.array([VerificationLimit_PerSkylineSubarea_single] * len(self.subareas_values))
            subarea_totalpairs_limitindexes = np.where(VerificationLimit_PerSkylineSubarea > subareas_totalpairs)
            VerificationLimit_PerSkylineSubarea[subarea_totalpairs_limitindexes] = subareas_totalpairs[subarea_totalpairs_limitindexes]
            subareas_require_new_verification = VerificationLimit_PerSkylineSubarea - subareas_alreadyinselector
            subarea_existingverificationexceedrequire_limitindexes = np.where(subareas_require_new_verification < 0)
            subareas_require_new_verification[subarea_existingverificationexceedrequire_limitindexes] = 0
            total_selectors = subareas_alreadyinselector + subareas_require_new_verification
            assert(np.any(total_selectors == 0) == False)
            if sum(total_selectors) == sum(subareas_totalpairs) or sum(subareas_require_new_verification) >= allowance_limit or sum(total_selectors) >= metainfo.paras.skyline_verify_steplimit:
                exempt = []
                while sum(subareas_require_new_verification) > allowance_limit or sum(total_selectors) > metainfo.paras.skyline_verify_steplimit:
                    most_selectors = np.argsort(0 - total_selectors)
                    most_selectors_index = 0
                    while most_selectors[most_selectors_index] in exempt:
                        most_selectors_index += 1
                        if most_selectors_index >= len(most_selectors):
                            break
                    if most_selectors_index < len(most_selectors):
                        current_subarea_index = most_selectors[most_selectors_index]
                        if total_selectors[current_subarea_index] >= 2:
                            if subareas_require_new_verification[current_subarea_index] >= 1:
                                subareas_require_new_verification[current_subarea_index] -= 1
                                total_selectors[current_subarea_index] -= 1
                            else:
                                exempt.append(current_subarea_index)
                        else:
                            break
                    else:
                        break
                if metainfo.top.OBSOLETE:
                    while sum(total_selectors) > metainfo.paras.skyline_verify_steplimit:
                        most_selectors = np.argsort(0 - total_selectors)
                        most_selectors_index = 0
                        current_subarea_index = most_selectors[most_selectors_index]
                        if total_selectors[current_subarea_index] >= 2:
                            assert(subareas_require_new_verification[current_subarea_index] == 0)
                            subareas_alreadyinselector[current_subarea_index] -= 1
                            total_selectors[current_subarea_index] -= 1
                        else:
                            break
                break
        v_total_selectors = subareas_alreadyinselector + subareas_require_new_verification
        assert(np.all(total_selectors - v_total_selectors == 0))
        assert(np.any(total_selectors == 0) == False)
        # Each subareas_value at least one verification to support complete skyline.
        if sum(subareas_require_new_verification) > allowance_limit:
            self.exceed = True
            return
        else:
            self.exceed = False
            for subarea_pairs_raw_skyline_index in range(0, len(self.subareas_values)):
                subarea_pairs = self.subareas_values[subarea_pairs_raw_skyline_index]
                inselector = total_selectors[subarea_pairs_raw_skyline_index]
                newinselector = inselector - subareas_alreadyinselector[subarea_pairs_raw_skyline_index]
                if newinselector < 0:
                    newinselector = 0
                assert (subareas_require_new_verification[subarea_pairs_raw_skyline_index] == newinselector)
                if metainfo.method.Rule_SkylineSort == True:
                    for eachsamesubareapair in subarea_pairs:
                        if inselector > 0:
                            self.skyline_verify_selector.add(eachsamesubareapair)
                            inselector -= 1
                            if eachsamesubareapair.ishumanlabeled() == False:
                                newinselector -= 1
                        else:
                            break
                    assert(newinselector == 0)
                else:
                    self.skyline_verify_selector = self.skyline_verify_selector.union(skyline_verify_selector.intersection(subarea_pairs))
                    for eachsamesubareapair in subarea_pairs:
                        if newinselector > 0:
                            if eachsamesubareapair not in self.skyline_verify_selector:
                                assert(eachsamesubareapair.ishumanlabeled() == False)
                                self.skyline_verify_selector.add(eachsamesubareapair)
                                newinselector -= 1
                        else:
                            break
            for eachpair in self.skyline_verify_selector:
                if eachpair.ishumanlabeled() == False:
                    self.require_new_verification_allowance += 1
            assert(self.require_new_verification_allowance == sum(subareas_require_new_verification))
        # Simple Distance for Addition. Also 0 distance skyline first. Addition requires when mean distributed subarea guaranteed and there is still allowance. Can be small not exceed.
        if metainfo.method.Rule_NotOnlySkyline == True and len(self.skyline_verify_selector) < metainfo.paras.skyline_verify_steplimit:
            trainingpool_distances = {}
            allowing_pair_range = self.trainingpool
            trainingpool_pair_status = collections.namedtuple('trainingpool_pair_status', ['is_skyline', 'skyline_distance', 'is_human_labeled', 'withprobe_predicted_entropy'])
            for each_trainingpool_pair in allowing_pair_range:
                if each_trainingpool_pair not in self.skyline_verify_selector:
                    trainingpool_distances[each_trainingpool_pair] = \
                        trainingpool_pair_status(is_skyline=int(each_trainingpool_pair in self.skyline_trainingpool), \
                                                 skyline_distance=self.skyline_distance(each_trainingpool_pair, self.skyline_nonevalueprocess_map_unique, self.skylinedistance_w), \
                                                 is_human_labeled=int(each_trainingpool_pair.ishumanlabeled()),
                                                 withprobe_predicted_entropy=runtime.entropy(each_trainingpool_pair.withprobe_get_probability()[0]))
            trainingpool_distances_items = list(trainingpool_distances.items())
            cmp_distance_fun = None
            if self.polar == 0:
                cmp_distance_fun = rule.cmp_distance_0
            else:
                cmp_distance_fun = rule.cmp_distance_1
            trainingpool_distances_items.sort(key=functools.cmp_to_key(cmp_distance_fun), reverse=False)
            inskyline_index = 0
            while len(self.skyline_verify_selector) < metainfo.paras.skyline_verify_steplimit and inskyline_index < len(trainingpool_distances) and self.require_new_verification_allowance < allowance_limit:
                self.skyline_verify_selector.add(trainingpool_distances_items[inskyline_index][0])
                if trainingpool_distances_items[inskyline_index][0].ishumanlabeled() == False:
                    self.require_new_verification_allowance += 1
                inskyline_index += 1
        if metainfo.method.Rule_NotOnlySkyline == False and len(self.skyline_verify_selector) < metainfo.paras.skyline_verify_steplimit_skylineleast \
            or metainfo.method.Rule_NotOnlySkyline == True and len(self.skyline_verify_selector) < metainfo.paras.skyline_verify_steplimit:
            self.exceed = True
            return
        if len(self.skyline_verify_selector) < metainfo.paras.skyline_verify_steplimit and \
            (metainfo.method.Rule_NotOnlySkyline == False and len(self.skyline_verify_selector) < len(self.skyline_trainingpool) or metainfo.method.Rule_NotOnlySkyline == True and len(self.skyline_verify_selector) < len(self.trainingpool)):
            self.exceed = True

    @staticmethod
    def cmp_distance_0(a, b):
        if a[1].is_skyline == b[1].is_skyline:
            if a[1].skyline_distance == b[1].skyline_distance:
                if a[1].is_human_labeled == b[1].is_human_labeled:
                    return runtime.cmp_reverse(a[1].withprobe_predicted_entropy, b[1].withprobe_predicted_entropy)
                else:
                    return runtime.cmp_reverse(a[1].is_human_labeled, b[1].is_human_labeled)
            else:
                return runtime.cmp(a[1].skyline_distance, b[1].skyline_distance)
        else:
            return runtime.cmp_reverse(a[1].is_skyline, b[1].is_skyline)

    @staticmethod
    def cmp_distance_1(a, b):
        if a[1].is_skyline == b[1].is_skyline:
            if a[1].skyline_distance == b[1].skyline_distance:
                if a[1].is_human_labeled == b[1].is_human_labeled:
                    return runtime.cmp_reverse(a[1].withprobe_predicted_entropy, b[1].withprobe_predicted_entropy)
                else:
                    return runtime.cmp_reverse(a[1].is_human_labeled, b[1].is_human_labeled)
            else:
                return runtime.cmp(a[1].skyline_distance, b[1].skyline_distance)
        else:
            return runtime.cmp_reverse(a[1].is_skyline, b[1].is_skyline)

    def verify(self):
        self.skyline_verify_0 = []
        self.skyline_verify_1 = []
        self.skyline_verify_indexes = []
        # If using raw_map, each_skyline_trainingpool_pair in self.skyline_verify_selector may NOT in self.skyline_trainingpool.
        for each_skyline_trainingpool_pair in self.skyline_verify_selector:
            each_skyline_trainingpool_pair.tolabel(fg.pair.labeltypes.HUMAN)
            if each_skyline_trainingpool_pair.label == 0:
                self.skyline_verify_0.append(each_skyline_trainingpool_pair)
            else:
                self.skyline_verify_1.append(each_skyline_trainingpool_pair)
            self.skyline_verify_indexes.append(each_skyline_trainingpool_pair.pid)
        self.skyline_support = None
        self.skyline_oppose = None
        if self.polar == 0:
            self.skyline_support = self.skyline_verify_0
            self.skyline_oppose = self.skyline_verify_1
        else:
            self.skyline_support = self.skyline_verify_1
            self.skyline_oppose = self.skyline_verify_0
        self.skyline_verify_indexes.sort(reverse=False)
        self.nondirectional_pair_indexes = list(set(range(0, self.gml.discretizefeature.discretize_map.shape[0])) - set(self.skyline_verify_indexes))
        self.nondirectional_pair_indexes.sort(reverse=False)
        self.probability = self.gml.GlobalBalance_probability(balance=False, label1count=len(self.skyline_verify_1), label0count=len(self.skyline_verify_0))
        if self.predicates in self.gml.certified_rules_criterion:
            self.gml.certified_rules_criterion[self.predicates] = math.fabs(1 - self.polar - self.probability)
        self.weight = confidence.weight(self.confidence, self.probability, polar=self.polar)
        polareffective = None
        # inner sub-skyline involved as a whole area, also human-verified strict unbalanced probability for rule approve as >> subarea.
        polareffective_weight = confidence.weight(confidence_coefficient=1, balanceprobability=self.probability, polar=self.polar)
        rule_approveprobability_onskyline = None
        if type(metainfo.paras.rule_approveprobability_onskyline) == list:
            rule_approveprobability_onskyline = metainfo.paras.rule_approveprobability_onskyline[self.polar]
        else:
            rule_approveprobability_onskyline = metainfo.paras.rule_approveprobability_onskyline
        polareffective = confidence.effective(weight=polareffective_weight, effectiveprobability=rule_approveprobability_onskyline, effectiveweight=None)
        if polareffective == True:
            # Probe in same round can be updated when conforming, leading to correction cleared to zero to be enlarged from local to global.
            self.toapprove()
        else:
            self.approve = False
            self.evo_subarea_genetics = []
            self.evo_rules_offensive_evolution = self.evo_subarea_genetic(violate=False)
            if metainfo.runningflags.Show_Detail == True:
                runtime.console.print(1, runtime.console.styles.PERIOD, [2], 'SGML > verify > genetic evolution (offensive)', '# evo_rules_offensive_evolution = ' + str(len(self.evo_rules_offensive_evolution)))
            for each_evo_rule in self.evo_rules_offensive_evolution:
                isnew, this_evo_subarea_genetic = genetic.find(gml=self.gml, rule_or_inherit=each_evo_rule)
                rule.inherit(inherit_type=rule.inherit.types.REBOOT, target_rule=each_evo_rule, flow1=self, flow2=None)
                if this_evo_subarea_genetic != None:
                    if this_evo_subarea_genetic == self.candidategenetic and each_evo_rule.polar == self.candidategenetic.polar:
                        continue
                    self.evo_subarea_genetics.append(this_evo_subarea_genetic)
            self.evo_subarea_violate_genetics = []
            self.evo_voilate_rules_offensive_evolution = self.evo_subarea_genetic(violate=True)
            if metainfo.runningflags.Show_Detail == True:
                runtime.console.print(1, runtime.console.styles.PERIOD, [2], 'SGML > verify > violate genetic evolution (offensive)', '# evo_voilate_rules_offensive_evolution = ' + str(len(self.evo_voilate_rules_offensive_evolution)))
            for each_evo_voilate_rule in self.evo_voilate_rules_offensive_evolution:
                isnew, this_evo_subarea_voilate_genetic = genetic.find(gml=self.gml, rule_or_inherit=each_evo_voilate_rule)
                rule.inherit(inherit_type=rule.inherit.types.REBOOT, target_rule=each_evo_voilate_rule, flow1=self, flow2=None)
                if this_evo_subarea_voilate_genetic != None:
                    self.evo_subarea_violate_genetics.append(this_evo_subarea_voilate_genetic)
        self.toevolution()
        self.toprocessed()

    def toevolution(self):
        assert(self.approve != None and (self.approve == False or self.approve == True))
        if self.approve == False:
            for eachgenetic in self.gml.genetics.values():
                if len(self.candidategenetic.genetic.formfeature.intersection(eachgenetic.formfeature)) == len(eachgenetic.formfeature):
                    rules_predicates_conforms = None
                    if self.polar == 0:
                        rules_predicates_conforms = eachgenetic.polar0
                    else:
                        rules_predicates_conforms = eachgenetic.polar1
                    if rules_predicates_conforms != None:
                        delpredicates = []
                        for eachpredicate in rules_predicates_conforms:
                            if runtime.predicate.issmallerthan(self.predicates, eachpredicate) == True:
                                delpredicates.append(eachpredicate)
                        for eachdelpredicate in delpredicates:
                            del rules_predicates_conforms[eachdelpredicate]
                            self.gml.del_certified(eachdelpredicate)
            self.gml.dominate_degenerated_rules_predicates.add(self.predicates)
        else:
            if self.candidategenetic != None:
                mutation = [0, 0]
                mutation[1 - self.polar] = metainfo.paras.genetics_mutation
                self.candidategenetic.genetic.active(self, mutation=mutation)
                for eachgenetic in self.gml.genetics.values():
                    if len(self.candidategenetic.genetic.formfeature.intersection(eachgenetic.formfeature)) == len(self.candidategenetic.genetic.formfeature):
                        rules_predicates_conforms = None
                        if self.polar == 0:
                            rules_predicates_conforms = eachgenetic.polar0
                        else:
                            rules_predicates_conforms = eachgenetic.polar1
                        if rules_predicates_conforms != None:
                            delpredicates = []
                            for eachpredicate in rules_predicates_conforms:
                                if runtime.predicate.issmallerthan(eachpredicate, self.predicates) == True:
                                    delpredicates.append(eachpredicate)
                            for eachdelpredicate in delpredicates:
                                del rules_predicates_conforms[eachdelpredicate]
                                self.gml.del_certified(eachdelpredicate)
            thedelrules = []
            for eachapprovedrule in self.gml.approved_rules:
                if eachapprovedrule.predicates != self.predicates and runtime.predicate.issmallerthan(eachapprovedrule.predicates, self.predicates) == True:
                    thedelrules.append(eachapprovedrule)
                    if eachapprovedrule.rulemetafeature != None:
                        eachapprovedrule.rulemetafeature.obsolete()
                    for eachconformpair in eachapprovedrule.conform_pairs:
                        if eachapprovedrule in eachconformpair.rules:
                            eachconformpair.rules.remove(eachapprovedrule)
            for eachdelrule in thedelrules:
                self.gml.approved_rules.remove(eachdelrule)
                if eachdelrule.predicates in self.gml.dominate_approved_rules_predicates:
                    self.gml.dominate_approved_rules_predicates.remove(eachdelrule.predicates)
                self.gml.del_certified(eachdelrule.predicates)
            self.gml.dominate_approved_rules_predicates.add(self.predicates)

    def evo_subarea_genetic(self, violate):
        labelindex = -1
        forestpolar = None
        if violate == False:
            forestpolar = self.polar
        else:
            forestpolar = 1 - self.polar
        main_map_updating = np.array(self.gml.discretizefeature.discretize_map)
        main_map_updating = main_map_updating[self.skyline_verify_indexes, :]
        nondirectional_map = self.gml.discretizefeature.discretize_map[self.nondirectional_pair_indexes, :]
        # Local Subarea, Can NOT be Global Balanced.
        genetics_forest = runtime.forest(balance=False, mainmap_knowledgeupdating=main_map_updating, labelindex=labelindex,
                                         splitters=self.gml.discretizefeature.map_unitfeatures, polar=forestpolar,
                                         weight=confidence.weight, probability=self.gml.GlobalBalance_probability,
                                         confidence_coefficient=confidence.confidence_coefficient,
                                         premilinary_condition_predicates=None,
                                         nondirectional_map=nondirectional_map, conform_map=None,
                                         roottrees=None)
        genetics_forest_rules = set()
        for each_parallel_index in range(0, len(genetics_forest.rules)):
            genetics_forest_rules.update(set(genetics_forest.rules[each_parallel_index]))
        for each_rule in genetics_forest_rules:
            each_rule.predicatedisplays = ''
            for eachpredicate in each_rule.predicates:
                assert (eachpredicate.valueex == None)
                currentpredicatedescribe = eachpredicate.print()
                if len(each_rule.predicatedisplays) > 0:
                    each_rule.predicatedisplays += ' & '
                each_rule.predicatedisplays += currentpredicatedescribe
            each_rule.gml = self.gml
        return genetics_forest_rules

    def skyline_distance(self, thepair, skyline_map, skylinedistance_w):
        if thepair in self.skyline_distances:
            return self.skyline_distances[thepair]
        else:
            distance = None
            if thepair in self.skyline:
                distance = 0
            else:
                thepair_map = None
                if metainfo.method.Rule_LocalSkyline == True:
                    thepair_map = self.nonevalueprocess_map[thepair.pid, self.formfeatureindexes]
                else:
                    thepair_map = self.nonevalueprocess_map[thepair.pid, 0:len(self.gml.discretizefeature.map_unitfeatures)]
                distance = runtime.skyline.distance(thepair_map, skyline_map, smallerprefer=(self.polar == 1), w=skylinedistance_w)
            self.skyline_distances[thepair] = distance
            return distance

    def factor(self):
        skyline_distances_maxdistance = max(self.skyline_distances.values())
        for each_conform_pair in self.conform_pairs:
            thefeaturevalue = None
            if self.polar == 0:
                thefeaturevalue = skyline_distances_maxdistance - self.skyline_distances[each_conform_pair]
            else:
                thefeaturevalue = self.skyline_distances[each_conform_pair]
            self.rulemetafeature.pairmetafeaturevalue(each_conform_pair, thefeaturevalue)
        self.rulemetafeature.tonormalize()

    def toapprove(self):
        self.approve = True
        superiorapproved = False
        for each_dominate_approved_rules_predicates in self.gml.dominate_approved_rules_predicates:
            if runtime.predicate.issmallerthan(self.predicates, each_dominate_approved_rules_predicates) == True:
                superiorapproved = True
                break
        if superiorapproved == True:
            return
        self.probepolar_oppositerule = set()
        self.gml.approved_rules.append(self)
        if self.candidategenetic != None:
            if self.candidategenetic.genetic not in self.gml.certified_rules:
                self.gml.certified_rules[self.candidategenetic.genetic] = set()
            self.gml.certified_rules[self.candidategenetic.genetic].add(self.predicates)
            if self.polar == 0:
                if self.candidategenetic.genetic not in self.gml.certified_rules_0:
                    self.gml.certified_rules_0[self.candidategenetic.genetic] = set()
                self.gml.certified_rules_0[self.candidategenetic.genetic].add(self.predicates)
            else:
                if self.candidategenetic.genetic not in self.gml.certified_rules_1:
                    self.gml.certified_rules_1[self.candidategenetic.genetic] = set()
                self.gml.certified_rules_1[self.candidategenetic.genetic].add(self.predicates)
            self.gml.certified_rules_criterion[self.predicates] = math.fabs(1 - self.polar - self.probability)
            self.gml.certified_rules_resolution[self.predicates] = self.resolution
        if metainfo.method.Rule_LearnableWeight == True:
            ruletype = None
            if self.polar == 0:
                ruletype = fg.metafeature.types.RULE_0
            else:
                ruletype = fg.metafeature.types.RULE_1
            self.rulemetafeature = fg.metafeature.find(self.gml, fg.metafeature.fid(self.gml, ruletype, metainfo.top.ALL_ATTRIBUTES, self.predicatedisplays))
            self.rulemetafeature.rule = self
        if metainfo.method.Rule_LearnableWeight == True:
            for each_conform_pair in self.conform_pairs:
                self.skyline_distance(each_conform_pair, self.skyline_nonevalueprocess_map_unique, self.skylinedistance_w)
            self.factor()
        for eachconformpair in self.conform_pairs:
            eachconformpair.rules.append(self)
            if eachconformpair.islabeled() == False:
                if eachconformpair.probe_correcting(self.polar, truth=False, newcoverageaware=None) == 1:
                    self.probepolar_oppositerule.add(eachconformpair)
                # When not Foreround where records already with Foreround Rules.
                if self.candidategenetic != None and metainfo.method.Rule_LearnableWeight == False:
                    eachconformpair.withprobe_updaterule(self.weight)
                    self.gml.discretizefeature.update(eachconformpair)

    def reportresult(self):
        require_new_verification_allowance_info = None
        if self.require_new_verification_allowance > 0:
            require_new_verification_allowance_info = 'verify # ' + str(self.require_new_verification_allowance)
        else:
            require_new_verification_allowance_info = 'free verify'
        if self.approve == True:
            if self.truth_correcting[0] >= 0:
                runtime.console(require_new_verification_allowance_info + ' Perfect Approving < rule', self.print(), runtime.console.styles.TOP)
            else:
                runtime.console(require_new_verification_allowance_info + ' False Approving < rule', self.print(), runtime.console.styles.EXCEPTION)
        elif self.approve == False and metainfo.runningflags.Show_Detail == True:
            if self.truth_correcting[0] >= 0:
                runtime.console(require_new_verification_allowance_info + ' False Reject < rule', self.print(), runtime.console.styles.SIMPLE_EXCEPTION)
            else:
                runtime.console(require_new_verification_allowance_info + ' Correct Reject < rule', self.print(), runtime.console.styles.REPORT)

    def __hash__(self):
        return hash(self.predicates)

class confidence:

    class subarea:

        def __init__(self, polar, func_probability, subarea_pairs = None, label_pairs = None):
            self.polar = polar
            self.func_probability = func_probability
            self.confidence = None
            self.totalcount = None
            self.labeledpairs = None
            self.labeledcount = None
            self.confidence = None
            # subarea probability is human-verified strict unbalanced probability for rule approve.
            self.probability = None
            self.weight = None
            self.effective = None
            if subarea_pairs != None and label_pairs != None:
                self.totalcount = len(subarea_pairs)
                self.labeledpairs = set(label_pairs).intersection(subarea_pairs)
                self.labeledcount = len(self.labeledpairs)
                self.confidence = confidence.confidence_coefficient(func_probability=func_probability, subareas=self)

        def stat(self):
            label0count = 0
            label1count = 0
            for eachpair in self.labeledpairs:
                if eachpair.label == 0:
                    label0count += 1
                else:
                    label1count += 1
            self.probability = self.func_probability(balance=False, label1count=label1count, label0count=label0count)
            self.weight = confidence.weight(confidence_coefficient=self.confidence, balanceprobability=self.probability, polar=self.polar)
            polareffective_weight = confidence.weight(confidence_coefficient=1, balanceprobability=self.probability, polar=self.polar)
            rule_approveprobability_onskyline = None
            if type(metainfo.paras.rule_approveprobability_onskyline) == list:
                rule_approveprobability_onskyline = metainfo.paras.rule_approveprobability_onskyline[self.polar]
            else:
                rule_approveprobability_onskyline = metainfo.paras.rule_approveprobability_onskyline
            self.effective = confidence.effective(weight=polareffective_weight, effectiveprobability=rule_approveprobability_onskyline, effectiveweight=None)

        @staticmethod
        def iseffective(subareas):
            for eachsubarea in subareas:
                eachsubarea.stat()
            effective = True
            for eachsubarea in subareas:
                if eachsubarea.effective != True:
                    effective = False
                    break
            return effective

        def print(self):
            return str(self.confidence) + ' 🎯 L ' + str(self.labeledcount) + ' / T ' + str(self.totalcount)

    @staticmethod
    def confidence_coefficient(func_probability, subareas):
        enhance_multiplier = metainfo.paras.balance_rule_weight_multiplier
        if subareas == max or metainfo.paras.tree_consider_trainingcountconfidence == True:
            return enhance_multiplier
        else:
            if type(subareas) == confidence.subarea:
                moststrict_priorp = 0.5
                subareas.confidence = runtime.confidentialsample(moststrict_priorp, metainfo.top.SMALL_PROBABILITY, subareas.labeledcount, None)
                theconfidence = subareas.confidence * enhance_multiplier
                return theconfidence
            else:
                totalcount = 0
                labaledcount = 0
                polar = None
                for eachsubarea in subareas:
                    totalcount += eachsubarea.totalcount
                    labaledcount += eachsubarea.labeledcount
                    if polar == None:
                        polar = eachsubarea.polar
                rulearea = confidence.subarea(polar=polar, func_probability=func_probability, subarea_pairs=None, label_pairs=None)
                rulearea.totalcount = totalcount
                rulearea.labeledcount = labaledcount
                return confidence.confidence_coefficient(func_probability, rulearea)

    @staticmethod
    def weight(confidence_coefficient, balanceprobability, polar):
        if runtime.probabilitypolar(balanceprobability) == polar:
            abs_weight = math.fabs(logit(balanceprobability))
            if abs_weight > metainfo.paras.regressiontaubound:
                abs_weight = metainfo.paras.regressiontaubound
            if abs_weight > metainfo.paras.weightbound:
                abs_weight = metainfo.paras.weightbound
            weight = abs_weight * confidence_coefficient
            if polar == 0:
                weight = (-1) * weight
            else:
                weight = weight
            return weight
        else:
            return None

    weight_bound = None
    effectivecriterion_bound = None
    probability_smallbound = None

    @staticmethod
    def init():
        confidence.weight_bound = confidence.weight(confidence.confidence_coefficient(None, max), 1.0, 1)
        confidence.effectivecriterion_bound = confidence.weight(1, 1.0, 1)
        confidence.probability_smallbound = metainfo.top.SMALL_VALUE
        if confidence.weight_bound < 0:
            confidence.weight_bound = (-1) * confidence.weight_bound
        if confidence.effectivecriterion_bound < 0:
            confidence.effectivecriterion_bound = (-1) * confidence.effectivecriterion_bound
        if confidence.probability_smallbound > 0.5:
            confidence.probability_smallbound = 1 - confidence.probability_smallbound

    @staticmethod
    def effective(weight, effectiveprobability, effectiveweight):
        if weight == None:
            return False
        if effectiveweight == None:
            effectiveweight = logit(effectiveprobability)
        effectiveweightcriterion = math.fabs(effectiveweight)
        if effectiveweightcriterion > confidence.effectivecriterion_bound:
            effectiveweightcriterion = confidence.effectivecriterion_bound
        weightcriterion = math.fabs(weight)
        if effectiveweightcriterion > 0:
            return weightcriterion >= effectiveweightcriterion
        else:
            return weightcriterion > effectiveweightcriterion

