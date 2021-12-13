from enum import Flag, auto, Enum
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from scipy.special import logit, expit
# python-texttable
from source.texttable import Texttable, get_color_string, bcolors
from collections import Counter
# string-color
# from stringcolor import cs

from source import metainfo
from source.runtime import runtime
from source.rule import confidence

class pair:

    class probe:

        def __init__(self):
            self.label = None
            self.probability = None
            self.weight = None

    class labeltypes():
        EASY = 'EASY'
        APPROXIMATE = 'APPROXIMATE'
        INFERENCE = 'INFERENCE'
        HUMAN = 'HUMAN'

    class pairtypes():
        TRAININGPOOL = 'TRAININGPOOL'
        VALIDATIONPOOL = 'VALIDATIONPOOL'
        TESTSET = 'TESTSET'

    class inferenceresult():
        SGML_RULE_CORRECT_0 = 'SR √ 0'
        SGML_RULE_CORRECT_1 = 'SR √ 1'
        SGML_RULE_MISJUDGE_0 = 'SR × 0'
        SGML_RULE_MISJUDGE_1 = 'SR × 1'
        SGML_CORRECT_0 = 'S~ √ 0'
        SGML_CORRECT_1 = 'S~ √ 1'
        SGML_MISJUDGE_0 = 'S~ × 0'
        SGML_MISJUDGE_1 = 'S~ × 1'
        USGML_BOTH_RIGHT_0 = 'US √ 0'
        USGML_BOTH_RIGHT_1 = 'US √ 1'
        USGML_BOTH_WRONG_0 = 'US × 0'
        USGML_BOTH_WRONG_1 = 'US × 1'

        GMLONLY_RIGHT = 'GML √'
        GMLONLY_WRONG = 'GML ×'
        BOTH_RIGHT = 'Both √'
        BOTH_WRONG = 'Both ×'
        RULE_CORRECT = 'RULE √'
        RULE_MISJUDGE = 'RULE ×'
        RULE_LEAN_CORRECT = 'RULE → √'
        RULE_LEAN_MISJUDGE = 'RULE → ×'

        NOT_AVAILABLE = 'N/A'

    def __init__(self, gml, pid):
        self.pairitem = gml.rawpairs.iloc[pid].tolist()
        self.gml = gml
        self.labeltype = None
        self.pairtype = None
        self.pid = pid
        self.metafeatures = {}
        self.similarity = None
        self.label = None
        self.truthlabel = int(self.pairitem[1])
        assert(self.truthlabel == 0 or self.truthlabel == 1)
        self.weight = None
        self.probability = None
        self.entropy = None

        self.metafeature_evidentialsupport = {}
        self.evidentialsupport = None
        self.approximateweight = None
        self.approximateprobability = None
        self.approximateentropy = None

        self.probe = pair.probe()
        self.ugmllabel = None

        self.rules = []
        self.ruleresult = None
        self.sgmlresult = None

        self.gml.pairs.append(self)
        if self.truthlabel == 1:
            self.gml.truth_1_pairs.add(self)
        assert(len(self.gml.pairs) == pid + 1)

        idlist = self.pairitem[0].split(",", 1)
        e1id = idlist[0]
        e2id = idlist[1]
        self.data1 = e1id
        self.data2 = e2id
        self.infere1token = self.gml.records[e1id][-2]
        self.infere2token = self.gml.records[e2id][-2]
        self.infercooccurtoken = None
        self.inferpartialoccurtoken = None
        self.infere1tokengroup = None
        self.infere2tokengroup = None
        self.infercooccurtokengroup = set()
        self.inferpartialoccurtokengroup = set()

        if self.gml.w2groups != None and metainfo.method.Token_SubCooccur == True:
            infere1token = set(self.infere1token)
            infere2token = set(self.infere2token)
            infercooccurtoken = infere1token.intersection(infere2token)
            infere1token = infere1token - infercooccurtoken
            infere2token = infere2token - infercooccurtoken
            for eachtoken in set(infere1token):
                for eachinfercooccurtoken in infercooccurtoken:
                    assert (eachtoken != infercooccurtoken)
                    if eachtoken in eachinfercooccurtoken or eachinfercooccurtoken in eachtoken:
                        infere1token.remove(eachtoken)
                        break
            for eachtoken in set(infere2token):
                for eachinfercooccurtoken in infercooccurtoken:
                    assert (eachtoken != infercooccurtoken)
                    if eachtoken in eachinfercooccurtoken or eachinfercooccurtoken in eachtoken:
                        infere2token.remove(eachtoken)
                        break
            uniontoken = infere1token.union(infere2token)
            inferpartialoccurtoken = uniontoken - infercooccurtoken
            weakcooccurspliter = {}
            for eachinferpartialoccurtoken_1 in inferpartialoccurtoken:
                for eachinferpartialoccurtoken_2 in inferpartialoccurtoken:
                    if eachinferpartialoccurtoken_1 != eachinferpartialoccurtoken_2:
                        if eachinferpartialoccurtoken_1 in eachinferpartialoccurtoken_2:
                            if eachinferpartialoccurtoken_2 in weakcooccurspliter:
                                weakcooccurspliter[eachinferpartialoccurtoken_2].add(eachinferpartialoccurtoken_1)
                            else:
                                weakcooccurspliter[eachinferpartialoccurtoken_2] = set([eachinferpartialoccurtoken_1])
            for eachsplitted in weakcooccurspliter:
                thesplitter = weakcooccurspliter[eachsplitted]
                nomoresplitter = False
                while nomoresplitter == False:
                    nomoresplitter = True
                    for eachsplitter in set(thesplitter):
                        if eachsplitter in weakcooccurspliter:
                            thesplitter.remove(eachsplitter)
                            thesplitter.update(weakcooccurspliter[eachsplitter])
                            nomoresplitter = False
            for eachtoken in set(infere1token):
                if eachtoken in weakcooccurspliter:
                    infere1token.remove(eachtoken)
                    infere1token.update(weakcooccurspliter[eachtoken])
            for eachtoken in set(infere2token):
                if eachtoken in weakcooccurspliter:
                    infere2token.remove(eachtoken)
                    infere2token.update(weakcooccurspliter[eachtoken])
            self.infere1token = infere1token.union(infercooccurtoken)
            self.infere2token = infere2token.union(infercooccurtoken)

        self.infercooccurtoken = self.infere1token.intersection(self.infere2token)
        uniontoken = self.infere1token.union(self.infere2token)
        self.inferpartialoccurtoken = uniontoken - self.infercooccurtoken

        if self.gml.w2groups != None:
            for eachtoken in self.inferpartialoccurtoken:
                if eachtoken in self.gml.w2groups:
                    thegroup = self.gml.w2groups[eachtoken]
                    self.inferpartialoccurtokengroup.add(thegroup)
            for eachtoken in self.infercooccurtoken:
                if eachtoken in self.gml.w2groups:
                    thegroup = self.gml.w2groups[eachtoken]
                    if thegroup not in self.inferpartialoccurtokengroup or thegroup in runtime.regularpattern.grouppatterns_cooccur_veto:
                        self.infercooccurtokengroup.add(thegroup)
            if len(runtime.regularpattern.grouppatterns_cooccur_veto) > 0:
                for eachgroup in runtime.regularpattern.grouppatterns_cooccur_veto:
                    if eachgroup in self.infercooccurtokengroup and eachgroup in self.inferpartialoccurtokengroup:
                        self.inferpartialoccurtokengroup.remove(eachgroup)
                    if eachgroup in runtime.regularpattern.grouppatterns_cooccur_veto_ground and eachgroup in self.infercooccurtokengroup:
                        self.infercooccurtokengroup.remove(eachgroup)
                        self.infercooccurtokengroup.add(runtime.regularpattern.grouppatterns_cooccur_veto_ground[eachgroup])
            if runtime.isNone(metainfo.paras.nlpw2vgroups) == False and 'sparse' in metainfo.paras.nlpw2vgroups:
                self.inferpartialoccurtoken.clear()
                self.infercooccurtoken.clear()

        # load basicmetric metafeatures
        for eachbasicmetricindex in self.gml.basicmetric_columns_indexes:
            themetafeature = metafeature.find(self.gml, self.gml.raw_basicmetric_columns[eachbasicmetricindex])
            value = self.pairitem[eachbasicmetricindex]
            themetafeature.pairmetafeaturevalue(self, value)

        # load diff metafeatures
        for eachdiffindex in self.gml.diff_columns_indexes:
            themetafeature = metafeature.find(self.gml, self.gml.raw_basicmetric_columns[eachdiffindex])
            value = runtime.round(self.pairitem[eachdiffindex])
            themetafeature.pairmetafeaturevalue(self, value)

        # load token metafeatures
        for eachcooccurtoken in self.infercooccurtoken:
            themetafeature = metafeature.find(self.gml, metafeature.fid(self.gml, metafeature.types.TOKEN_COOCCUR, metainfo.top.ALL_ATTRIBUTES, eachcooccurtoken))
            themetafeature.pairmetafeaturevalue(self, metainfo.top.WAITING)
        for eachpartialoccurtoken in self.inferpartialoccurtoken:
            themetafeature = metafeature.find(self.gml, metafeature.fid(self.gml, metafeature.types.TOKEN_PARTIALOCCUR, metainfo.top.ALL_ATTRIBUTES, eachpartialoccurtoken))
            themetafeature.pairmetafeaturevalue(self, metainfo.top.WAITING)

        # load tokengroup metafeatures
        for eachcooccurtokengroup in self.infercooccurtokengroup:
            themetafeature = metafeature.find(self.gml, metafeature.fid(self.gml, metafeature.types.TOKENGROUP_COOCCUR, metainfo.top.ALL_ATTRIBUTES, eachcooccurtokengroup))
            themetafeature.pairmetafeaturevalue(self, metainfo.top.WAITING)
        for eachpartialoccurtokengroup in self.inferpartialoccurtokengroup:
            themetafeature = metafeature.find(self.gml, metafeature.fid(self.gml, metafeature.types.TOKENGROUP_PARTIALOCCUR, metainfo.top.ALL_ATTRIBUTES, eachpartialoccurtokengroup))
            themetafeature.pairmetafeaturevalue(self, metainfo.top.WAITING)

    def metaweight(self, themetafeature):
        return themetafeature.metaweight(self, self.labeltype) * themetafeature.coeffective[self.pid]

    def ruleweight(self, detailed):
        ruleweight = {}
        if metainfo.method.Rule_LearnableWeight == True:
            for eachrule in self.rules:
                ruleweight[eachrule] = self.metaweight(eachrule.rulemetafeature)
        else:
            for eachrule in self.rules:
                ruleweight[eachrule] = eachrule.weight
        if detailed == False:
            ruleweight = sum(ruleweight.values())
        return ruleweight

    def islabeled(self):
        if self.labeltype == None and self.label == None and self in self.gml.unlabeledpairs:
            return False
        elif self.labeltype != None and (self.label == 0 or self.label == 1) and self not in self.gml.unlabeledpairs:
            return True
        else:
            info = {'labeltype': self.labeltype, 'label': self.label, 'in unlabeledpairs': self in self.gml.unlabeledpairs}
            runtime.console('ERROR > Label Status', info, runtime.console.styles.EXCEPTION)
            sys.breakpointhook()

    def ishumanlabeled(self):
        if self.islabeled() == False or self.labeltype != pair.labeltypes.HUMAN:
            return False
        else:
            return True

    def withprobe_get_probability(self):
        probability = None
        weight = None
        if self.islabeled() == True:
            if self.labeltype == pair.labeltypes.INFERENCE:
                probability = self.probability
                weight = self.weight
            elif self.labeltype == pair.labeltypes.APPROXIMATE:
                probability = self.approximateprobability
                weight = self.approximateweight
            else:
                probability = self.label
                if self.label == 1:
                    weight = metainfo.paras.regressiontaubound
                else:
                    weight = (-1) * metainfo.paras.regressiontaubound
            assert(probability != None)
        elif self.gml.probegml != None:
            probability = self.probe.probability
            weight = self.probe.weight
            assert(self.probe.probability == self.gml.probegml.probes[self.pid].probability)
            assert(probability != None)
        return probability, weight

    def probe_getlabel(self):
        if self.islabeled() == True:
            return self.label
        else:
            return self.probe.label

    def probe_correcting(self, polar, truth, newcoverageaware):
        correcting = None
        if truth == True:
            if self.islabeled() == False:
                if self.probe.label == 1 - self.truthlabel:
                    if polar == self.truthlabel:
                        correcting = 1
                    else:
                        correcting = 0
                else:
                    if polar == self.truthlabel:
                        correcting = 0
                    else:
                        correcting = -1
            else:
                correcting = 0
        else:
            if self.islabeled() == False and self.probe.label == (1 - polar):
                correcting = 1
            else:
                correcting = 0
            if newcoverageaware == True:
                existingpolarrulecount = 0
                for eachrule in self.rules:
                    # Ignore OppoPolar Adversarial, assuming more rules always lead to reward and 0 rule coverage must have priority.
                    if eachrule.polar == polar:
                        existingpolarrulecount += 1
                correcting /= math.pow(2, existingpolarrulecount)
        return correcting

    def withprobe_updaterule(self, ruleweight):
        self.probe.weight = self.probe.weight + ruleweight
        self.probe.probability = expit(self.probe.weight)
        self.probe.label = runtime.probabilitypolar(self.probe.probability)
        probe = self.gml.probegml.probes[self.pid]
        probe.weight = self.probe.weight
        probe.probability = self.probe.probability
        probe.label = self.probe.label
        self.gml.GlobalBalance_probability(balance=True, label1count=None, label0count=None)

    def print(self):
        if type(self.data1) != list or type(self.data2) != list:
            e1id = self.data1
            e2id = self.data2
            if metainfo.top.OBSOLETE:
                self.data1 = self.gml.data1.loc[self.gml.data1[self.gml.data.idname] == e1id].values.tolist()[0]
                data2 = None
                if runtime.isNone(self.gml.data2) == False:
                    data2 = self.gml.data2
                else:
                    data2 = self.gml.data1
                self.data2 = data2.loc[data2[self.gml.data.idname] == e2id].values.tolist()[0]
            self.data1 = self.gml.records[e1id][0:len(self.gml.RecordAttributes)]
            self.data2 = self.gml.records[e2id][0:len(self.gml.RecordAttributes)]
        table = Texttable(max_width=300)
        table.set_deco(Texttable.BORDER | Texttable.VLINES | Texttable.HLINES)
        table.set_cols_align(["l"] * (len(self.gml.data1.columns)))
        table.set_cols_valign(["m"] * (len(self.gml.data1.columns)))
        data1 = [get_color_string(bcolors.BLUE, str(self.truthlabel))]
        data2 = ['']
        for eachindex in range(1, len(self.data1)):
            color = None
            if self.data1[eachindex] == self.data2[eachindex]:
                color = bcolors.GREEN
            else:
                color = bcolors.RED
            data1.append(get_color_string(color, self.data1[eachindex]))
            data2.append(get_color_string(color, self.data2[eachindex]))
        table.add_row(data1)
        table.add_row(data2)
        print(table.draw())
        self.rules.sort(key=lambda x:x.polar)
        if len(self.rules) > 0:
            runtime.console('🧭 rules # ' + str(len(self.rules)), None, runtime.console.styles.INFO)
            for eachruleindex in range(0, len(self.rules)):
                eachrule = self.rules[eachruleindex]
                info = eachrule.print(self)
                if eachrule.polar != self.truthlabel:
                    runtime.console(None, info, runtime.console.styles.EXCEPTION)
                else:
                    runtime.console(None, info, runtime.console.styles.SIMPLE_CORRECTION)
        print()
        if metainfo.runningflags.Save_Mislabeledinfo == True:
            mislabeledinfo = {}
            mislabeledinfo['L'] = str(self.truthlabel)
            for eachindex in range(1, len(self.data1)):
                mislabeledinfo[self.gml.RecordAttributes[eachindex] + '_a'] = str(self.data1[eachindex])
                mislabeledinfo[self.gml.RecordAttributes[eachindex] + '_b'] = str(self.data2[eachindex])
            rulesinfo = []
            for eachruleindex in range(0, len(self.rules)):
                eachrule = self.rules[eachruleindex]
                rulesinfo.append(str(eachrule.polar) + ' : ' + eachrule.predicatedisplays)
            mislabeledinfo['R'] = str(rulesinfo)
            self.gml.mislabeledinfo.append(mislabeledinfo)

    def tolabel(self, labeltype, labelpara = None):
        if self.islabeled() == True:
            if self.labeltype == self.labeltypes.HUMAN:
                return
            for eachmetafeature in self.metafeatures:
                eachmetafeature.remove(self)
            if self.pairtype == pair.pairtypes.TESTSET:
                self.gml.test_unlabeledpairs.add(self)
            self.gml.unlabeledpairs.add(self)
            self.label = None
            self.labeltype = None
        self.labeltype = labeltype
        if labeltype == pair.labeltypes.INFERENCE:
            self.label = runtime.probabilitypolar(self.probability)
        elif labeltype == pair.labeltypes.APPROXIMATE:
            self.label = runtime.probabilitypolar(self.approximateprobability)
        elif labeltype == pair.labeltypes.EASY:
            self.label = labelpara
        elif labeltype == pair.labeltypes.HUMAN:
            assert(self.gml.humancostallowance >= 1)
            self.gml.humancostallowance -= 1
            if self.gml.humancostallowance_thisround != None:
                assert (self.gml.humancostallowance_thisround >= 1)
                self.gml.humancostallowance_thisround -= 1
            assert(self not in self.gml.test_unlabeledpairs)
            self.label = self.truthlabel
            self.gml.humanlabeled_indexes.append(self.pid)
        self.gml.unlabeledpairs.remove(self)
        if self in self.gml.test_unlabeledpairs:
            assert(self.pairtype == self.pairtypes.TESTSET)
            self.gml.test_unlabeledpairs.remove(self)

        for eachmetafeature in self.metafeatures:
            eachmetafeature.append(self)

        self.gml.scalableinference_updatecache_absdirtycount += 1

        finalprobability, finalweight = self.withprobe_get_probability()

        if self.gml.probegml != None:
            self.probe.label = self.label
            probe = self.gml.probegml.probes[self.pid]
            probe.label = self.label
            self.probe.probability = finalprobability
            self.probe.weight = finalweight
            probe.probability = finalprobability
            probe.weight = finalweight

        if self.gml.discretizefeature != None:
            self.gml.discretizefeature.update(self)

        ruleweight = None
        inferenceresult = metainfo.top.NOT_AVAILABLE
        if self.labeltype == pair.labeltypes.APPROXIMATE or self.labeltype == pair.labeltypes.INFERENCE:
            self.ruleresult, ruleweight = runtime.weightresultcorrect(finalweight, self, tolabeljudge=True)
            self.gml.supervised_results[self.ruleresult] += 1
            self.sgmlresult = runtime.sgmlresultcorrect(finalweight, self)
            self.gml.supervised_results[self.sgmlresult] += 1
            inferenceresult = self.sgmlresult + ', ' + self.ruleresult

        if metainfo.runningflags.Show_Detail == True and self.labeltype == pair.labeltypes.HUMAN:
            info = {'sim':runtime.round(self.similarity), 'truthlabel':self.truthlabel, 'humancostallowance':self.gml.humancostallowance_thisround}
            style = runtime.console.styles.OUTLOOK
            runtime.console('GML > Label > ' + self.labeltype, info, style)
        elif self.label == 1 or self.truthlabel == 1 or self.ugmllabel == 1:
            if self.pairtype == pair.pairtypes.TESTSET:
                if self.label == 1:
                    self.gml.current_test_label_1_pairs.add(self)
                if self.truthlabel == 1:
                    self.gml.current_test_truth_1_pairs.add(self)
                    if self.label == 1:
                        self.gml.current_test_truelabel_1_pairs.add(self)
            if self.labeltype == pair.labeltypes.APPROXIMATE or self.labeltype == pair.labeltypes.INFERENCE:
                info = {'sim': runtime.round(self.similarity), 'truthlabel': self.truthlabel, 'inferenceresult': inferenceresult, 'finalweight': runtime.round(finalweight), 'ruleweight': runtime.round(ruleweight)}
                f1 = self.gml.f1()
                style = None
                if self.pairtype == pair.pairtypes.TESTSET:
                    if self.label == self.truthlabel:
                        style = runtime.console.styles.CORRECTION
                    else:
                        style = runtime.console.styles.EXCEPTION
                else:
                    if self.label == self.truthlabel:
                        style = runtime.console.styles.SIMPLE_CORRECTION
                    else:
                        style = runtime.console.styles.SIMPLE_EXCEPTION
                if metainfo.runningflags.Show_Detail == True or style == runtime.console.styles.EXCEPTION:
                    runtime.console('GML > Label > ' + self.labeltype, info, style)
                    runtime.console(None, f1, runtime.console.styles.INFO)
                    if style == runtime.console.styles.EXCEPTION:
                        self.print()
        return

    def __eq__(self, another):
        if type(self) == type(another) and self.pid != None and self.pid == another.pid:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.pid)

class metafeature:

    evidenceinterval = runtime.uniforminterval(metainfo.paras.evidenceintervalcount)

    class types():
        BASICMETRIC = 'sim'
        DIFF = 'diff'
        TOKEN_COOCCUR = 'token_cooccur'
        TOKEN_PARTIALOCCUR = 'token_partialoccur'
        TOKENGROUP_COOCCUR = 'tokengroup_cooccur'
        TOKENGROUP_PARTIALOCCUR = 'tokengroup_partialoccur'
        RULE_0 = 'rule_0'
        RULE_1 = 'rule_1'
        abbr = 'abbr'
        parser = '.'
        para = '&'
        for_inference_polarnon = [BASICMETRIC]
        for_inference_polar0 = [TOKEN_PARTIALOCCUR, TOKENGROUP_PARTIALOCCUR, RULE_0]
        for_inference_polar1 = [TOKEN_COOCCUR, TOKENGROUP_COOCCUR, RULE_1]
        for_inference = for_inference_polarnon + for_inference_polar0 + for_inference_polar1
        bilateral_same = [TOKENGROUP_COOCCUR]
        bilateral_diff = [TOKENGROUP_PARTIALOCCUR]
        bilateral = bilateral_same + bilateral_diff
        rule = [RULE_0, RULE_1]
        normalize_type = [RULE_0, RULE_1]
        normalize_function = 'ToNormalize'
        OBSOLETE = 'obsolete'

    class updateopes(Flag):
        PROBE = auto()
        LOG_ONLY = auto()
        FULL_INFLUENCE = auto()

    def update(self, updateop):
        if self.for_inference == True:
            if updateop == metafeature.updateopes.FULL_INFLUENCE:
                assert(self.regression != None)
                self.regression.performregression()
            if self.frozen_monotony == False:
                if len(self.label0pairxs) > 0:
                    self.alphabound[0] = np.mean(list(self.label0pairxs.values()))
                else:
                    self.alphabound[0] = -math.inf
                if len(self.label1pairxs) > 0:
                    self.alphabound[1] = np.mean(list(self.label1pairxs.values()))
                else:
                    self.alphabound[1] = math.inf
                if (self.alphabound[1] > self.alphabound[0]) and (updateop == metafeature.updateopes.PROBE or updateop == metafeature.updateopes.LOG_ONLY or (updateop == metafeature.updateopes.FULL_INFLUENCE and self.regression.monotonyeffective == True)):
                    self.monotonyeffective = True
                else:
                    self.monotonyeffective = False
            else:
                self.alphabound = [-math.inf, math.inf]
            self.singleside_harmonicparabound()
            if updateop == metafeature.updateopes.PROBE:
                monotonyeffective = self.monotonyeffective
                if self.frozen_monotony == False:
                    self.alphabound = [-math.inf, math.inf]
                    self.monotonyeffective = None
                self.label0pairxs = {}
                self.label1pairxs = {}
                return monotonyeffective
            else:
                return self.monotonyeffective

    def append(self, newlabeledpair):
        if self.for_inference == True:
            assert(self.gml == newlabeledpair.gml)
            assert(newlabeledpair.islabeled() == True)
            newx = newlabeledpair.metafeatures[self]
            if newlabeledpair.label == 0:
                self.label0pairxs[newlabeledpair] = newx
                self.labelpairXY[newlabeledpair] = [newx, metainfo.paras.regressiontau * (-1)]
                if self.regression != None:
                    self.regression.append(newx, metainfo.paras.regressiontau * (-1), hardlabel=newlabeledpair.ishumanlabeled())
            else:
                self.label1pairxs[newlabeledpair] = newx
                self.labelpairXY[newlabeledpair] = [newx, metainfo.paras.regressiontau]
                if self.regression != None:
                    self.regression.append(newx, metainfo.paras.regressiontau, hardlabel=newlabeledpair.ishumanlabeled())
            for eachevidenceintervalindex in range(0, len(self.evidenceinterval)):
                if newx >= metafeature.evidenceinterval[eachevidenceintervalindex][0] and newx < metafeature.evidenceinterval[eachevidenceintervalindex][1]:
                    self.evidenceinterval[eachevidenceintervalindex].add(newlabeledpair)
                    if newlabeledpair.ishumanlabeled() == True:
                        self.human_evidenceinterval[eachevidenceintervalindex].add(newlabeledpair)
            updateop = None
            if self.regression != None:
                updateop = metafeature.updateopes.FULL_INFLUENCE
            else:
                updateop = metafeature.updateopes.LOG_ONLY
            self.update(updateop = updateop)
            return

    def remove(self, removelabeledpair):
        if self.for_inference == True:
            XY = self.labelpairXY[removelabeledpair]
            assert (XY[0] == removelabeledpair.metafeatures[self])
            del self.labelpairXY[removelabeledpair]
            if removelabeledpair.label == 0:
                assert (XY[1] == (-1) * metainfo.paras.regressiontau)
                del self.label0pairxs[removelabeledpair]
            else:
                assert (XY[1] == metainfo.paras.regressiontau)
                del self.label1pairxs[removelabeledpair]
            if self.regression != None:
                self.regression.disable(XY[0], XY[1])
            for eachevidenceintervalindex in range(0, len(self.evidenceinterval)):
                if XY[0] >= metafeature.evidenceinterval[eachevidenceintervalindex][0] and XY[0] < metafeature.evidenceinterval[eachevidenceintervalindex][1]:
                    self.evidenceinterval[eachevidenceintervalindex].remove(removelabeledpair)
                assert(removelabeledpair not in self.human_evidenceinterval[eachevidenceintervalindex])
            return

    def __init__(self, gml, fid):
        self.gml = gml
        self.fid = fid
        self.type, self.abbr, self.attributename, self.attributeindex, self.function, self.parameter = metafeature.rfid(gml, fid)
        self.pairs = set()
        self.gml.metafeatures[fid] = self
        self.for_inference = self.type in metafeature.types.for_inference
        self.frozen_monotony = self.type in metafeature.types.normalize_type
        self.normalize = self.type in metafeature.types.normalize_type or metafeature.types.normalize_function in self.function
        self.ruletype = self.type in metafeature.types.rule
        if self.ruletype == True:
            self.rule = metainfo.top.WAITING
        if self.for_inference == True:
            self.fid_index = metainfo.top.WAITING
            self.monotonyeffective = None
            self.polarenforce = None
            if self.type in metafeature.types.for_inference_polar0:
                self.polarenforce = 0
            elif self.type in metafeature.types.for_inference_polar1:
                self.polarenforce = 1
            self.alphabound = [-math.inf, math.inf]
            self.valuebound = [math.inf, -math.inf]
            self.label0pairxs = {}
            self.label1pairxs = {}
            self.labelpairXY = {}
            self.regression = None
            self.evidenceinterval = []
            self.human_evidenceinterval = []
            for eachevidenceintervalindex in range(0, len(metafeature.evidenceinterval)):
                self.evidenceinterval.append(set())
                self.human_evidenceinterval.append(set())
            self.tau = metainfo.top.NOT_AVAILABLE
            self.alpha = metainfo.top.NOT_AVAILABLE
            self.coeffective = None
        if self.frozen_monotony == True:
            self.alphabound = [-math.inf, math.inf]
            self.monotonyeffective = True

    def obsolete(self):
        for eachpair in self.pairs:
            del eachpair.metafeatures[self]
        self.pairs.clear()
        self.for_inference = False
        self.type = metafeature.types.OBSOLETE
        del self.gml.metafeatures[self.fid]

    def tonormalize(self):
        if self.for_inference == True:
            scaler = self.valuebound[1]
            if scaler == 0:
                assert (self.valuebound[0] == 0)
                for eachpair in self.pairs:
                    assert (eachpair.metafeatures[self] == 0)
                    eachpair.metafeatures[self] = 0.5
                self.valuebound = [0, 1]
            elif self.normalize == True:
                self.valuebound = [math.inf, - math.inf]
                for eachpair in self.pairs:
                    eachpair.metafeatures[self] = eachpair.metafeatures[self] / scaler
                    if eachpair.metafeatures[self] < self.valuebound[0]:
                        self.valuebound[0] = eachpair.metafeatures[self]
                    elif eachpair.metafeatures[self] > self.valuebound[1]:
                        self.valuebound[1] = eachpair.metafeatures[self]
            for eachpair in self.pairs:
                if eachpair.islabeled() == True:
                    self.append(eachpair)

    def singleside_harmonicparabound(self):
        # 标0的x和个数
        Counter0 = Counter(list(self.label0pairxs.values()))
        # 标1的x和个数
        Counter1 = Counter(list(self.label1pairxs.values()))
        # 标0的x范围
        Counter0bound = None
        # 标1的x范围
        Counter1bound = None
        if len(Counter0) > 0:
            Counter0bound = [min(Counter0.keys()), max(Counter0.keys())]
        else:
            Counter0bound = [math.inf, -math.inf]
        if len(Counter1) > 0:
            Counter1bound = [min(Counter1.keys()), max(Counter1.keys())]
        else:
            Counter1bound = [math.inf, -math.inf]
        # 标了的x最小值
        labaledxbound0 = min(Counter0bound[0], Counter1bound[0])
        # 标了的x最大值
        labaledxbound1 = max(Counter0bound[1], Counter1bound[1])
        # 标了的x最小值有标0的个数
        labaledxbound0_in0 = 0
        # 标了的x最小值有标1的个数
        labaledxbound0_in1 = 0
        if labaledxbound0 in Counter0:
            labaledxbound0_in0 = Counter0[labaledxbound0]
        if labaledxbound0 in Counter1:
            labaledxbound0_in1 = Counter1[labaledxbound0]
        # 标了的x最大值有标0的个数
        labaledxbound1_in0 = 0
        # 标了的x最大值有标1的个数
        labaledxbound1_in1 = 0
        if labaledxbound1 in Counter0:
            labaledxbound1_in0 = Counter0[labaledxbound1]
        if labaledxbound1 in Counter1:
            labaledxbound1_in1 = Counter1[labaledxbound1]
        if labaledxbound0_in1 > labaledxbound0_in0:
            if self.regression != None and self.regression.regression != None:
                if self.regression.regression.coef_[0] < 0:
                    singlepoint_XY = [[labaledxbound0, metainfo.paras.regressiontau]] * labaledxbound0_in1
                    singlepoint_XY += [[labaledxbound0, metainfo.paras.regressiontau * (-1)]] * labaledxbound0_in0
                    singlepoint_regression = runtime.linearregression(themetafeature=metainfo.top.SIFT, XY=singlepoint_XY, polarenforce=1, variablebound=[labaledxbound0, labaledxbound0])
                    conservative_predict = singlepoint_regression.regression.predict(np.array([labaledxbound0]).reshape(-1, 1))[0][0]
                    assert(conservative_predict > 0)
                    self.regression.regression.coef_[0] = 0
                    self.regression.regression.intercept_[0] = conservative_predict
                    self.regression.k = self.regression.regression.coef_[0]
                    self.regression.b = self.regression.regression.intercept_[0]
            self.monotonyeffective = True
            self.alphabound[0] = - math.inf
        if labaledxbound1_in0 > labaledxbound1_in1:
            if self.regression != None and self.regression.regression != None:
                if self.regression.regression.coef_[0] < 0:
                    singlepoint_XY = [[labaledxbound1, metainfo.paras.regressiontau]] * labaledxbound1_in1
                    singlepoint_XY += [[labaledxbound1, metainfo.paras.regressiontau * (-1)]] * labaledxbound1_in0
                    singlepoint_regression = runtime.linearregression(themetafeature=metainfo.top.SIFT, XY=singlepoint_XY, polarenforce=0, variablebound=[labaledxbound1, labaledxbound1])
                    conservative_predict = singlepoint_regression.regression.predict(np.array([labaledxbound1]).reshape(-1, 1))[0][0]
                    assert(conservative_predict < 0)
                    self.regression.regression.coef_[0] = 0
                    self.regression.regression.intercept_[0] = conservative_predict
                    self.regression.k = self.regression.regression.coef_[0]
                    self.regression.b = self.regression.regression.intercept_[0]
            self.monotonyeffective = True
            self.alphabound[1] = math.inf

    def metaweight(self, thepair, labeltype):
        themetaweight = None
        evidentialsupport, espredict = self.regression.predictconfidence(thepair.metafeatures[self])
        if labeltype == pair.labeltypes.APPROXIMATE:
            themetaweight = espredict
        elif self.tau != metainfo.top.NOT_AVAILABLE and self.alpha != metainfo.top.NOT_AVAILABLE:
            themetaweight = evidentialsupport * self.tau * (thepair.metafeatures[self] - self.alpha)
        if self.polarenforce == 0 and themetaweight > 0:
            themetaweight = 0
        elif self.polarenforce == 1 and themetaweight < 0:
            themetaweight = 0
        return themetaweight

    def regressionmodeling(self):
        if self.for_inference == True:
            self.regression = runtime.linearregression(themetafeature = self, XY = self.labelpairXY.values(), polarenforce = self.polarenforce, variablebound = self.valuebound)

    @staticmethod
    def fid(gml, metafeaturetype, attributeindex, functionname, abbr=False, parameter=None):
        attributename = None
        if type(attributeindex) == int:
            attributename = gml.RecordAttributes[attributeindex]
        else:
            attributename = attributeindex
        fid = (metafeaturetype + metafeature.types.parser + str(attributename) + metafeature.types.parser)
        if abbr is True:
            fid += (metafeature.types.abbr + metafeature.types.parser)
        fid += functionname
        if parameter is not None:
            parameterlabel = ''
            for eachparameter in parameter:
                if len(parameterlabel) > 0:
                    parameterlabel += metafeature.types.para
                parameterlabel += str(eachparameter)
            fid += (metafeature.types.parser + parameterlabel)
        return fid

    @staticmethod
    def rfid(gml, fid):
        fids = fid.split(metafeature.types.parser)
        if len(fids) >= 3:
            type = fids[0]
            if fids[1] == metafeature.types.abbr:
                abbr = True
                fids.remove(metafeature.types.abbr)
            else:
                abbr = False
            attributename = fids[1]
            attributeindex = int(gml.RecordAttributes.index(attributename))
            function = fids[2]
            parameter = None
            if len(fids) == 4:
                parameter = fids[3].split(metafeature.types.para)
            return type, abbr, attributename, attributeindex, function, parameter
        else:
            return None

    @staticmethod
    def find(gml, fid):
        if fid in gml.metafeatures:
            return gml.metafeatures[fid]
        else:
            return metafeature(gml, fid)

    def pairmetafeaturevalue(self, thepair, value):
        assert(self.gml == thepair.gml)
        if value != metainfo.top.NONE_VALUE:
            thepair.metafeatures[self] = value
            self.pairs.add(thepair)
            if self.for_inference == True and value != metainfo.top.WAITING:
                if value < self.valuebound[0]:
                    self.valuebound[0] = value
                if value > self.valuebound[1]:
                    self.valuebound[1] = value

    def __eq__(self, another):
        if type(self) == type(another) and self.fid == another.fid:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.fid)

class factorgraphinference(nn.Module):

    class stage(Flag):
        LEARNING = auto()
        INFERENCE = auto()

    inferencepair = None
    ruleresult = None
    sgmlresult = None
    classweight = None
    factors_polar0 = None
    factors_polar1 = None
    factors_polarnon = None
    # [tParatau]1,#Factor X ([tEvidentialsupport]#Factor,#Pair · ([x_Factor_i,Pair_j]#Factor,#Pair - [tAlpha]#Factor,1 X [1]#1,#Pair)).
    # torch.mul 对应位置的元素相乘 tensor.mm 矩阵相乘 torch.add 相加 torch.clamp 上下界截断 需要进行拼接torch.cat.
    # torch.sigmoid, torch.log, torch.sum, x = x.cuda().
    cFactor = None
    cFactorpolarnon = None
    cFactorpolar0 = None
    cFactorpolar1 = None
    cPair = None
    # for nn.Module auto reg
    # tParatau_1F = None
    tParataubound = None
    tEvidentialsupport_FP = None
    tEvidentialsupport_FpolarnonP = None
    tEvidentialsupport_Fpolar0P = None
    tEvidentialsupport_Fpolar1P = None
    tCoeffective_FP = None
    tCoeffective_FpolarnonP = None
    tCoeffective_Fpolar0P = None
    tCoeffective_Fpolar1P = None
    iPairpid = None
    tFeaturevalue_FpolarnonP = None
    tFeaturevalue_Fpolar0P = None
    tFeaturevalue_Fpolar1P = None
    tParaalphaboundlist_polarnon = None
    tParaalphaboundlist_polar0 = None
    tParaalphaboundlist_polar1 = None
    h1Pair_1P = None
    iFactormap = None
    iFactorpolarnonmap = None
    iFactorpolar0map = None
    iFactorpolar1map = None
    iPairmap = None
    tTraininglabels_1P = None
    tHardLabel_Dispatcher = None

    # for nn.Module auto reg
    # tParaalphalist_polarnon = None
    # tParaalphalist_polar0 = None
    # tParaalphalist_polar1 = None

    def __init__(self, inferencepair):
        self.inferencepair = inferencepair
        variables = set()
        factors = set()
        for eachfactor in inferencepair.metafeatures:
            if eachfactor.type in metafeature.types.for_inference and eachfactor.monotonyeffective == True:
                factors.add(eachfactor)
                if metainfo.paras.supervised == True:
                    for eachevidenceinterval in eachfactor.human_evidenceinterval:
                        if len(eachevidenceinterval) > 0:
                            selectedevidences = []
                            if metainfo.paras.evidenceintervallimit != None and len(eachevidenceinterval) > metainfo.paras.evidenceintervallimit:
                                # eachevidenceinterval - variables
                                existanceexcludedchoiceset = eachevidenceinterval - variables
                                existcount = len(eachevidenceinterval) - len(existanceexcludedchoiceset)
                                if existcount < metainfo.paras.evidenceintervallimit:
                                    existanceexcludedneedcount = metainfo.paras.evidenceintervallimit - existcount
                                    selectedevidences = list(np.random.choice(list(existanceexcludedchoiceset), size=existanceexcludedneedcount, replace=False, p=None))
                            else:
                                selectedevidences = eachevidenceinterval
                            variables.update(selectedevidences)
                for eachevidenceinterval in eachfactor.evidenceinterval:
                    if len(eachevidenceinterval) > 0:
                        selectedevidences = []
                        if metainfo.paras.evidenceintervallimit != None and len(eachevidenceinterval) > metainfo.paras.evidenceintervallimit:
                            # eachevidenceinterval - variables
                            existanceexcludedchoiceset = eachevidenceinterval - variables
                            existcount = len(eachevidenceinterval) - len(existanceexcludedchoiceset)
                            if existcount < metainfo.paras.evidenceintervallimit:
                                existanceexcludedneedcount = metainfo.paras.evidenceintervallimit - existcount
                                selectedevidences = list(np.random.choice(list(existanceexcludedchoiceset), size=existanceexcludedneedcount, replace=False, p=None))
                        else:
                            selectedevidences = eachevidenceinterval
                        variables.update(selectedevidences)

        super(factorgraphinference, self).__init__()
        cLabel0Pair = 0
        cLabel1Pair = 0
        self.cFactor = len(factors)
        self.cPair = len(variables)
        self.iPairmap = {}
        tTraininglabels_1P = [None] * self.cPair
        tHardLabel_Dispatcher = []
        pairindex = 0
        self.iPairpid = []
        for eachinferencepair in variables:
            self.iPairmap[pairindex] = eachinferencepair
            self.iPairmap[eachinferencepair] = pairindex
            self.iPairpid.append(eachinferencepair.pid)
            if eachinferencepair.ishumanlabeled() == False:
                tHardLabel_Dispatcher.append([1, 1, 0, 0])
                if eachinferencepair.label == 0:
                    tTraininglabels_1P[pairindex] = 0
                else:
                    tTraininglabels_1P[pairindex] = 1
            else:
                tHardLabel_Dispatcher.append([0, 0, 1, 1])
                if eachinferencepair.label == 0:
                    tTraininglabels_1P[pairindex] = 2
                else:
                    tTraininglabels_1P[pairindex] = 3
            if eachinferencepair.label == 0:
                cLabel0Pair += 1
            else:
                cLabel1Pair += 1
            pairindex += 1
        class1weight = 1
        if cLabel1Pair > 0:
            class1weight = float(cLabel0Pair) / cLabel1Pair
        self.classweight = torch.cuda.FloatTensor([1, class1weight, math.pow(metainfo.paras.hard_label_learn_enhance_multiplier, metainfo.paras.hard_label_learn_enhance_multiplier_coefficient) * math.pow(1, metainfo.paras.class_weight_multiplier_coefficient), math.pow(metainfo.paras.hard_label_learn_enhance_multiplier, metainfo.paras.hard_label_learn_enhance_multiplier_coefficient) * math.pow(class1weight, metainfo.paras.class_weight_multiplier_coefficient)])
        self.tTraininglabels_1P = torch.cuda.FloatTensor(tTraininglabels_1P).long()
        self.tHardLabel_Dispatcher = torch.cuda.FloatTensor(tHardLabel_Dispatcher)
        self.iFactorpolarnonmap = {}
        self.iFactorpolar0map = {}
        self.iFactorpolar1map = {}
        self.iFactormap = {}
        factorindex = 0
        factorpolar0index = 0
        factorpolar1index = 0
        factorpolarnonindex = 0
        self.factors_polarnon = {}
        self.factors_polar0 = {}
        self.factors_polar1 = {}
        for eachfactor in factors:
            if eachfactor.polarenforce == None:
                self.iFactormap[factorindex] = eachfactor
                self.iFactormap[eachfactor] = factorindex
                self.iFactorpolarnonmap[factorpolarnonindex] = eachfactor
                self.iFactorpolarnonmap[eachfactor] = factorpolarnonindex
                factorindex += 1
                factorpolarnonindex += 1
                currentfactorpairset = set()
                self.factors_polarnon[eachfactor] = currentfactorpairset
                for eachpair in variables:
                    if eachpair in eachfactor.pairs:
                        currentfactorpairset.add(eachpair)
        self.cFactorpolarnon = factorpolarnonindex
        for eachfactor in factors:
            if eachfactor.polarenforce == 0:
                self.iFactormap[factorindex] = eachfactor
                self.iFactormap[eachfactor] = factorindex
                self.iFactorpolar0map[factorpolar0index] = eachfactor
                self.iFactorpolar0map[eachfactor] = factorpolar0index
                factorindex += 1
                factorpolar0index += 1
                currentfactorpairset = set()
                self.factors_polar0[eachfactor] = currentfactorpairset
                for eachpair in variables:
                    if eachpair in eachfactor.pairs:
                        currentfactorpairset.add(eachpair)
        self.cFactorpolar0 = factorpolar0index
        for eachfactor in factors:
            if eachfactor.polarenforce == 1:
                self.iFactormap[factorindex] = eachfactor
                self.iFactormap[eachfactor] = factorindex
                self.iFactorpolar1map[factorpolar1index] = eachfactor
                self.iFactorpolar1map[eachfactor] = factorpolar1index
                factorindex += 1
                factorpolar1index += 1
                currentfactorpairset = set()
                self.factors_polar1[eachfactor] = currentfactorpairset
                for eachpair in variables:
                    if eachpair in eachfactor.pairs:
                        currentfactorpairset.add(eachpair)
        self.cFactorpolar1 = factorpolar1index

        self.h1Pair_1P = torch.cuda.FloatTensor(1, self.cPair).fill_(1)
        self.tParatau_1F = nn.Parameter(torch.cuda.FloatTensor(1, self.cFactor).fill_(metainfo.paras.inferencetauinit))
        self.tParataubound = [0, metainfo.paras.regressiontaubound]

        tFeaturevalue_FpolarnonP = []
        tEvidentialsupport_FpolarnonP = []
        tCoeffective_FpolarnonP = []
        self.tParaalphaboundlist_polarnon = []
        self.tParaalphalist_polarnon = nn.ParameterList().cuda()
        for factorindex in range(0, self.cFactorpolarnon):
            thefactor = self.iFactorpolarnonmap[factorindex]
            featurevaluerow = [metainfo.top.NONE_VALUE] * self.cPair
            Evidentialsupportrow = [0] * self.cPair
            referpairs = self.factors_polarnon[thefactor]
            for eachreferpair in referpairs:
                currentreferpairindex = self.iPairmap[eachreferpair]
                featurevaluerow[currentreferpairindex] = eachreferpair.metafeatures[thefactor]
                Evidentialsupportrow[currentreferpairindex] = eachreferpair.metafeature_evidentialsupport[thefactor]
            tFeaturevalue_FpolarnonP.append(featurevaluerow)
            tEvidentialsupport_FpolarnonP.append(Evidentialsupportrow)
            tCoeffective_FpolarnonP.append(thefactor.coeffective[self.iPairpid])
            self.tParaalphaboundlist_polarnon.append(thefactor.alphabound)
            initalphavalue = None
            if math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = np.mean(thefactor.alphabound)
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = thefactor.valuebound[0]
            elif math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = thefactor.valuebound[1]
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = np.mean(thefactor.valuebound)
            currentalphaparameter = nn.Parameter(torch.cuda.FloatTensor(1, 1).fill_(initalphavalue), requires_grad=True)
            self.tParaalphalist_polarnon.append(currentalphaparameter)
        self.tFeaturevalue_FpolarnonP = torch.cuda.FloatTensor(tFeaturevalue_FpolarnonP)
        self.tEvidentialsupport_FpolarnonP = torch.cuda.FloatTensor(tEvidentialsupport_FpolarnonP)
        self.tCoeffective_FpolarnonP = torch.cuda.FloatTensor(tCoeffective_FpolarnonP)

        tFeaturevalue_Fpolar0P = []
        tEvidentialsupport_Fpolar0P = []
        tCoeffective_Fpolar0P = []
        self.tParaalphaboundlist_polar0 = []
        self.tParaalphalist_polar0 = nn.ParameterList().cuda()
        for factorindex in range(0, self.cFactorpolar0):
            thefactor = self.iFactorpolar0map[factorindex]
            featurevaluerow = [metainfo.top.NONE_VALUE] * self.cPair
            Evidentialsupportrow = [0] * self.cPair
            referpairs = self.factors_polar0[thefactor]
            for eachreferpair in referpairs:
                currentreferpairindex = self.iPairmap[eachreferpair]
                featurevaluerow[currentreferpairindex] = eachreferpair.metafeatures[thefactor]
                Evidentialsupportrow[currentreferpairindex] = eachreferpair.metafeature_evidentialsupport[thefactor]
            tFeaturevalue_Fpolar0P.append(featurevaluerow)
            tEvidentialsupport_Fpolar0P.append(Evidentialsupportrow)
            tCoeffective_Fpolar0P.append(thefactor.coeffective[self.iPairpid])
            self.tParaalphaboundlist_polar0.append(thefactor.alphabound)
            initalphavalue = None
            if math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = np.mean(thefactor.alphabound)
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = thefactor.valuebound[0]
            elif math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = thefactor.valuebound[1]
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = np.mean(thefactor.valuebound)
            currentalphaparameter = nn.Parameter(torch.cuda.FloatTensor(1, 1).fill_(initalphavalue), requires_grad=True)
            self.tParaalphalist_polar0.append(currentalphaparameter)
        self.tFeaturevalue_Fpolar0P = torch.cuda.FloatTensor(tFeaturevalue_Fpolar0P)
        self.tEvidentialsupport_Fpolar0P = torch.cuda.FloatTensor(tEvidentialsupport_Fpolar0P)
        self.tCoeffective_Fpolar0P = torch.cuda.FloatTensor(tCoeffective_Fpolar0P)

        tFeaturevalue_Fpolar1P = []
        tEvidentialsupport_Fpolar1P = []
        tCoeffective_Fpolar1P = []
        self.tParaalphaboundlist_polar1 = []
        self.tParaalphalist_polar1 = nn.ParameterList().cuda()
        for factorindex in range(0, self.cFactorpolar1):
            thefactor = self.iFactorpolar1map[factorindex]
            featurevaluerow = [metainfo.top.NONE_VALUE] * self.cPair
            Evidentialsupportrow = [0] * self.cPair
            referpairs = self.factors_polar1[thefactor]
            for eachreferpair in referpairs:
                currentreferpairindex = self.iPairmap[eachreferpair]
                featurevaluerow[currentreferpairindex] = eachreferpair.metafeatures[thefactor]
                Evidentialsupportrow[currentreferpairindex] = eachreferpair.metafeature_evidentialsupport[thefactor]
            tFeaturevalue_Fpolar1P.append(featurevaluerow)
            tEvidentialsupport_Fpolar1P.append(Evidentialsupportrow)
            tCoeffective_Fpolar1P.append(thefactor.coeffective[self.iPairpid])
            self.tParaalphaboundlist_polar1.append(thefactor.alphabound)
            initalphavalue = None
            if math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = np.mean(thefactor.alphabound)
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == False:
                initalphavalue = thefactor.valuebound[0]
            elif math.isinf(thefactor.alphabound[0]) == False and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = thefactor.valuebound[1]
            elif math.isinf(thefactor.alphabound[0]) == True and math.isinf(thefactor.alphabound[1]) == True:
                initalphavalue = np.mean(thefactor.valuebound)
            currentalphaparameter = nn.Parameter(torch.cuda.FloatTensor(1, 1).fill_(initalphavalue), requires_grad=True)
            self.tParaalphalist_polar1.append(currentalphaparameter)
        self.tFeaturevalue_Fpolar1P = torch.cuda.FloatTensor(tFeaturevalue_Fpolar1P)
        self.tEvidentialsupport_Fpolar1P = torch.cuda.FloatTensor(tEvidentialsupport_Fpolar1P)
        self.tCoeffective_Fpolar1P = torch.cuda.FloatTensor(tCoeffective_Fpolar1P)

        self.tEvidentialsupport_FP = torch.cat((self.tEvidentialsupport_FpolarnonP, self.tEvidentialsupport_Fpolar0P, self.tEvidentialsupport_Fpolar1P), dim=0)
        self.tCoeffective_FP = torch.cat((self.tCoeffective_FpolarnonP, self.tCoeffective_Fpolar0P, self.tCoeffective_Fpolar1P), dim=0)

    def performinference(self):
        if self.cPair > 0:
            criterion = nn.NLLLoss(weight=self.classweight)
            optimizer = optim.Adam(self.parameters())
            y_labels = self.tTraininglabels_1P
            for eachround in range(0, metainfo.paras.optimizerrounds):
                optimizer.zero_grad()
                outputs = self(stage=factorgraphinference.stage.LEARNING)
                loss = criterion(outputs, y_labels)
                loss.backward()
                optimizer.step()
            self.boundenforce()
            for eachindex in range(0, self.cFactorpolarnon):
                currentparaalpha = self.tParaalphalist_polarnon[eachindex].item()
                thefactor = self.iFactorpolarnonmap[eachindex]
                thefactor.alpha = currentparaalpha
            for eachindex in range(0, self.cFactorpolar0):
                currentparaalpha = self.tParaalphalist_polar0[eachindex].item()
                thefactor = self.iFactorpolar0map[eachindex]
                thefactor.alpha = currentparaalpha
            for eachindex in range(0, self.cFactorpolar1):
                currentparaalpha = self.tParaalphalist_polar1[eachindex].item()
                thefactor = self.iFactorpolar1map[eachindex]
                thefactor.alpha = currentparaalpha
            for eachindex in range(0, self.cFactor):
                currentparatau = self.tParatau_1F[0][eachindex].item()
                thefactor = self.iFactormap[eachindex]
                thefactor.tau = currentparatau
        self(stage=factorgraphinference.stage.INFERENCE)
        style = None
        if runtime.probabilitypolar(self.inferencepair.probability) != self.inferencepair.truthlabel:
            style = runtime.console.styles.EXCEPTION
        else:
            style = runtime.console.styles.CORRECTION
        if metainfo.runningflags.Show_Detail == True:
            runtime.console.print(1, style, [1, 3, 5, 7, 9], 'GML:', self.ruleresult, ' ✉ P =', runtime.round(self.inferencepair.probability), 'sim =', runtime.round(self.inferencepair.similarity), 'with rule #', len(self.inferencepair.rules), 'w =', runtime.round(self.inferencepair.weight))

    def boundenforce(self):
        if self.cFactorpolarnon > 0:
            for factorindex in range(0, self.cFactorpolarnon):
                thealphabound = self.tParaalphaboundlist_polarnon[factorindex]
                self.tParaalphalist_polarnon[factorindex].data = self.tParaalphalist_polarnon[factorindex].clamp(thealphabound[0], thealphabound[1])
        if self.cFactorpolar0 > 0:
            for factorindex in range(0, self.cFactorpolar0):
                thealphabound = self.tParaalphaboundlist_polar0[factorindex]
                self.tParaalphalist_polar0[factorindex].data = self.tParaalphalist_polar0[factorindex].clamp(thealphabound[0], thealphabound[1])
        if self.cFactorpolar1 > 0:
            for factorindex in range(0, self.cFactorpolar1):
                thealphabound = self.tParaalphaboundlist_polar1[factorindex]
                self.tParaalphalist_polar1[factorindex].data = self.tParaalphalist_polar1[factorindex].clamp(thealphabound[0], thealphabound[1])
        self.tParatau_1F.data = self.tParatau_1F.clamp(self.tParataubound[0], self.tParataubound[1])

    def forward(self, stage):
        # [tParatau]1,#Factor X ([tEvidentialsupport]#Factor,#Pair · ([x_Factor_i,Pair_j]#Factor,#Pair - [tAlpha]#Factor,1 X [1]#1,#Pair)).
        # torch.mul 对应位置的元素相乘 tensor.mm 矩阵相乘 torch.add 相加 torch.clamp 上下界截断 需要进行拼接torch.cat.
        # torch.sigmoid, torch.log, torch.sum, x = x.cuda().
        if stage == factorgraphinference.stage.LEARNING:
            self.boundenforce()

            tAlphaweightpair_FpolarnanP = None
            tAlphaweightpair_Fpolar0P = None
            tAlphaweightpair_Fpolar1P = None

            if self.cFactorpolarnon > 0:
                tParaalphalist_Fpolarnan1 = torch.cat(list(self.tParaalphalist_polarnon), dim=0)
                tAlphaweightpair_FpolarnanP = self.tFeaturevalue_FpolarnonP.add(torch.neg(tParaalphalist_Fpolarnan1.mm(self.h1Pair_1P)))
            if self.cFactorpolar0 > 0:
                tParaalphalist_Fpolar01 = torch.cat(list(self.tParaalphalist_polar0), dim=0)
                tAlphaweightpair_Fpolar0P = self.tFeaturevalue_Fpolar0P.add(torch.neg(tParaalphalist_Fpolar01.mm(self.h1Pair_1P))).clamp((-1) * math.inf, 0)
            if self.cFactorpolar1 > 0:
                tParaalphalist_Fpolar11 = torch.cat(list(self.tParaalphalist_polar1), dim=0)
                tAlphaweightpair_Fpolar1P = self.tFeaturevalue_Fpolar1P.add(torch.neg(tParaalphalist_Fpolar11.mm(self.h1Pair_1P))).clamp(0, math.inf)

            tAlphaweightpaircat = []
            if runtime.isNone(tAlphaweightpair_FpolarnanP) == False:
                tAlphaweightpaircat.append(tAlphaweightpair_FpolarnanP)
            if runtime.isNone(tAlphaweightpair_Fpolar0P) == False:
                tAlphaweightpaircat.append(tAlphaweightpair_Fpolar0P)
            if runtime.isNone(tAlphaweightpair_Fpolar1P) == False:
                tAlphaweightpaircat.append(tAlphaweightpair_Fpolar1P)

            tAlphaweightpair_FP = torch.cat(tAlphaweightpaircat, dim=0)
            tOut_1P = self.tParatau_1F.mm(self.tCoeffective_FP.mul(self.tEvidentialsupport_FP.mul(tAlphaweightpair_FP))).sigmoid()
            tOut_1P = tOut_1P.reshape(self.cPair, 1)
            tOut_1P = tOut_1P.clamp(confidence.probability_smallbound, 1)
            tProbability0 = 1 - tOut_1P
            tProbability0 = tProbability0.clamp(confidence.probability_smallbound, 1)
            tClassdistribution = torch.cat((tProbability0, tOut_1P, tProbability0, tOut_1P), dim=1)
            tClassdistribution = torch.log(tClassdistribution)
            tClassdistribution = tClassdistribution.mul(self.tHardLabel_Dispatcher)
            return tClassdistribution
        else:
            inpairFeaturevalue_FpolarnonP = []
            inpairFeaturevalue_Fpolar0P = []
            inpairFeaturevalue_Fpolar1P = []
            inpairEvidentialsupport_FP = []
            inpairCoeffective = []
            tAlphaweightpair_FpolarnanP = None
            tAlphaweightpair_Fpolar0P = None
            tAlphaweightpair_Fpolar1P = None
            if self.cFactorpolarnon > 0:
                for eachindex in range(0, self.cFactorpolarnon):
                    thefactor = self.iFactorpolarnonmap[eachindex]
                    featurevaluerow = [self.inferencepair.metafeatures[thefactor]]
                    Evidentialsupportrow = [self.inferencepair.metafeature_evidentialsupport[thefactor]]
                    inpairFeaturevalue_FpolarnonP.append(featurevaluerow)
                    inpairEvidentialsupport_FP.append(Evidentialsupportrow)
                    inpairCoeffective.append([thefactor.coeffective[self.inferencepair.pid]])
                inpairFeaturevalue_FpolarnonP = torch.cuda.FloatTensor(inpairFeaturevalue_FpolarnonP)
                tParaalphalist_Fpolarnan1 = torch.cat(list(self.tParaalphalist_polarnon), dim=0)
                tAlphaweightpair_FpolarnanP = inpairFeaturevalue_FpolarnonP.add(torch.neg(tParaalphalist_Fpolarnan1))
            if self.cFactorpolar0 > 0:
                for eachindex in range(0, self.cFactorpolar0):
                    thefactor = self.iFactorpolar0map[eachindex]
                    featurevaluerow = [self.inferencepair.metafeatures[thefactor]]
                    Evidentialsupportrow = [self.inferencepair.metafeature_evidentialsupport[thefactor]]
                    inpairFeaturevalue_Fpolar0P.append(featurevaluerow)
                    inpairEvidentialsupport_FP.append(Evidentialsupportrow)
                    inpairCoeffective.append([thefactor.coeffective[self.inferencepair.pid]])
                inpairFeaturevalue_Fpolar0P = torch.cuda.FloatTensor(inpairFeaturevalue_Fpolar0P)
                tParaalphalist_Fpolar01 = torch.cat(list(self.tParaalphalist_polar0), dim=0)
                tAlphaweightpair_Fpolar0P = inpairFeaturevalue_Fpolar0P.add(torch.neg(tParaalphalist_Fpolar01)).clamp((-1) * math.inf, 0)
            if self.cFactorpolar1 > 0:
                for eachindex in range(0, self.cFactorpolar1):
                    thefactor = self.iFactorpolar1map[eachindex]
                    featurevaluerow = [self.inferencepair.metafeatures[thefactor]]
                    Evidentialsupportrow = [self.inferencepair.metafeature_evidentialsupport[thefactor]]
                    inpairFeaturevalue_Fpolar1P.append(featurevaluerow)
                    inpairEvidentialsupport_FP.append(Evidentialsupportrow)
                    inpairCoeffective.append([thefactor.coeffective[self.inferencepair.pid]])
                inpairFeaturevalue_Fpolar1P = torch.cuda.FloatTensor(inpairFeaturevalue_Fpolar1P)
                tParaalphalist_Fpolar11 = torch.cat(list(self.tParaalphalist_polar1), dim=0)
                tAlphaweightpair_Fpolar1P = inpairFeaturevalue_Fpolar1P.add(torch.neg(tParaalphalist_Fpolar11)).clamp(0, math.inf)
            inpairEvidentialsupport_FP = torch.cuda.FloatTensor(inpairEvidentialsupport_FP)
            inpairCoeffective = torch.cuda.FloatTensor(inpairCoeffective)

            sumsigruleweight = None
            if metainfo.method.Rule_LearnableWeight == False:
                sumsigruleweight = self.inferencepair.ruleweight(detailed=False)
            else:
                sumsigruleweight = 0

            tAlphaweightpaircat = []
            if type(tAlphaweightpair_FpolarnanP) != runtime.types.NoneType:
                tAlphaweightpaircat.append(tAlphaweightpair_FpolarnanP)
            if type(tAlphaweightpair_Fpolar0P) != runtime.types.NoneType:
                tAlphaweightpaircat.append(tAlphaweightpair_Fpolar0P)
            if type(tAlphaweightpair_Fpolar1P) != runtime.types.NoneType:
                tAlphaweightpaircat.append(tAlphaweightpair_Fpolar1P)

            finalweight = None

            if len(tAlphaweightpaircat) > 0:
                tAlphaweightpair_FP = torch.cat(tAlphaweightpaircat, dim=0)
                twithoutOut_1weight = self.tParatau_1F.mm(inpairCoeffective.mul(inpairEvidentialsupport_FP.mul(tAlphaweightpair_FP)))
                tOut_1weight = None
                if len(self.inferencepair.rules) == 0:
                    tOut_1weight = twithoutOut_1weight
                else:
                    tOut_1weight = (twithoutOut_1weight.mul(1)).add(sumsigruleweight)
                finalweight = tOut_1weight.item()
                self.inferencepair.weight, self.inferencepair.probability, self.inferencepair.entropy = runtime.weight2probabilityentropy(finalweight)
            else:
                # 0 is always much more than 1.
                zeroevidentialwithout_1weight = metainfo.top.SMALL_VALUE * (-1)
                zeroevidential_1weight = None
                if len(self.inferencepair.rules) == 0:
                    zeroevidential_1weight = zeroevidentialwithout_1weight
                else:
                    zeroevidential_1weight = zeroevidentialwithout_1weight + sumsigruleweight
                finalweight = zeroevidential_1weight.item()
                self.inferencepair.weight, self.inferencepair.probability, self.inferencepair.entropy = runtime.weight2probabilityentropy(finalweight)
            self.ruleresult, copy_sumsigruleweight = runtime.weightresultcorrect(finalweight, self.inferencepair, tolabeljudge=False)
            return