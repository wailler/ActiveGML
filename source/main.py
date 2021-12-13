import sys
import os
import warnings
import math
import numpy as np
from copy import deepcopy

from source import metainfo
from source.runtime import runtime
from source.SupervisedGML import SGML
from source import rule

infer_keytoken = metainfo.data.infer_keytoken

class debug:
    @staticmethod
    def init():
        pass

    @staticmethod
    def gml(thegml):
        pass

def metainit(data, eachexperiment):
    if metainfo.runningflags.refresh_trainingpool == True:
        if eachexperiment > 0:
            metainfo.paras.trainingpoolpath = metainfo.top.AUTO
    else:
        trainingpoolpathprefix = 'trainingpool-'
        trainingpoolpath = trainingpoolpathprefix + str(eachexperiment + 1) + '.csv'
        metainfo.paras.trainingpoolpath = metainfo.top.processpath + data.name + metainfo.top.pathindicator + trainingpoolpath
    metainfo.paras.tree_maxdepth = None
    metainfo.paras.infer_keytoken = None
    metainfo.paras.nlpw2vgroups = None
    GMLparas = metainfo.paras.__dict__
    for eachflag in data._fields:
        for eachpara in GMLparas:
            if eachpara[0:2] != '__':
                if eachflag == eachpara:
                    exec('metainfo.paras.' + eachpara + ' = ' + str(data.__getattribute__(eachflag)))
    metainfo.paras.nlpw2vgroups = metainfo.paras.infer_keytoken.nlpw2vgroups
    metainfo.paras.skyline_verify_steplimit = metainfo.paras.skyline_verify_steplimit_init
    metainfo.paras.genetics_recombination_cutcount = deepcopy(metainfo.paras.genetics_recombination_cutcount_init)
    metainfo.paras.raw_rule_approveprobability = deepcopy(metainfo.paras.raw_rule_approveprobability_init)

def init():
    metainfo.top.MAX_ENTROPY = runtime.entropy(0.5)
    rule.confidence.init()

def inited():
    metainfo.runningflags.refresh_cache = False
    metainfo.runningflags.refresh_trainingpool = False

class runningflags:
    supervised = metainfo.paras.supervised

if __name__ == '__main__':
    debug.init()
    icon = '🐱'
    title = 'Active Gradual Machine Learning / Skyline Learning'
    version = '2022'
    author = 'Anonymity (Double Blind Reviewing)'
    warnings.filterwarnings("ignore")
    print(runtime.console.color.UNDERLINE + runtime.console.color.BACKGROUND + runtime.console.color.DARKBLUE + icon + ' ' + title + ' ( Ver.' + version + ' )' + runtime.console.color.BACKGROUND + runtime.console.color.BLUE + ' - by ' + author + runtime.console.color.END)
    runtime.console.print(0, runtime.console.styles.INFO, [0], 'Select a data set:')
    runtime.console(None, metainfo.data.dataindexes, runtime.console.styles.INFO)
    data = metainfo.data.datas[int(input('dataindex :> '))]
    GMLparas = metainfo.paras.__dict__
    Runningflags = metainfo.runningflags.__dict__

    runtime.console.print(0, runtime.console.styles.INFO, [0], 'Select running flags:')
    flags = {}
    runningflags = runningflags.__dict__
    for eachflag in runningflags:
        if eachflag[0:2] != '__':
            flags[eachflag] = bool(runningflags[eachflag])
    runtime.console('running flags', flags, runtime.console.styles.INFO)
    changeflags = runtime.awaitinput('', 10, 'PLEASE ENTER', 'run flag paras', 2)
    if len(changeflags) > 0:
        flagindex = 0
        for eachflag in runningflags:
            if eachflag[0:2] != '__':
                flag = int(changeflags[flagindex])
                if type(runningflags[eachflag]) == bool:
                    flag = bool(flag)
                flags[eachflag] = flag
                for eachpara in GMLparas:
                    if eachpara[0:2] != '__':
                        if eachflag == eachpara:
                            exec('metainfo.paras.' + eachpara + ' = ' + str(flags[eachflag]))
                for eachpara in Runningflags:
                    if eachpara[0:2] != '__':
                        if eachflag == eachpara:
                            exec('metainfo.runningflags.' + eachpara + ' = ' + str(flags[eachflag]))
                flagindex += 1
        runtime.console('running flags saved', flags, runtime.console.styles.INFO)

    runtime.console.print(0, runtime.console.styles.INFO, [0], 'Change default paras (eg: humanproportion = 0.1 ; activerounds = 10):')
    cmd_paras = runtime.awaitinput('',3,'PLEASE ENTER','command codes').split(';')
    for each_cmd_para in cmd_paras:
        each_cmd_para = each_cmd_para.strip()
        if len(each_cmd_para) > 0:
            exec('metainfo.paras.' + each_cmd_para)
            para_value = each_cmd_para.split('=')
            para = para_value[0].strip()
            runtime.console.print(1, runtime.console.styles.REPORT, [0], 'metainfo.paras.' + para + ' = ' + str(eval('metainfo.paras.' + para)))

    runtime.console.print(0, runtime.console.styles.INFO, [0], 'para adapt:')
    for eachpara in GMLparas:
        if eachpara[0:2] != '__':
            if runtime.regularpattern.ispattern(string=eachpara, pattern=metainfo.paras.para_adapt, matchway=runtime.regularpattern.matchway.exact) == True:
                newvalue = None
                exec('newvalue = GMLparas[eachpara]' + metainfo.paras.para_adapt_formula)
                if newvalue == math.inf:
                    newvalue = 'math.inf'
                elif newvalue == (-1) * math.inf:
                    newvalue = '(-1) * math.inf'
                exec('metainfo.paras.' + eachpara + ' = ' + str(newvalue))
                runtime.console.print(1, runtime.console.styles.REPORT, [0], 'metainfo.paras.' + eachpara + ' = ' + str(eval('metainfo.paras.' + eachpara)))

    if metainfo.paras.supervised == True and data.name in metainfo.paras.supervised_easyproportion:
        metainfo.paras.easyproportion = metainfo.paras.supervised_easyproportion[data.name]
    if metainfo.runningflags.refresh_cache == True:
        csvfilepath = metainfo.top.resultspath + data.name + '_' + version + '.csv'
        existed = os.path.exists(csvfilepath)
        if existed == True:
            os.remove(csvfilepath)

    experiments = 10
    snapresult = metainfo.top.desktoppath + data.name + '.txt'
    if metainfo.paras.supervised == False:
        experiments = 1
        snapresult = None
    else:
        if os.path.exists(snapresult):
            os.remove(snapresult)
    c_results = []

    for eachexperiment in range(0, experiments):
        metainit(data, eachexperiment)
        init()

        probegml = None
        while(True):
            gml = SGML(version = version, data = data, probegml = probegml)
            debug.gml(thegml=gml)
            gml.activelearning()
            gml.EasyInstanceLabeling()
            gml.InfluenceModeling(initregression = True)
            gml.ScalableInference()
            gml.probe()
            gml.saveresult()
            probegml = deepcopy(gml.probegml)
            inited()
            if probegml.next == False:
                c_results.append(gml.results['F1'])
                if snapresult is not None:
                    with open(snapresult, "a") as f:
                        f.write(str(eachexperiment) + '    -----    ' + str(gml.results['F1']) + '  (' + str(gml.results['recall']) + ', ' + str(gml.results['precision']) + ', ↑ ' + str(gml.results['F1'] - gml.results['ugml_F1']) + ') , avg = ' + str(runtime.round(np.mean(c_results))) + '\r\n')
                break
            else:
                if snapresult is not None:
                    with open(snapresult, "a") as f:
                        info = 'round ' + str(gml.active_round) + '  :  ' + str(gml.results['F1']) + '  (' + str(gml.results['recall']) + ', ' + str(gml.results['precision']) + ')\r\n'
                        if gml.active_round == 1:
                            info = str(eachexperiment) + '    : ' + info
                        else:
                            info = ' ' * len(str(eachexperiment) + '    : ') + info
                        f.write(info)
                if metainfo.paras.raw_rule_approveprobability_adjust_period != None and gml.active_round % metainfo.paras.raw_rule_approveprobability_adjust_period == 0:
                    is_list_raw_rule_approveprobability = None
                    if type(metainfo.paras.raw_rule_approveprobability) == list:
                        is_list_raw_rule_approveprobability = True
                        metainfo.paras.raw_rule_approveprobability = np.array(metainfo.paras.raw_rule_approveprobability)
                    else:
                        is_list_raw_rule_approveprobability = False
                    is_list_raw_rule_approveprobability_adjust = None
                    if type(metainfo.paras.raw_rule_approveprobability_adjust) == list:
                        is_list_raw_rule_approveprobability_adjust = True
                        metainfo.paras.raw_rule_approveprobability_adjust = np.array(metainfo.paras.raw_rule_approveprobability_adjust)
                    else:
                        is_list_raw_rule_approveprobability_adjust = False
                    metainfo.paras.raw_rule_approveprobability += metainfo.paras.raw_rule_approveprobability_adjust
                    if is_list_raw_rule_approveprobability == True:
                        metainfo.paras.raw_rule_approveprobability = metainfo.paras.raw_rule_approveprobability.tolist()
                    if is_list_raw_rule_approveprobability_adjust == True:
                        metainfo.paras.raw_rule_approveprobability_adjust = metainfo.paras.raw_rule_approveprobability_adjust.tolist()
                    if is_list_raw_rule_approveprobability == True:
                        is_list_raw_rule_approveprobability_max_threshold = None
                        if type(metainfo.paras.raw_rule_approveprobability_max_threshold) == list:
                            is_list_raw_rule_approveprobability_max_threshold = True
                        else:
                            is_list_raw_rule_approveprobability_max_threshold = False
                            metainfo.paras.raw_rule_approveprobability_max_threshold = [metainfo.paras.raw_rule_approveprobability_max_threshold] * 2
                        for eachpolar in range(0, 2):
                            if metainfo.paras.raw_rule_approveprobability[eachpolar] > metainfo.paras.raw_rule_approveprobability_max_threshold[eachpolar]:
                                metainfo.paras.raw_rule_approveprobability[eachpolar] = metainfo.paras.raw_rule_approveprobability_max_threshold[eachpolar]
                        if is_list_raw_rule_approveprobability_max_threshold == False:
                            metainfo.paras.raw_rule_approveprobability_max_threshold = metainfo.paras.raw_rule_approveprobability_max_threshold[0]
                    else:
                        if metainfo.paras.raw_rule_approveprobability > metainfo.paras.raw_rule_approveprobability_max_threshold:
                            metainfo.paras.raw_rule_approveprobability = metainfo.paras.raw_rule_approveprobability_max_threshold