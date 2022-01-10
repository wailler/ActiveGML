import os
import collections
import platform
import math
from copy import deepcopy

class top:
    platform = platform.system()
    pathindicator = None
    if platform == 'Windows':
        pathindicator = '\\'
    else:
        pathindicator = '/'
    path = os.path.abspath(os.path.dirname(__file__))
    datapath = path[0:path.rfind(pathindicator)] + pathindicator + 'data' + pathindicator
    processpath = path[0:path.rfind(pathindicator)] + pathindicator + 'process' + pathindicator
    resultspath = path[0:path.rfind(pathindicator)] + pathindicator + 'results' + pathindicator
    APPROXI_FORBID_VALUE = -1
    MODERATE_SMALL_VALUE = 1e-3
    SMALL_VALUE = 1e-6
    SMALL_PROBABILITY = 0.05
    AUTO = 'AUTO'
    WAITING = 'WAITING'
    NOT_AVAILABLE = 'N/A'
    ALL_ATTRIBUTES = 'ALL_ATTRIBUTES'
    ALL_ATTRIBUTES_INDEX = 0
    NONE_VALUE = -1
    UNLABELED_VALUE = -1
    INDETERMINATE = -2
    SIFT = -3
    RESERVED = -4
    GROUND_TRUTH = 'groundtruth'
    MAX_BOUND = 1e20
    MAX_RETRY = 100
    specialvalue = {math.e:'math.e', math.inf:'∞', 10:'10'}
    MAX_ENTROPY = None
    OBSOLETE = False
    desktoppath = os.path.join(os.path.expanduser('~'), "Desktop") + pathindicator

class data:
    infer_keytoken = collections.namedtuple(                                                                                                                                              'infer_keytoken',
                                                                                                                                                                                                          ['idfrange'                                     , 'freqrange', 'patternrestrict'         , 'nlpw2vgroups'])
    datasetup = collections.namedtuple(
             'datasetup', ['name'    , 'data1path' , 'data2path'         , 'pairpath'          ,    'idname',  'approximateentropy_lowthreshold', 'tree_maxdepth', 'necessary_attribute', 'infer_keytoken'                                                                                                         ])
    abtbuy  = datasetup   ('abtbuy'  , 'tableA.csv', 'tableB.csv'        , 'pair_info_2020.csv',    'id'    ,  top.SMALL_PROBABILITY            ,  2             ,  [None, None]        ,  infer_keytoken([None, 1, 0, 0]                                 , 0          , 'grouppatterns,groups'    , 'ex,sparse'   ))
    songs   = datasetup   ('songs'   , 'msd.csv'   , 'msd.csv'           , 'pair_info_2020.csv',    'id'    ,  top.MODERATE_SMALL_VALUE         ,  2             ,  [None, None]        ,  infer_keytoken([None, 1, 0, 0, 0, 0, 0, 1]                     , 0          , 'grouppatterns,groups'    , 'attroccur'   ))
    acm     = datasetup   ('acm'     , 'tableA.csv', 'tableB.csv'        , 'pair_info_2020.csv',    'id'    ,  top.MODERATE_SMALL_VALUE         ,  2             ,  [None, None]        ,  infer_keytoken([None, 0.95, 1, 1, 1]                           , 0          , 'grouppatterns,groups'    , None          ))
    itunes  = datasetup   ('itunes'  , 'tableA.csv', 'tableB.csv'        , 'pair_info_2020.csv',    'id'    ,  top.MODERATE_SMALL_VALUE         ,  2             ,  [None, None]        ,  infer_keytoken([None, 0.95, 0, 0, 0, 0, 0, 0, 0]               , 0          , 'grouppatterns,groups'    , None          ))

    datas = [abtbuy, songs, acm, itunes]
    dataindexes = {}
    for eachindex in range(0, len(datas)):
        dataindexes[eachindex] = datas[eachindex].name

class paras:
    supervised = True
    easyproportion = 0.3
    supervised_easyproportion = {'abtbuy': 0.1, 'songs': 0.3, 'acm': 0.3, 'itunes': 0.3}
    humanproportion_perround = {'abtbuy': 0.01, 'songs': 0.01, 'acm': 0.01, 'itunes': 0.03}
    active_rounds = {'abtbuy': 4, 'songs': 4, 'acm': 4, 'itunes': 5}
    humanproportion = deepcopy(active_rounds)
    for eachdataset in humanproportion:
        if humanproportion_perround[eachdataset] != None and active_rounds[eachdataset] != None:
            humanproportion[eachdataset] = humanproportion_perround[eachdataset] * active_rounds[eachdataset]
    trainingpoolproportion = {'abtbuy': 0.6, 'songs': 0.6, 'acm': 0.6, 'itunes': 0.6}
    validationpoolproportion = {'abtbuy': 0.2, 'songs': 0.2, 'acm': 0.2, 'itunes': 0.2}
    skyline_verify_steplimit_init = {'abtbuy': 20, 'songs': 20, 'acm': 20, 'itunes': 20}
    skyline_verify_steplimit = skyline_verify_steplimit_init
    skyline_verify_steplimit_adjust = None
    skyline_verify_steplimit_skylineleast = 3
    evidenceintervallimit = 200
    evidentialsupport_topm = 2000
    approximateentropy_lowestk = 10
    optimizerrounds = 300
    updatecache_abscapacity = 100
    updatecache_proportion = top.SMALL_PROBABILITY
    regressiontaubound = math.inf
    weightbound = top.MAX_BOUND
    regressiontau = 10
    regressiondelta = 2
    inferencetauinit = 1
    rounddigits = 3
    evidenceintervalcount = 10
    approximateentropy_lowthreshold = top.MODERATE_SMALL_VALUE
    tree_split_significantdelta = 0
    tree_maxdepth = None
    tree_tridepth_exhausted = True
    tree_consider_trainingcountconfidence = False
    infer_keytoken = None
    nlpw2vgroups = None
    trainingpoolpath = None
    balance_rule_weight_multiplier = 1
    para_adapt = '((active|genetics_basicmetric|raw_rule|rule_approveprobability|discretize_splitcount)_.+)|(humanproportion.*)|(.+poolproportion)|(skyline_verify_steplimit_init)|(skyline_verify_steplimit)'
    para_adapt_formula = '[data.name]'
    discretize_splitcount_min = {'abtbuy':20, 'songs':20, 'acm':20, 'itunes':20}
    discretize_splitcount_max = {'abtbuy':20, 'songs':20, 'acm':20, 'itunes':20}
    genetics_mutation = 1
    genetics_recombination_cutcount_init = None
    genetics_recombination_cutcount = deepcopy(genetics_recombination_cutcount_init)
    genetics_recombination_cutcount_adjust = None
    genetics_basicmetric_selector = {'abtbuy':None, 'songs':None, 'acm':None, 'itunes':None}
    raw_rule_approveprobability_init = {'abtbuy':[0.999, 0.6], 'songs':[0.95, 0.95], 'acm':[0.95, 0.95], 'itunes':[0.95, 0.95]}
    raw_rule_approveprobability = deepcopy(raw_rule_approveprobability_init)
    raw_rule_approveprobability_adjust = {'abtbuy':[0.0, 0.0], 'songs':[0.0, 0.0], 'acm':[0.0, 0.0], 'itunes':[0.0, 0.0]}
    raw_rule_approveprobability_adjust_period = {'abtbuy':None, 'songs':None, 'acm':None, 'itunes':None}
    raw_rule_approveprobability_max_threshold = deepcopy(raw_rule_approveprobability_init)
    rule_approveprobability_onskyline = {'abtbuy':[0.95, 0.95], 'songs':[0.95, 0.95], 'acm':[0.95, 0.95], 'itunes':[0.95, 0.95]}
    # Large training pool to ensure knowable.
    rule_knowable = False
    forerule_inherit = True
    hard_label_learn_enhance_multiplier = 1
    hard_label_learn_enhance_multiplier_coefficient = 1
    class_weight_multiplier_coefficient = 1
    ruletype_coefficient = 0.5
    otherinfertype_coefficient = 1 - ruletype_coefficient
    rule_roi_fitness_coefficient = 0.5
    rule_roi_correcting_coefficient = 1 - rule_roi_fitness_coefficient

class method:
    Rule_LearnableWeight = True
    Rule_LocalSkyline = True
    Rule_NotOnlySkyline = True
    Rule_SkylineSubArea = False
    Rule_Balance = True
    Rule_Coefficient_Balance = False
    RuleCandidate_NewCoverage = False
    Token_SubCooccur = True
    Rule_SkylineSort = False

class runningflags:
    refresh_cache = True
    refresh_trainingpool = False
    basicmetric_embed_w2group = False
    Show_Detail = False
    Save_Mislabeledinfo = False
