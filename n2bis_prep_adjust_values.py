


trigger_allsujet = {

    #### adjust full protocol
    'CHEe': {
        'trig_name' : ['CV_start', 'CV_stop', '31',   '32',   '11',   '12',   '71',   '72',   '11',   '12',   '51',    '52',    '11',    '12',    '51',    '52',    '31',    '32'],
        'trig_time' : [0,          153600,    463472, 555599, 614615, 706758, 745894, 838034, 879833, 971959, 1009299, 1101429, 1141452, 1233580, 1285621, 1377760, 1551821, 1643948],
    },

    'GOBc': {
        'trig_name' : ['CV_start', 'CV_stop', '31',   '32',   '11',   '12',   '31',   '32',   '11',    '12',     '51',    '52',    '11',    '12',    '51',    '52',    '61',    '62',    '61',    '62',    'MV_start', 'MV_stop'],
        'trig_time' : [0,          307200,    312192, 496437, 572960, 757212, 848948, 1033187, 1109909, 1294144, 1355106, 1539378, 1630292, 1814548, 1877125, 2061389, 2437513  ,2621760,2682622,  2866880, 2947600,    3219072],
    },

    'MAZm': {
        'trig_name' : ['CV_start','CV_stop',  '31',   '32',   '11',   '12',   '31',   '32',   '11',   '12',   '51',   '52',   '11',   '12',   '51',   '52',   '61',    '62',    '61',    '62',    'MV_start', 'MV_stop'],
        'trig_time' : [0,         153600,     164609, 240951, 275808, 367946, 396690, 488808, 529429, 621558, 646959, 739078, 763014, 855141, 877518, 969651, 1102256, 1194377, 1218039, 1310170, 1391000,    1558000],
    },  

    'TREt': {
        'trig_name' : ['CV_start','CV_stop',  '31',   '32',   '11',   '12',   '31',   '32',   '11',   '12',   '51',   '52',   '11',     '12',   '51',     '52',   '61',    '62',    '61',    '62',    'MV_start', 'MV_stop'],
        'trig_time' : [0,         153600,     419610, 511741, 553776, 645892, 679831, 771950, 797374, 889488, 917962, 1010072, 1037339, 1129467, 1157323, 1249450, 1492770, 1584894, 1666461, 1679260, 1679260,    1751039],
    },  

    #### adjust FR_CV
    'BANc': {
        'trig_name' : ['CV_start', 'CV_stop'],
        'trig_time' : [0,          153600],
    },  

    'KOFs': {
        'trig_name' : ['CV_start', 'CV_stop'],
        'trig_time' : [0,           153600],
    },  

    'LEMl': {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [0,           153600]
    },

    'MUGa': {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [0,           153600]
    },

    'pat_02459_0912' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [830000, 980000]
    },

    'pat_02476_0929' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [1430000, 1575000]
    },

    'pat_02495_0949' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [996000, 1146000]
    },

    'pat_03083_1527' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [384000, 534000]
    },

    'pat_03105_1551' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [812000, 962000]
    },

    'pat_03128_1591' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [120500, 270500]
    },

    'pat_03138_1601' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [859000, 1009000]
    },

    'pat_03146_1608' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [275000, 425000]
    },
    
    'pat_03174_1634' : {   'trig_name' : ['CV_start', 'CV_stop'],
                'trig_time' : [531000, 681000]
    },

}






ecg_adjust_allsujet = {

    #### adjust full protocole
    'CHEe': {   'ecg_events_corrected' : [339111, 347393, 358767, 360242, 363559, 460709, 554965, 870178, 871428, 873406, 1142520, 1298203, 1297285, 1297760],
                'ecg_events_to_remove' : []
    },
    'GOBc': {   'ecg_events_corrected' : [574640, 902240, 903060, 1205660, 1206290, 1632024, 1784291, 1895429, 2963796, 2973557, 2991529, 2998127, 914419, 1206975, 1.70231e6, 1721770, 1730034, 1730871, 1731349, 1732781, 1.78158e6, 1.78227e6, 1.78493e6, 2199475, 2321365, 2322851, 2473316, 2797246, 2800339, 2.88987e6, 2.89661e6, 2968938],
                'ecg_events_to_remove' : []
    },
    'MAZm': {   'ecg_events_corrected' : [8.6539e5, 1.28444e6, 1.35341e6, 1285153, 1286958, 1287326, 1287692, 1288073, 1376352, 799279, 999758, 1011149, 1162670],
                'ecg_events_to_remove' : [1353229]
    },
    'TREt': {   'ecg_events_corrected' : [21506, 142918, 289897, 1308016, 1.36292e6, 1.36483e6, 1.36523e6, 1.36563e6, 1.36647e6, 1.36690e6, 1.36730e6, 1.36968e6, 1.37006e6, 1.60849e6, 1626322, 1629380, 1630109, 1636351, 1641558, 1642374, 1645133],
                'ecg_events_to_remove' : []
    },

    #### adjust FR_CV
    'MUGa': {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },

    'BANc': {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'KOFs': {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'LEMl': {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_02459_0912' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_02476_0929' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_02495_0949' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03083_1527' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03105_1551' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03128_1591' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03138_1601' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03146_1608' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    },
    'pat_03174_1634' : {   'ecg_events_corrected' : [],
                'ecg_events_to_remove' : []
    }
}