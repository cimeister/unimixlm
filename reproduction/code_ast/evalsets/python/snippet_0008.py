# flake8: noqa

from cereal import car
from selfdrive.car import dbc_dict
from common.params import Params
Ecu = car.CarParams.Ecu

# Steer torque limits
class SteerLimitParams:
  params = Params()
  STEER_MAX = int(params.get('SteerMaxAdj'))   # 409 is the max, 255 is stock
  STEER_DELTA_UP = int(params.get('SteerDeltaUpAdj'))
  STEER_DELTA_DOWN = int(params.get('SteerDeltaDownAdj'))
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1

class CAR:
  # genesis
  GENESIS = "GENESIS 2015-2016"
  GENESIS_G70 = "GENESIS G70 2018"
  GENESIS_G80 = "GENESIS G80 2017"
  GENESIS_G90 = "GENESIS G90 2017"
  # hyundai
  ELANTRA = "HYUNDAI ELANTRA LIMITED ULTIMATE 2017"
  ELANTRA_GT_I30 = "HYUNDAI I30 N LINE 2019 & GT 2018 DCT"
  SONATA = "HYUNDAI SONATA 2020"
  SONATA_HEV = "HYUNDAI SONATA HEV 2020"
  SONATA19 = "HYUNDAI SONATA 2019"
  SONATA19_HEV = "HYUNDAI SONATA 2019 HEV"
  KONA = "HYUNDAI KONA 2019"
  KONA_EV = "HYUNDAI KONA EV 2019"
  KONA_HEV = "HYUNDAI KONA HEV 2019"
  IONIQ_EV = "HYUNDAI IONIQ ELECTRIC LIMITED 2019"
  IONIQ_HEV = "HYUNDAI IONIQ HYBRID PREMIUM 2017"
  SANTA_FE = "HYUNDAI SANTA FE LIMITED 2019"
  PALISADE = "HYUNDAI PALISADE 2020"
  VELOSTER = "HYUNDAI VELOSTER 2019"
  GRANDEUR = "GRANDEUR IG 2017-2020"
  GRANDEUR_HEV = "GRANDEUR IG HEV 2019-2020"
  NEXO = "HYUNDAI NEXO"
  # kia
  FORTE = "KIA FORTE E 2018"
  OPTIMA = "KIA OPTIMA SX 2019 & 2016"
  OPTIMA_HEV = "KIA OPTIMA HYBRID 2017 & SPORTS 2019"
  SPORTAGE = "KIA SPORTAGE S 2020"
  SORENTO = "KIA SORENTO GT LINE 2018"
  STINGER = "KIA STINGER GT2 2018"
  NIRO_EV = "KIA NIRO EV 2020 PLATINUM"
  NIRO_HEV = "KIA NIRO HEV 2018"
  CEED = "KIA CEED 2019"
  CADENZA = "KIA K7 2016-2019"
  CADENZA_HEV = "KIA K7 HEV 2016-2019"

class Buttons:
  NONE = 0
  RES_ACCEL = 1
  SET_DECEL = 2
  GAP_DIST = 3
  CANCEL = 4

params = Params()
fingerprint_issued_fix = params.get('FingerprintIssuedFix') == "1"

if fingerprint_issued_fix: # 핑거인식문제 혹은 다른차량과 핑거프린트 충돌이 나는경우 여기다가 핑거를 넣으시고 개발자 메뉴에서 핑거프린트 이슈차량 전용을 켜면 적용됩니다.
  FINGERPRINTS = {
    # genesis
    CAR.GENESIS: [{}],
    CAR.GENESIS_G70: [{}],
    CAR.GENESIS_G80: [{}],
    CAR.GENESIS_G90: [{}],
    # hyundai
    CAR.ELANTRA: [{}],
    CAR.ELANTRA_GT_I30: [{}],
    CAR.SONATA: [{}],
    CAR.SONATA_HEV: [{}],
    CAR.SONATA19: [{}],
    CAR.SONATA19_HEV: [{}],
    CAR.KONA: [{}],
    CAR.KONA_EV: [{}],
    CAR.KONA_HEV: [{}],
    CAR.IONIQ_HEV: [{}],
    CAR.IONIQ_EV: [{}],
    CAR.SANTA_FE: [{}],
    CAR.PALISADE: [{}],
    CAR.VELOSTER: [{}], 
    CAR.GRANDEUR: [{}],
    CAR.GRANDEUR_HEV: [{}],
    CAR.NEXO: [{}],
    # kia
    CAR.FORTE: [{}],
    CAR.OPTIMA: [{}],
    CAR.OPTIMA_HEV: [{}],
    CAR.SPORTAGE: [{}],
    CAR.SORENTO: [{}],
    CAR.STINGER: [{}],
    CAR.NIRO_EV: [{}],
    CAR.NIRO_HEV: [{}], 
    CAR.CEED: [{}],
    CAR.CADENZA: [{}],  
    CAR.CADENZA_HEV: [{}]
  }
else: # 핑거 프린트 이슈 없는 차량은 이곳에 넣으세요.
  FINGERPRINTS = {
    # genesis
    CAR.GENESIS: [{}],
    CAR.GENESIS_G70: [{}],
    CAR.GENESIS_G80: [{}],
    CAR.GENESIS_G90: [{}],
    # hyundai
    CAR.ELANTRA: [{}],
    CAR.ELANTRA_GT_I30: [{}],
    CAR.SONATA: [{}],
    CAR.SONATA_HEV: [{}],
    CAR.SONATA19: [{}],
    CAR.SONATA19_HEV: [{}],
    CAR.KONA: [{}],
    CAR.KONA_EV: [{}],
    CAR.KONA_HEV: [{}],
    CAR.IONIQ_HEV: [{68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 576: 8, 593: 8, 688: 5, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1173: 8, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1419: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470:8, 1476: 8, 1535: 8},
                    {68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 576:8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1173: 8, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8},
                    {68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 544: 8, 576: 8, 832: 8, 881: 8, 882: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1173: 8, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8}],
    CAR.IONIQ_EV: [{127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 7, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1168: 7, 1173: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1419: 8, 1425: 2, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1507: 8, 1535: 8},
                    {127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 7, 545: 8, 546: 8, 548: 8, 549: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1168: 7, 1173: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1507: 8, 1535: 8},
                    {127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 7, 546: 8, 832: 8, 881: 8, 882: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1168: 7, 1173: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1322: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1507: 8}],    
    CAR.SANTA_FE: [{}],
    CAR.PALISADE: [{}],
    CAR.VELOSTER: [{}], 
    CAR.GRANDEUR: [{67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 516: 8, 524: 8, 528: 8, 532: 8, 544: 8, 576: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 8, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 8, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 8, 1170: 8, 1173: 8, 1180: 8, 1186: 2, 1191: 2, 1193: 8, 1210: 8, 1225: 8, 1227: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 8, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1384: 8, 1407: 8, 1419: 8, 1427: 6, 1456: 4, 1470: 8},{
                    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 854 : 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8 , 1151: 6, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1185: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312 : 8, 1322: 8, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6 , 1456: 4, 1470: 8},{
                    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 546: 8, 547: 8, 549: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1185: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1384: 8, 1407: 8, 1419: 8, 1427: 6, 1456: 4, 1470: 8},{
                    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 546: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1185: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1456: 4, 1470: 8},{
                    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1185: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1456: 4, 1470: 8},{
                    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1156: 8, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1185: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1456: 4, 1470: 8}],
    CAR.GRANDEUR_HEV: [{68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 546: 8, 576: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1151: 6, 1156: 8, 1157: 4, 1168: 7, 1173: 8, 1185: 8, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8},{
                    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 576: 8, 593: 8, 688: 5, 832: 8, 865: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1108: 8, 1136: 6, 1138: 5, 1151: 8, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 8, 1173: 8, 1180: 8, 1186: 2, 1191: 2, 1193: 8, 1210: 8, 1225: 8, 1227: 8, 1265: 4, 1268: 8, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1407: 8, 1419: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8},{
                    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 516: 8, 544: 8, 576: 8, 593: 8, 688: 5, 832: 8, 865: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1108: 8, 1136: 6, 1138: 5, 1151: 8, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 8, 1173: 8, 1180: 8, 1186: 2, 1191: 2, 1193: 8, 1210: 8, 1225: 8, 1227: 8, 1265: 4, 1268: 8, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1407: 8, 1419: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8},{
                    68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 546: 8, 576: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1151: 6, 1156: 8, 1157: 4, 1168: 7, 1173: 8, 1185: 8, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1379: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8}],
    CAR.NEXO: [{}],
    # kia
    CAR.FORTE: [{}],
    CAR.OPTIMA: [{}],
    CAR.OPTIMA_HEV: [{}],
    CAR.SPORTAGE: [{}],
    CAR.SORENTO: [{}],
    CAR.STINGER: [{}],
    CAR.NIRO_EV: [{}],
    CAR.NIRO_HEV: [{}], 
    CAR.CEED: [{}],
    CAR.CADENZA: [{}],  
    CAR.CADENZA_HEV: [{}]
  }

# Don't use these fingerprints for fingerprinting, they are still used for ECU detection
IGNORED_FINGERPRINTS = [CAR.VELOSTER, CAR.GENESIS_G70, CAR.KONA]

CHECKSUM = {
  "crc8": [CAR.SANTA_FE, CAR.SONATA, CAR.PALISADE, CAR.SONATA_HEV],
  "6B": [CAR.SORENTO, CAR.GENESIS],
}

FEATURES = {
  # 캔오류 관련, 오류가 발생하는 경우는 본인 차종에 맞지 않는 캔신호가 들어오기때문입니다. 대부분 이곳을 수정하면 해결되나, 부득이 판다코드를 수정해야 될수도 있습니다.
  # debug 코드가 포함되어 있으면, /data/openpilot/selfdrive/debug 안에 몇가지 툴이 들어있습니다. 실행하시면 디버그에 도움이 되실겁니다. 팟팅!!!   
  # Use Cluster for Gear Selection, rather than Transmission
  "use_cluster_gears": {CAR.ELANTRA, CAR.KONA, CAR.ELANTRA_GT_I30, CAR.CADENZA, CAR.GRANDEUR},
  # Use TCU Message for Gear Selection
  "use_tcu_gears": {CAR.OPTIMA, CAR.SONATA19, CAR.VELOSTER},
  # Use E_GEAR Message for Gear Selection
  "use_elect_gears": {CAR.SONATA_HEV, CAR.SONATA19_HEV, CAR.KONA_EV, CAR.KONA_HEV, CAR.IONIQ_EV, CAR.IONIQ_HEV, CAR.GRANDEUR_HEV, CAR.NEXO,
                      CAR.OPTIMA_HEV, CAR.CADENZA_HEV, CAR.NIRO_EV, CAR.NIRO_HEV}, # 전기차 or 하이브리드 기어인식 부분
  # Use E_EMS11 Message for Gas and Brake for Hybrid/ELectric
  "use_elect_ems": {CAR.SONATA_HEV, CAR.SONATA19_HEV, CAR.KONA_EV, CAR.KONA_HEV, CAR.IONIQ_EV, CAR.IONIQ_HEV, CAR.GRANDEUR_HEV, CAR.NEXO,
                    CAR.OPTIMA_HEV, CAR.CADENZA_HEV, CAR.NIRO_EV, CAR.NIRO_HEV}, # 전기차 or 하이브리드 차량 넣어주세요.(가속페달관련)
  # send LFA MFA message for new HKG models
  "send_lfa_mfa": {CAR.GRANDEUR_HEV, CAR.GRANDEUR, CAR.KONA_HEV}, #차량의 LFA아이콘(핸들모양 아이콘)을 켜지게 하려면 여기다가 본인 차종을 넣으세요.
  "has_scc13": set([]), 
  "has_scc14": set([]), 
  # these cars use the FCA11 message for the AEB and FCW signals, all others use SCC12
  "use_fca": {CAR.SONATA, CAR.ELANTRA, CAR.ELANTRA_GT_I30, CAR.PALISADE, CAR.GENESIS_G70, CAR.GRANDEUR_HEV, CAR.KONA_HEV}, # 전방추돌관련 계기판 오류가 발생할 경우 여기다 본인 차종을 넣어보세요.
  "use_bsm": {CAR.SONATA, CAR.PALISADE, CAR.GENESIS, CAR.GENESIS_G70, CAR.GENESIS_G80, CAR.GENESIS_G90, CAR.NEXO,
              CAR.KONA, CAR.SONATA_HEV, CAR.SONATA19_HEV, CAR.KONA_EV, CAR.KONA_HEV, CAR.IONIQ_EV, CAR.IONIQ_HEV, CAR.GRANDEUR_HEV,
              CAR.OPTIMA_HEV, CAR.CADENZA_HEV, CAR.NIRO_EV, CAR.NIRO_HEV, CAR.ELANTRA, CAR.KONA, CAR.ELANTRA_GT_I30, CAR.CADENZA, CAR.GRANDEUR,
              CAR.OPTIMA, CAR.SONATA19, CAR.VELOSTER}, #후측방 감지 BSM 옵션이 있는 차량의 경우 넣어주세요.
}

DBC = {
  # genesis
  CAR.GENESIS: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G70: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G80: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G90: dbc_dict('hyundai_kia_generic', None),
  # hyundai
  CAR.ELANTRA: dbc_dict('hyundai_kia_generic', None),
  CAR.ELANTRA_GT_I30: dbc_dict('hyundai_kia_generic', None),
  CAR.SONATA: dbc_dict('hyundai_kia_generic', None),
  CAR.SONATA_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.SONATA19: dbc_dict('hyundai_kia_generic', None),
  CAR.SONATA19_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA_EV: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_EV: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.SANTA_FE: dbc_dict('hyundai_kia_generic', None),
  CAR.PALISADE: dbc_dict('hyundai_kia_generic', None),
  CAR.VELOSTER: dbc_dict('hyundai_kia_generic', None),
  CAR.GRANDEUR: dbc_dict('hyundai_kia_generic', None),
  CAR.GRANDEUR_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.NEXO: dbc_dict('hyundai_kia_generic', None),
  # kia
  CAR.FORTE: dbc_dict('hyundai_kia_generic', None),
  CAR.OPTIMA: dbc_dict('hyundai_kia_generic', None),
  CAR.OPTIMA_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.SPORTAGE: dbc_dict('hyundai_kia_generic', None),
  CAR.SORENTO: dbc_dict('hyundai_kia_generic', None),
  CAR.STINGER: dbc_dict('hyundai_kia_generic', None),
  CAR.NIRO_EV: dbc_dict('hyundai_kia_generic', None),
  CAR.NIRO_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.CEED: dbc_dict('hyundai_kia_generic', None),
  CAR.CADENZA: dbc_dict('hyundai_kia_generic', None),
  CAR.CADENZA_HEV: dbc_dict('hyundai_kia_generic', None),
}

STEER_THRESHOLD = 150