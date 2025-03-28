import os

uap_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
uap_path = os.path.abspath(os.path.dirname(__file__))
data_root = os.path.join("/root/autodl-tmp/OPV2V")
model_root = os.path.join(uap_root, "models/OpenCOOD")
squeeze_root = os.path.join(uap_root, "models")
third_party_root = os.path.join(uap_root, "third_party")
SemanticOPV2V_root = "/share/SemanticOPV2V-OpenMMLab/SemanticOPV2V/4LidarCampos"


class_id_inv_map = {
    "unlabeled": 0,
    "car": 1,
    "truck": 2,
    "person": 3,
    "bicycle": 4,
    "other": 5,
}

len_record = [
    178,
    276,
    423,
    590,
    741,
    812,
    916,
    1083,
    1217,
    1319,
    1460,
    1537,
    1672,
    1776,
    1996,
    2170,
]

opv2v_label_mapping = {
    0: 0,  # unlabeled
    1: 1,  # Building
    2: 2,  # Fence
    3: 0,  # unlabeled
    4: 0,  # unlabeled
    5: 4,  # Pole
    6: 5,  # Road
    7: 5,  # Road
    8: 6,  # SideWalk
    9: 7,  # Vegetation
    10: 8,  # Vehicles
    11: 9,  # Wall
    12: 11,  # TrafficSign
    13: 0,  # unlabeled
    14: 3,  # Ground
    15: 12,  # Bridge
    16: 0,  # unlabeled
    17: 10,  # GuardRail
    18: 4,  # Pole
    19: 0,  # unlabeled
    20: 8,  # Vehicles
    21: 0,  # unlabeled
    22: 3,  # Terrain
}

scenario_maps = {
    "2021_08_20_21_48_35": "Town06",
    "2021_08_18_19_48_05": "Town06",
    "2021_08_20_21_10_24": "Town06",
    "2021_08_21_09_28_12": "Town06",
    "2021_08_22_07_52_02": "Town05",
    "2021_08_22_09_08_29": "Town05",
    "2021_08_22_21_41_24": "Town05",
    "2021_08_23_12_58_19": "Town05",
    "2021_08_23_15_19_19": "Town04",
    "2021_08_23_16_06_26": "Town04",
    "2021_08_23_17_22_47": "Town04",
    "2021_08_23_21_07_10": "Town10HD",
    "2021_08_23_21_47_19": "Town10HD",
    "2021_08_24_07_45_41": "Town10HD",
    "2021_08_24_11_37_54": "Town07",
    "2021_08_24_20_09_18": "Town04",
    "2021_08_24_20_49_54": "Town04",
    "2021_08_24_21_29_28": "Town04",
    "2021_08_16_22_26_54": "Town06",
    "2021_08_18_09_02_56": "Town06",
    "2021_08_18_18_33_56": "Town06",
    "2021_08_18_21_38_28": "Town06",
    "2021_08_18_22_16_12": "Town06",
    "2021_08_18_23_23_19": "Town06",
    "2021_08_19_15_07_39": "Town06",
    "2021_08_20_16_20_46": "Town06",
    "2021_08_20_20_39_00": "Town06",
    "2021_08_20_21_00_19": "Town06",
    "2021_08_21_09_09_41": "Town06",
    "2021_08_21_15_41_04": "Town05",
    "2021_08_21_16_08_42": "Town05",
    "2021_08_21_17_00_32": "Town05",
    "2021_08_21_21_35_56": "Town05",
    "2021_08_21_22_21_37": "Town05",
    "2021_08_22_06_43_37": "Town05",
    "2021_08_22_07_24_12": "Town05",
    "2021_08_22_08_39_02": "Town05",
    "2021_08_22_09_43_53": "Town05",
    "2021_08_22_10_10_40": "Town05",
    "2021_08_22_10_46_58": "Town06",
    "2021_08_22_11_29_38": "Town06",
    "2021_08_22_22_30_58": "Town05",
    "2021_08_23_10_47_16": "Town04",
    "2021_08_23_11_06_41": "Town05",
    "2021_08_23_11_22_46": "Town04",
    "2021_08_23_12_13_48": "Town05",
    "2021_08_23_13_10_47": "Town05",
    "2021_08_23_16_42_39": "Town04",
    "2021_08_23_17_07_55": "Town04",
    "2021_08_23_19_27_57": "Town10HD",
    "2021_08_23_20_47_11": "Town10HD",
    "2021_08_23_22_31_01": "Town10HD",
    "2021_08_23_23_08_17": "Town10HD",
    "2021_08_24_09_25_42": "Town07",
    "2021_08_24_09_58_32": "Town07",
    "2021_08_24_12_19_30": "Town07",
    "2021_09_09_13_20_58": "Town03",
    "2021_09_09_19_27_35": "Town01",
    "2021_09_10_12_07_11": "Town04",
    "2021_09_09_23_21_21": "Town03",
    "2021_08_21_17_30_41": "Town05",
    "2021_08_22_13_37_16": "Town06",
    "2021_08_22_22_01_17": "Town05",
    "2021_08_23_10_51_24": "Town05",
    "2021_08_23_13_17_21": "Town05",
    "2021_08_23_19_42_07": "Town10HD",
    "2021_09_09_22_21_11": "Town02",
    "2021_09_11_00_33_16": "Town10HD",
    "2021_08_18_19_11_02": "Town06",
}
