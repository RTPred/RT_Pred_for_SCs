#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : Get_Features.py


import rdkit
import xlrd
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# def read_data(file):
#     data = xlrd.open_workbook(file)
#     table = data.sheets()[3]
#     cols = table.ncols
#     rows = table.nrows
#     #print(f"The numbers of Rows:{rows} ")
#     all_vlue = []
#     all_Smi = []
#     T_Smi = []
#     for i in range(rows):
#         cell_Smi = table.cell_value(i,2)
#         cell_y = table.cell_value(i,3)
#         all_Smi.append(cell_Smi)
#         all_vlue.append(cell_y)
#
#     for i in range(9):
#         cell_Smi_T = table.cell_value(i,4)
#         T_Smi.append(cell_Smi_T)
#     Smi = all_Smi[1:]
#     RT = all_vlue[1:]
#     test_smi = T_Smi[1:]
#     print(f"Numbers of Smiles:{len(Smi)};Numbers of RT:{len(RT)};Numbers of Tset Smi:{len(test_smi)}")
#     return Smi, RT, test_smi
#
# def Calculate_Descriptors():
#     smi, rt, test_smi = read_data("RPLC_Data.xlsx")
#     canon_smiles = []
#     canon_smiles_T = []
#     for smiles in smi:
#         try:
#             smil = Chem.CanonSmiles(smiles)
#             canon_smiles.append(smil)
#         except:
#             print(f"The Invalid Smiles: {smiles}")
#
#     for smiles_t in test_smi:
#         try:
#             smi_T = Chem.CanonSmiles(smiles_t)
#             canon_smiles_T.append(smi_T)
#         except:
#             print(f"The Invalid Smiles: {smiles_t}")
#
#     #print(f"Number of canon_smiles:{len(canon_smiles)}")
#     ms = [Chem.MolFromSmiles(x) for x in canon_smiles]
#     ms_T = [Chem.MolFromSmiles(x) for x in canon_smiles_T]
#     descs = [desc_name[0] for desc_name in Descriptors._descList]
#     #print(f"Numbers of Descriptors: {len(descs)}")
#     desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
#     descriptors = pd.DataFrame([desc_calc.CalcDescriptors(m) for m in ms])
#     descriptors_T = pd.DataFrame([desc_calc.CalcDescriptors(m) for m in ms_T])
#     descriptors.columns = descs
#     descriptors_T.columns = descs
#
#     #print(descriptors)
#     RT = pd.DataFrame([r for r in rt])
#     RT.columns = ["Exp_RT"]
#
#     return descriptors, descriptors_T, RT
#
#    # dataset.to_csv('minidatabase.csv')

#coding=utf8
import sys
import importlib
importlib.reload(sys)

def get_value():
    file = pd.read_csv("Name_Smiles.csv", encoding='gbk')
    smi = file['Smiles']
    label = file['tR']
    return smi, label


def Calculate_PubChem():
    df = pd.DataFrame(pd.read_csv("./Descriptors/PBFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps1 = df[x_columns]
    print(f"PubChem Length:{np.array(fps1).shape}")
    return np.array(fps1)


def Calculate_ExtECFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/EXFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps2 = df[x_columns]
    print(f"ExtECFP Length:{np.array(fps2).shape}")
    return np.array(fps2)


def Calculate_EStECFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/ESFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps3 = df[x_columns]
    print(f"EStECFP Length:{np.array(fps3).shape}")
    return np.array(fps3)

def Calculate_FP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/FP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps4 = df[x_columns]
    print(f"FP Length:{np.array(fps4).shape}")
    return np.array(fps4)

def Calculate_GOFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/GOFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps5 = df[x_columns]
    print(f"GOFP Length:{np.array(fps5).shape}")
    return np.array(fps5)

def Calculate_KRFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/KRFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps6 = df[x_columns]
    print(f"KRFP Length:{np.array(fps6).shape}")
    return np.array(fps6)

def Calculate_KRFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/KRFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps7 = df[x_columns]
    print(f"KRFPC Length:{np.array(fps7).shape}")
    return np.array(fps7)

def Calculate_MACCSFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/MACCSFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps8 = df[x_columns]
    print(f"MACCSFP Length:{np.array(fps8).shape}")
    return np.array(fps8)

def Calculate_SBFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/SBFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps9 = df[x_columns]
    print(f"SBFP Length:{np.array(fps9).shape}")
    return np.array(fps9)

def Calculate_SBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/SBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps10 = df[x_columns]
    print(f"SBFPC Length:{np.array(fps10).shape}")
    return np.array(fps10)

def Calculate_1D_2D():
    df = pd.DataFrame(pd.read_csv("./Descriptors/1D_2D.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps11 = df[x_columns]
    print(f"1D_2D Length:{np.array(fps11).shape}")
    return np.array(fps11)

def Calculate_AP2DFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/AP2DFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps12 = df[x_columns]
    print(f"AP2DFP Length:{np.array(fps12).shape}")
    return np.array(fps12)

def Calculate_AP2DFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/AP2DFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps13 = df[x_columns]
    print(f"AP2DFPC Length:{np.array(fps13).shape}")
    return np.array(fps13)

def Calculate_KRSBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/KRSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps14 = df[x_columns]
    print(f"KRSBFPC Length:{np.array(fps14).shape}")
    return np.array(fps14)

def Calculate_FPSBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/FPSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps15 = df[x_columns]
    print(f"FPSBFPC Length:{np.array(fps15).shape}")
    return np.array(fps15)

def Calculate_FPKRFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/FPKRFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps16 = df[x_columns]
    print(f"FPKRFPC Length:{np.array(fps16).shape}")
    return np.array(fps16)

def Calculate_FPKRSBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/FPKRSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps17 = df[x_columns]
    print(f"FPKRSBFPC Length:{np.array(fps17).shape}")
    return np.array(fps17)

def Calculate_1D_2DSBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/1D_2DSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps18 = df[x_columns]
    print(f"1D_2DSBFPC Length:{np.array(fps18).shape}")
    return np.array(fps18)

def Calculate_1D_2DKRFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/1D_2DKRFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps19 = df[x_columns]
    print(f"1D_2DKRFPC Length:{np.array(fps19).shape}")
    return np.array(fps19)

def Calculate_KRFPCKRFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/KRFPCKRFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps20 = df[x_columns]
    print(f"KRFPCKRFP Length:{np.array(fps20).shape}")
    return np.array(fps20)

def Calculate_KRFPCKRFPSBFPC():
    df = pd.DataFrame(pd.read_csv("./Descriptors/KRFPCKRFPSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps21 = df[x_columns]
    print(f"KRFPCKRFPSBFPC Length:{np.array(fps21).shape}")
    return np.array(fps21)

def Calculate_1D_2DKRFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/1D_2DKRFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps22 = df[x_columns]
    print(f"1D_2DKRFP Length:{np.array(fps22).shape}")
    return np.array(fps22)


def Calculate_1D_2DKRFPCKRFP():
    df = pd.DataFrame(pd.read_csv("./Descriptors/1D_2DKRFPCKRFP.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps23 = df[x_columns]
    print(f"1D_2DKRFPCKRFP Length:{np.array(fps23).shape}")
    return np.array(fps23)

def Calculate_Des_4a5f():
    df = pd.DataFrame(pd.read_csv("./Descriptors/Des_4a5f.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps24 = df[x_columns]
    print(f"Des_4a5f Length:{np.array(fps24).shape}")
    return np.array(fps24)

def Calculate_test_prediction():
    df = pd.DataFrame(pd.read_csv("./Descriptors/test prediction.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps25 = df[x_columns]
    print(f"test prediction Length:{np.array(fps25).shape}")
    return np.array(fps25)

def Calculate_RFtest_prediction():
    df = pd.DataFrame(pd.read_csv("./Descriptors/RF test prediction.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps26 = df[x_columns]
    print(f"RF test prediction Length:{np.array(fps26).shape}")
    return np.array(fps26)

def Calculate_232_prediction():
    df = pd.DataFrame(pd.read_csv("./Descriptors/232 FPSBFPC.csv"))
    x_columns = [x for x in df.columns if x not in ['Name']]
    fps27 = df[x_columns]
    print(f"232 FPSBFPC Length:{np.array(fps27).shape}")
    return np.array(fps27)

def main():
    smi, label = get_value()
    fps1 = Calculate_PubChem()
    fps2 = Calculate_ExtECFP()
    fps3 = Calculate_EStECFP()
    fps4 = Calculate_FP()
    fps5 = Calculate_GOFP()
    fps6 = Calculate_KRFP()
    fps7 = Calculate_KRFPC()
    fps8 = Calculate_MACCSFP()
    fps9 = Calculate_SBFP()
    fps10 = Calculate_SBFPC()
    fps11 = Calculate_1D_2D()
    fps12 = Calculate_AP2DFP()
    fps13 = Calculate_AP2DFPC()
    fps14 = Calculate_KRSBFPC()
    fps15 = Calculate_FPSBFPC()
    fps16 = Calculate_FPKRFPC()
    fps17 = Calculate_FPKRSBFPC()
    fps18 = Calculate_1D_2DSBFPC()
    fps19 = Calculate_1D_2DKRFPC()
    fps20 = Calculate_KRFPCKRFP()
    fps21 = Calculate_KRFPCKRFPSBFPC()
    fps22 = Calculate_1D_2DKRFP()
    fps23 = Calculate_1D_2DKRFPCKRFP()
    fps24 = Calculate_Des_4a5f()
    fps25 = Calculate_test_prediction()
    fps26 = Calculate_RFtest_prediction()
    fps27 = Calculate_232_prediction()

    print(len(smi))


if __name__ == '__main__':
    main()
