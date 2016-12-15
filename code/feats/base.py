"""
input :
output: feature engineer 23 basic features
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    # load data
    trn = pd.read_csv('../input/train_clean.csv')
    tst = pd.read_csv('../input/test_clean.csv')

    # label encode
    col = 'ind_empleado'
    keys = ['A', 'B', 'F', 'N', 'NN', 'S']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'pais_residencia'
    keys = ['LV', 'BE', 'BG', 'BA', 'BM', 'BO', 'JP', 'JM', 'BR', 'CR', 'BY', 'BZ', 'RU', 'RS', 'RO', 'GW', 'GT', 'GR',
            'GQ', 'GE', 'GB', 'GA', 'GN', 'GM', 'GI', 'GH', 'OM', 'HR', 'HU', 'HK', 'HN', 'VE', 'PR', 'PT', 'PY', 'PA',
            'PE', 'PK', 'PH', 'PL', 'EE', 'EG', 'ZA', 'EC', 'AL', 'AO', 'ET', 'ZW', 'ES', 'MD', 'MA', 'MM', 'ML', 'US',
            'MT', 'MR', 'UA', 'MX', 'IL', 'FR', 'FI', 'NI', 'NL', 'NN', 'NO', 'NG', 'NZ', 'CI', 'CH', 'CO', 'CN', 'CM',
            'CL', 'CA', 'CG', 'CF', 'CD', 'CZ', 'UY', 'CU', 'KE', 'KH', 'SV', 'SK', 'KR', 'KW', 'SN', 'SL', 'KZ', 'SA',
            'SG', 'SE', 'DO', 'DJ', 'DK', 'DE', 'DZ', 'MK', 'LB', 'TW', 'TR', 'TN', 'LT', 'LU', 'TH', 'TG', 'LY', 'AE',
            'AD', 'IS', 'IT', 'VN', 'AR', 'AU', 'AT', 'IN', 'IE', 'QA', 'MZ']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'sexo'
    keys = ['H', 'NN', 'V']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'tiprel_1mes'
    keys = ['A', 'NN', 'I', 'P']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'indresi'
    keys = ['N', 'NN', 'S']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'indext'
    keys = ['N', 'NN', 'S']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'conyuemp'
    keys = ['N', 'NN', 'S']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'canal_entrada'
    keys = ['013', 'KBJ', 'KBH', 'KBN', 'KBO', 'KBL', 'KBM', 'KBB', 'KBF', 'KBG', 'KBD', 'KBE', 'KBZ', 'KBX', 'KBY',
            'KBR', 'KBS', 'KBP', 'KBQ', 'KBV', 'KBW', 'KBU', 'RED', 'KDL', 'KDM', 'KDN', 'KDO', 'KDH', 'KDI', 'KDD',
            'KDE', 'KDF', 'KDG', 'KDA', 'KDB', 'KDC', 'KDX', 'KDY', 'KDZ', 'KDT', 'KDU', 'KDV', 'KDW', 'KDP', 'KDQ',
            'KDR', 'KDS', 'KFV', 'KFT', 'KFU', 'KFR', 'KFS', 'KFP', 'KFF', 'KFG', 'KFD', 'KFE', 'KFB', 'KFC', 'KFA',
            'KFN', 'KFL', 'KFM', 'KFJ', 'KFK', 'KFH', 'KFI', '007', '004', 'KHP', 'KHQ', 'KHR', 'KHS', 'KHK', 'KHL',
            'KHM', 'KHN', 'KHO', 'KHA', 'KHC', 'KHD', 'KHE', 'KHF', '025', 'KAC', 'KAB', 'KAA', 'KAG', 'KAF', 'KAE',
            'KAD', 'KAK', 'KAJ', 'KAI', 'KAH', 'KAO', 'KAN', 'KAM', 'KAL', 'KAS', 'KAR', 'KAQ', 'KAP', 'KAW', 'KAV',
            'KAU', 'KAT', 'KAZ', 'KAY', 'KCE', 'KCD', 'KCG', 'KCF', 'KCA', 'KCC', 'KCB', 'KCM', 'KCL', 'KCO', 'KCN',
            'KCI', 'KCH', 'KCK', 'KCJ', 'KCU', 'KCT', 'KCV', 'KCQ', 'KCP', 'KCS', 'KCR', 'KCX', 'NN', 'K00', 'KEO',
            'KEN', 'KEM', 'KEL', 'KEK', 'KEJ', 'KEI', 'KEH', 'KEG', 'KEF', 'KEE', 'KED', 'KEC', 'KEB', 'KEA', 'KEZ',
            'KEY', 'KEW', 'KEV', 'KEU', 'KES', 'KEQ', 'KGU', 'KGW', 'KGV', 'KGY', 'KGX', 'KGC', 'KGN']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'indfall'
    keys = ['N', 'NN', 'S']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'nomprov'
    keys = ['BURGOS', 'MADRID', 'CIUDAD REAL', 'BADAJOZ', 'LEON', 'SORIA', 'LERIDA', 'RIOJA, LA', 'PONTEVEDRA', 'NN',
            'SEVILLA', 'OURENSE', 'JAEN', 'CADIZ', 'AVILA', 'CORU', 'SEGOVIA', 'NAVARRA', 'SALAMANCA', 'PALENCIA',
            'BALEARS, ILLES', 'LUGO', 'PALMAS, LAS', 'GIPUZKOA', 'BIZKAIA', 'ZARAGOZA', 'TARRAGONA', 'GRANADA',
            'GIRONA', 'SANTA CRUZ DE TENERIFE', 'CEUTA', 'HUESCA', 'VALLADOLID', 'ZAMORA', 'CUENCA', 'MELILLA',
            'CORDOBA', 'ALICANTE', 'CASTELLON', 'VALENCIA', 'HUELVA', 'ALBACETE', 'TOLEDO', 'BARCELONA', 'GUADALAJARA',
            'MALAGA', 'MURCIA', 'ASTURIAS', 'ALAVA', 'TERUEL', 'CANTABRIA', 'CACERES', 'ALMERIA']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    col = 'segmento'
    keys = ['01 - TOP', 'NN', '03 - UNIVERSITARIO', '02 - PARTICULARES']
    lb = LabelEncoder().fit(keys)
    trn[col] = lb.transform(trn[col])
    tst[col] = lb.transform(tst[col])

    # save
    trn.to_csv('../input/feats/train_base.csv', index=False)
    tst.to_csv('../input/feats/test_base.csv', index=False)
