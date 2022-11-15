import pandas as pd
import numpy as np

feature_names = ['Accrual', 'BtM_GP', 'F_Score', 'GPA', 'NI_ReA', 'BM_GP_Mom', 'BM_Mom', 'FP', 'Ivol',
                 'Mom', 'NI_ReM', 'ROE', 'SUE', 'IRR', 'Season', "???", 'Beta', 'BidAsk_Spread',
                 'Days_ZeroTrade', 'Div',
                 'Dol_Volume', 'DolVolume_ME', 'IndMom_Size3', 'Max_Ret', 'Mom_0612', 'Mom_1318',
                 'Mom0206_volume5',
                 'Price', 'Size_Chen', 'Turnover_Vol', 'Volume_Trend', 'Volume_Vol', 'WeekHigh52', 'Abr_1m',
                 'AssetGrowth_Chen',
                 'AssetTurnover', 'BtM_Chen', 'CashProd', 'ConvDebt', 'DelCAPEX', 'DelEmp', 'DelEq',
                 'DelGM_DelSale',
                 'DelInvt', 'DelLTNOA', 'DelMom', 'DelPPE_Invt', 'DelSale', 'DelSale_DelAR', 'DelSale_DelInvt',
                 'DelSale_DelXSGA',
                 'DelSO', 'DelTax', 'Illiquidity', 'Ind_Mom_Chen', 'Ind_Mom0206', 'Ind_Mom0212', 'NumCEI',
                 'OrgCap', 'RD_ME', 'Sale_ME', 'SUE_Chen', 'Turnover']

# the feature importances are not ranked
df_all = pd.DataFrame({"feature_names": feature_names})
scores_all = pd.DataFrame({"feature_names": feature_names})
indices = [29, 30]
for i in indices:
    file_name = "Feature_importance" + str(i) + ".xlsx"
    df = pd.read_excel(file_name, usecols=[2])
    df.columns = ["Feature_importance" + str(i)]
    df_all = pd.concat([df_all, df], axis=1)
    ranking = np.array(df.rank(ascending=False, method="min")).squeeze()
    scores = 11 - ranking
    for k in range(len(scores)):
        if 1 <= scores[k] <= 5:
            scores[k] = 5
        elif -4 <= scores[k] <= 0:
            scores[k] = 3
        elif -9 <= scores[k] <= -5:
            scores[k] = 1
        elif scores[k] < -9:
            scores[k] = 0
    df2 = pd.DataFrame({"Score" + str(i): scores})
    scores_all = pd.concat([scores_all, df2], axis=1)

# with pd.ExcelWriter("Aggregated_feature_importance.xlsx") as writer:
#     df_all.to_excel(writer, index=False)

scores_all["Sum"] = scores_all.sum(axis=1)
with pd.ExcelWriter("Aggregated_feature_score.xlsx") as writer:
    scores_all.to_excel(writer, index=False)
