# -*- coding: utf-8 -*-
"""Project 5: Phan tich bo du lieu IBM HR Analytics Employee Attrition & Performance.
File nay co the chay doc lap hoac chuyen thanh notebook Jupyter.
"""
# %% 01. Khoi tao moi truong va doc du lieu
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

ROOT = Path.cwd()
if (ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv').exists():
    SOURCE = ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
elif (ROOT / 'WA_Fn-UseC_-HR-Employee-Attrition.csv').exists():
    SOURCE = ROOT / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
else:
    # Colab: tai du lieu goc tu trang du lieu cong khai Kaggle.
    # Neu Kaggle khong truy cap duoc, hay tai CSV vao Colab va chay lai.
    try:
        import kagglehub
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
        import kagglehub
    folder = Path(kagglehub.dataset_download('pavansubhasht/ibm-hr-analytics-attrition-dataset'))
    files = list(folder.rglob('WA_Fn-UseC_-HR-Employee-Attrition.csv'))
    if not files:
        raise FileNotFoundError('Khong tim thay CSV tu Kaggle. Hay tai file goc vao thu muc lam viec.')
    SOURCE = files[0]

OUTPUT = ROOT / 'figures'
OUTPUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(SOURCE)
df = raw.copy()
df['AttritionFlag'] = df['Attrition'].eq('Yes').astype(int)
print('Kich thuoc bo du lieu:', raw.shape)
print('Phan bo nhan Attrition:\n', raw['Attrition'].value_counts())

# %% 02. Kiem tra chat luong va tinh nhat quan
missing = raw.isna().sum()
constant_cols = raw.columns[raw.nunique(dropna=False) == 1].tolist()
duplicates = int(raw.duplicated().sum())
repeated_ids = int(raw['EmployeeNumber'].duplicated().sum())
logical_checks = {
    'YearsAtCompany > TotalWorkingYears': int((raw.YearsAtCompany > raw.TotalWorkingYears).sum()),
    'YearsInCurrentRole > YearsAtCompany': int((raw.YearsInCurrentRole > raw.YearsAtCompany).sum()),
    'YearsSinceLastPromotion > YearsAtCompany': int((raw.YearsSinceLastPromotion > raw.YearsAtCompany).sum()),
    'YearsWithCurrManager > YearsAtCompany': int((raw.YearsWithCurrManager > raw.YearsAtCompany).sum()),
    'Age < 18': int((raw.Age < 18).sum()),
    'MonthlyIncome <= 0': int((raw.MonthlyIncome <= 0).sum()),
}
assert missing.sum() == 0 and duplicates == 0 and repeated_ids == 0

# IQR chi canh bao gia tri cao thap, khong dong nghia voi du lieu sai.
quant_cols = ['MonthlyIncome', 'DistanceFromHome', 'Age', 'TotalWorkingYears',
              'YearsAtCompany', 'DailyRate', 'NumCompaniesWorked', 'YearsSinceLastPromotion']
outliers = {}
for col in quant_cols:
    q1, q3 = raw[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers[col] = {'n': int(((raw[col] < low) | (raw[col] > high)).sum()),
                     'lower': float(low), 'upper': float(high),
                     'min': float(raw[col].min()), 'max': float(raw[col].max())}
# Diagnostic cho toan bo truong so: cac thang do thu bac/so dem co the bi IQR gan co nham.
all_numeric = [c for c in raw.select_dtypes(include='number').columns
               if c not in ('EmployeeNumber','EmployeeCount','StandardHours')]
all_iqr = []
for col in all_numeric:
    q1_all,q3_all=raw[col].quantile([.25,.75]); iq=q3_all-q1_all
    all_iqr.append({'variable':col,'q1':float(q1_all),'q3':float(q3_all),
                    'iqr':float(iq),
                    'n_flagged':int(((raw[col]<q1_all-1.5*iq)|(raw[col]>q3_all+1.5*iq)).sum())})
all_iqr_df=pd.DataFrame(all_iqr).sort_values('n_flagged',ascending=False)
all_iqr_df.to_csv(ROOT/'iqr_all_numeric.csv',index=False,encoding='utf-8-sig')
print('Gia tri thieu:', int(missing.sum()), 'Ban ghi trung:', duplicates)
print('Cot hang so:', constant_cols)
print('Kiem tra logic:', logical_checks)
print('Ngoai le IQR:', {k: v['n'] for k, v in outliers.items()})

# %% 03. Ma hoa nhom phuc vu phan tich mo ta
# Giữ nguyên dữ liệu gốc. Nhóm được định nghĩa trước để tránh dò ngưỡng có lợi.
df['TenureGroup'] = pd.cut(df['YearsAtCompany'], [-1, 1, 4, 9, 50],
                           labels=['0 đến 1 năm', '2 đến 4 năm', '5 đến 9 năm', 'Từ 10 năm'])
df['IncomeQuintile'] = pd.qcut(df['MonthlyIncome'], q=5,
                              labels=['Q1 thấp nhất', 'Q2', 'Q3', 'Q4', 'Q5 cao nhất'])
df['OT_TenureGroup'] = np.select(
    [(df['OverTime'].eq('Yes')) & (df['YearsAtCompany'] <= 1),
     (df['OverTime'].eq('Yes')) & (df['YearsAtCompany'] > 1),
     (df['OverTime'].eq('No')) & (df['YearsAtCompany'] <= 1)],
    ['Có tăng ca, 0 đến 1 năm', 'Có tăng ca, trên 1 năm',
     'Không tăng ca, 0 đến 1 năm'], default='Không tăng ca, trên 1 năm')

def group_summary(col):
    t = df.groupby(col, observed=False)['AttritionFlag'].agg(['sum','count','mean']).reset_index()
    t.columns = [col, 'left', 'n', 'rate']
    return t

def chi2_v(col):
    tab = pd.crosstab(df[col], df['AttritionFlag'])
    chi, p, _, _ = chi2_contingency(tab, correction=False)
    n = tab.to_numpy().sum()
    v = float(np.sqrt(chi/(n * min(tab.shape[0] - 1, tab.shape[1] - 1))))
    return {'stat':float(chi), 'p':float(p), 'v':v, 'dof':int((tab.shape[0]-1)*(tab.shape[1]-1))}

test_cols = {'H1':'OverTime', 'H2':'BusinessTravel', 'H3':'JobSatisfaction',
             'H4':'TenureGroup', 'H6':'WorkLifeBalance', 'H7':'JobRole',
             'H8':'PerformanceRating', 'H9':'StockOptionLevel'}
tests = {k: dict(variable=v, **chi2_v(v)) for k,v in test_cols.items()}
left_pay = df.loc[df.AttritionFlag == 1, 'MonthlyIncome']
stay_pay = df.loc[df.AttritionFlag == 0, 'MonthlyIncome']
u_stat, u_p = mannwhitneyu(left_pay, stay_pay, alternative='two-sided')
rank_biserial = float(2*u_stat/(len(left_pay)*len(stay_pay)) - 1)
tests['H5'] = {'variable':'MonthlyIncome', 'stat':float(u_stat), 'p':float(u_p),
               'rank_biserial':rank_biserial}
keys = ['H1','H2','H3','H4','H5','H6','H7','H8','H9']
_, adjusted, _, _ = multipletests([tests[k]['p'] for k in keys], alpha=.05,
                                   method='fdr_bh')
for k, a in zip(keys, adjusted):
    tests[k]['p_bh'] = float(a)
print('Ket qua kiem dinh:\n', pd.DataFrame(tests).T.to_string())

# %% 04. Truc quan hoa nhung phat hien can thiet
NAVY = '#2e536d'
COPPER = '#c36b51'
GRAY = '#6b7280'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.titlesize':12, 'axes.labelsize':10,
                     'figure.facecolor':'white', 'axes.spines.top':False,
                     'axes.spines.right':False, 'savefig.bbox':'tight'})

def save_chart(name):
    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT / (name + '.png'), dpi=175, facecolor='white')
    plt.close()

def bars(col, name, title, order=None, horiz=False):
    s = group_summary(col)
    if order is not None:
        s[col] = pd.Categorical(s[col], categories=order, ordered=True)
        s = s.sort_values(col)
    vn_labels={'OverTime':{'No':'Không tăng ca','Yes':'Có tăng ca'},
      'BusinessTravel':{'Non-Travel':'Không công tác','Travel_Rarely':'Công tác ít',
                        'Travel_Frequently':'Công tác thường xuyên'}}
    display_labels=s[col].astype(str).replace(vn_labels.get(col,{}))
    if horiz:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        cc=[COPPER if float(v)>=float(s.rate.max())-1e-12 else NAVY for v in s.rate]
        bars_ = ax.barh(display_labels, s.rate*100, color=cc)
        ax.invert_yaxis()
        ax.set_xlim(0, min(60, float(s.rate.max()*100) + 12))
        for b, (_, row) in zip(bars_, s.iterrows()):
            ax.text(b.get_width()+.6, b.get_y()+b.get_height()/2,
                    f"{np.floor(row['rate']*1000+.5000001)/10:.1f}%  (n={int(row['n'])})", va='center', fontsize=9)
        ax.set_xlabel('Tỷ lệ ghi nhận nghỉ việc (%)')
    else:
        fig, ax = plt.subplots(figsize=(8.2, 3.9))
        x = np.arange(len(s)); cc=[COPPER if float(v)>=float(s.rate.max())-1e-12 else NAVY for v in s.rate]
        bb=ax.bar(x,s.rate*100,color=cc,width=.65)
        ax.set_xticks(x, display_labels, fontsize=9)
        ax.set_ylim(0,float(s.rate.max()*100)+max(9, float(s.rate.max()*100)*.25))
        for b, (_,row) in zip(bb,s.iterrows()):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.7,
                    f"{np.floor(row['rate']*1000+.5000001)/10:.1f}%\nn={int(row['n'])}", ha='center',va='bottom',fontsize=9)
        ax.set_ylabel('Tỷ lệ ghi nhận nghỉ việc (%)')
    ax.set_title(title, loc='left', fontweight='bold', pad=14)
    ax.grid(axis='x' if horiz else 'y',alpha=.16)
    save_chart(name)

fig, ax = plt.subplots(figsize=(7.7,3.4))
c = raw.Attrition.value_counts().reindex(['No','Yes'])
b = ax.bar(['Không nghỉ', 'Đã nghỉ'],c.values,color=[NAVY,COPPER],width=.55)
for el,v in zip(b,c.values):
    ax.text(el.get_x()+el.get_width()/2,el.get_height()+20,
            f'{v:,}  ({100*v/len(raw):.1f}%)',ha='center',fontsize=10)
ax.set_ylim(0,c.max()*1.2);ax.set_title('Phân bố trạng thái nghỉ việc trong mẫu',loc='left',fontweight='bold')
ax.set_ylabel('Số người trong bộ dữ liệu');ax.grid(axis='y',alpha=.14)
save_chart('01_attrition_distribution')

fig, ax=plt.subplots(figsize=(8.3,3.8))
outs = pd.Series({k:v['n'] for k,v in outliers.items()}).sort_values(ascending=True)
b = ax.barh(outs.index,outs.values,color=NAVY)
for el,v in zip(b,outs.values): ax.text(v+2,el.get_y()+el.get_height()/2,str(v),va='center',fontsize=9)
ax.set_xlim(0,max(outs)*1.18);ax.set_xlabel('Số điểm nằm ngoài hàng rào IQR')
ax.set_title('Cảnh báo ngoại lệ trong các biến số',loc='left',fontweight='bold')
ax.grid(axis='x',alpha=.15)
save_chart('02_iqr_outliers')

bars('OverTime','03_overtime','Nghỉ việc theo tình trạng tăng ca', ['No','Yes'])
bars('BusinessTravel','04_business_travel','Nghỉ việc theo tần suất công tác',
     ['Non-Travel','Travel_Rarely','Travel_Frequently'])
bars('JobSatisfaction','05_job_satisfaction','Nghỉ việc theo mức hài lòng công việc',[1,2,3,4])
bars('TenureGroup','06_tenure','Nghỉ việc theo thâm niên tại doanh nghiệp',
     ['0 đến 1 năm','2 đến 4 năm','5 đến 9 năm','Từ 10 năm'])
bars('WorkLifeBalance','08_work_life','Nghỉ việc theo cân bằng công việc và đời sống',[1,2,3,4])
bars('IncomeQuintile','07_income_quintiles','Nghỉ việc theo nhóm thu nhập tháng',
     ['Q1 thấp nhất','Q2','Q3','Q4','Q5 cao nhất'])
bars('JobRole','09_job_role','Nghỉ việc theo vị trí công việc',
     group_summary('JobRole').sort_values('rate',ascending=False)['JobRole'].tolist(),horiz=True)
bars('PerformanceRating','10_performance_rating','Nghỉ việc theo mức đánh giá hiệu suất',[3,4])
bars('StockOptionLevel','11_stock_option','Nghỉ việc theo mức quyền chọn cổ phiếu',[0,1,2,3])
bars('OT_TenureGroup','12_overtime_tenure','Kết hợp tăng ca và thâm niên',
     ['Có tăng ca, 0 đến 1 năm','Có tăng ca, trên 1 năm',
      'Không tăng ca, 0 đến 1 năm','Không tăng ca, trên 1 năm'],horiz=True)

# Hinh 13: dung bieu do thanh he so Spearman, tranh goi nham la ma tran.
corr_vars=['Age','DistanceFromHome','JobSatisfaction','JobInvolvement',
           'MonthlyIncome','PerformanceRating','TotalWorkingYears','YearsAtCompany',
           'WorkLifeBalance','StockOptionLevel']
vn_names={'Age':'Tuổi','DistanceFromHome':'Khoảng cách',
          'JobSatisfaction':'Hài lòng công việc','JobInvolvement':'Gắn bó công việc',
          'MonthlyIncome':'Thu nhập tháng','PerformanceRating':'Hiệu suất',
          'TotalWorkingYears':'Kinh nghiệm','YearsAtCompany':'Thâm niên',
          'WorkLifeBalance':'Cân bằng công việc','StockOptionLevel':'Quyền chọn cổ phiếu'}
cs=df[corr_vars+['AttritionFlag']].corr(method='spearman')['AttritionFlag'].drop('AttritionFlag')
cs=cs.reindex(cs.abs().sort_values().index)
fig,ax=plt.subplots(figsize=(8.4,4.6))
bb=ax.barh([vn_names.get(c,c) for c in cs.index],cs.values,
           color=[COPPER if v>0 else NAVY for v in cs.values])
ax.axvline(0,color=GRAY,linewidth=.9)
for bar,value in zip(bb,cs.values):
    ax.text(value+(.008 if value>=0 else -.008),bar.get_y()+bar.get_height()/2,
            f'{value:+.2f}',ha='left' if value>=0 else 'right',va='center',fontsize=9)
ax.set_xlim(-.31,.18);ax.set_xlabel('Hệ số Spearman với nhãn nghỉ việc')
ax.set_title('Mối liên hệ của các đặc trưng với nghỉ việc',loc='left',fontweight='bold')
ax.grid(axis='x',alpha=.15)
save_chart('13_spearman_correlations')
multicollinearity={
 'MonthlyIncome_JobLevel':float(raw[['MonthlyIncome','JobLevel']].corr('spearman').iloc[0,1]),
 'YearsAtCompany_YearsInCurrentRole':float(raw[['YearsAtCompany','YearsInCurrentRole']].corr('spearman').iloc[0,1])}

# %% 05. Mo hinh minh hoa bo sung, dung danh gia ngoai mau
# Loai cot dinh danh, cot hang so va ba bien nhan khau nhay cam khoi mo hinh.
# Cac cot so duoc scaled, cac cot phan loai dung One Hot.
exclude = {'Attrition','AttritionFlag','EmployeeNumber','EmployeeCount',
           'Over18','StandardHours','Age','Gender','MaritalStatus'}
features = [c for c in raw.columns if c not in exclude]
X, y = raw[features].copy(), df['AttritionFlag'].copy()
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()
preprocess = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer',SimpleImputer(strategy='median')),
                      ('scaler',StandardScaler())]),num_cols),
    ('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                      ('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat_cols)
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,
                                                stratify=y,random_state=42)
models = {
    'Dự báo đa số': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': Pipeline([('prep',preprocess),
                                    ('model',LogisticRegression(max_iter=2500,
                                             class_weight='balanced', random_state=42))]),
    'Random Forest': Pipeline([('prep',preprocess),
                              ('model',RandomForestClassifier(n_estimators=250,
                                        max_depth=9,min_samples_leaf=5,
                                        class_weight='balanced',random_state=42,
                                        n_jobs=-1))])
}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores={}
for name,model in models.items():
    if name == 'Dự báo đa số': continue
    scores=cross_val_score(model,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1)
    cv_scores[name]={'mean':float(np.mean(scores)),'std':float(np.std(scores))}
print('ROC AUC trung binh 5 fold:', cv_scores)
selected=max(cv_scores,key=lambda n:cv_scores[n]['mean'])
model_eval={}
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    if hasattr(model,'predict_proba'):
        proba=model.predict_proba(X_test)[:,1]
    else: proba=pred
    model_eval[name] = {'accuracy':float(accuracy_score(y_test,pred)),
                       'precision':float(precision_score(y_test,pred,zero_division=0)),
                       'recall':float(recall_score(y_test,pred,zero_division=0)),
                       'f1':float(f1_score(y_test,pred,zero_division=0)),
                       'roc_auc':float(roc_auc_score(y_test,proba)),
                       'pr_auc':float(average_precision_score(y_test,proba)),
                       'confusion':confusion_matrix(y_test,pred).tolist()}
print('Bo kiem tra:',len(y_test),'so nghi viec:',int(y_test.sum()))
print('Danh gia mo hinh:',pd.DataFrame(model_eval).T.to_string())
print('Mo hinh chon theo CV tren training:', selected)

# Kiem tra do nhay doc lap: mo hinh baseline phia tren khong thay doi.
# 1. Thay thang do StandardScaler bang RobustScaler.
# 2. Chuyen hai bien co mau hinh khong don dieu sang nhom danh muc One Hot.
def sensitivity_check(label, alternative_preprocess):
    m=Pipeline([('prep',alternative_preprocess),
       ('model',LogisticRegression(max_iter=2500,class_weight='balanced',random_state=42))])
    c=cross_val_score(m,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1)
    m.fit(X_train,y_train)
    yp=m.predict(X_test); ps=m.predict_proba(X_test)[:,1]
    return {'model':label,'cv_roc_auc':float(np.mean(c)),
      'test_roc_auc':float(roc_auc_score(y_test,ps)),
      'test_pr_auc':float(average_precision_score(y_test,ps)),
      'test_accuracy':float(accuracy_score(y_test,yp)),
      'test_recall':float(recall_score(y_test,yp))}
alt_robust_pre=ColumnTransformer(transformers=[
  ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),
                   ('scaler',RobustScaler())]),num_cols),
  ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                   ('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat_cols)])
nonlinear=['WorkLifeBalance','StockOptionLevel']
alt_num=[c for c in num_cols if c not in nonlinear]
alt_cat=cat_cols+nonlinear
alt_nonlin_pre=ColumnTransformer(transformers=[
  ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),
                   ('scaler',RobustScaler())]),alt_num),
  ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                   ('onehot',OneHotEncoder(handle_unknown='ignore'))]),alt_cat)])
sensitivity=[sensitivity_check('RobustScaler',alt_robust_pre),
             sensitivity_check('RobustScaler + OneHot hai bien thu bac',alt_nonlin_pre)]
pd.DataFrame(sensitivity).to_csv(ROOT/'robustness_checks.csv',index=False,encoding='utf-8-sig')
print('Ket qua kiem tra do nhay:',sensitivity)


fig,ax=plt.subplots(figsize=(7.5,4.6))
for name,color in [('Logistic Regression',NAVY),('Random Forest',COPPER)]:
    RocCurveDisplay.from_estimator(models[name],X_test,y_test,ax=ax,
                                  name=name,
                                  color=color)
ax.plot([0,1],[0,1],color='gray',linestyle=':',label='Ngẫu nhiên')
ax.set_title('Đường cong ROC trên tập kiểm tra 30%',loc='left',fontweight='bold')
ax.set_xlabel('Tỷ lệ báo động nhầm');ax.set_ylabel('Tỷ lệ phát hiện đúng nghỉ việc')
ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=.15)
save_chart('14_roc_models')

fig,ax=plt.subplots(figsize=(5.6,4.3))
ConfusionMatrixDisplay.from_estimator(models[selected],X_test,y_test,ax=ax,
          labels=[0,1],display_labels=['Không nghỉ','Nghỉ'],cmap='Blues',colorbar=False)
ax.set_title(f'Ma trận nhầm lẫn của {selected}',loc='left',fontweight='bold')
ax.set_xlabel('Dự báo');ax.set_ylabel('Thực tế')
save_chart('15_confusion_matrix')

imp=permutation_importance(models[selected],X_test,y_test,n_repeats=7,
                            random_state=42,scoring='roc_auc',n_jobs=1)
order=np.argsort(imp.importances_mean)[::-1][:10]
fig,ax=plt.subplots(figsize=(8.2,4.4))
vals=imp.importances_mean[order][::-1]
labels=np.array(features)[order][::-1]
ax.barh(labels,vals,color=NAVY)
ax.axvline(0,color=GRAY,linewidth=.7)
ax.set_xlabel('Mức giảm ROC AUC khi hoán đổi một biến')
ax.set_title('Tầm quan trọng dự báo của các đặc trưng',loc='left',fontweight='bold')
ax.grid(axis='x',alpha=.15)
save_chart('16_permutation_importance')

# %% 06. Luu bang ket qua de kiem chung so lieu bao cao
summary_vars=['OverTime','BusinessTravel','JobSatisfaction','TenureGroup',
              'WorkLifeBalance','IncomeQuintile','JobRole','PerformanceRating',
              'StockOptionLevel','OT_TenureGroup']
summaries={name:[{str(k): (float(v) if isinstance(v,(np.floating,float)) else
                                       int(v) if isinstance(v,(np.integer,int)) else str(v))
                          for k,v in row.items()} for row in group_summary(name).to_dict('records')]
           for name in summary_vars}
result = {
    'source_file':SOURCE.name,
    'rows':int(len(raw)),'original_columns':int(raw.shape[1]),
    'dtypes':{c:str(raw[c].dtype) for c in raw.columns},
    'distincts':{c:int(raw[c].nunique()) for c in raw.columns},
    'missing_total':int(missing.sum()),'missing_by_column':missing.astype(int).to_dict(),
    'duplicate_rows':duplicates,'duplicate_employee_numbers':repeated_ids,
    'constant_columns':constant_cols,'logical_checks':logical_checks,
    'outliers':outliers,'outliers_all':all_iqr,'attrition_counts':raw.Attrition.value_counts().to_dict(),
    'attrition_rate':float(y.mean()),'summaries':summaries,
    'tests':tests,
    'income_left':{'mean':float(left_pay.mean()),'median':float(left_pay.median()),
                   'n':int(len(left_pay))},
    'income_stay':{'mean':float(stay_pay.mean()),'median':float(stay_pay.median()),
                   'n':int(len(stay_pay))},
    'model_features':features,'model_num_features':num_cols,
    'model_cat_features':cat_cols,'train_rows':len(X_train),
    'test_rows':len(X_test),'train_positive':int(y_train.sum()),
    'test_positive':int(y_test.sum()),
    'cv_scores':cv_scores,'model_eval':model_eval,'selected_model':selected,
    'robustness':sensitivity,
    'spearman_multicollinearity':multicollinearity,
    'monthlyincome_skew':float(raw.MonthlyIncome.skew()),
    'importance':{features[i]:float(imp.importances_mean[i]) for i in range(len(features))}
}
with open(ROOT / 'analysis_results.json','w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
pd.DataFrame(tests).T.to_csv(ROOT/'hypothesis_tests.csv',encoding='utf-8-sig')
pd.DataFrame(model_eval).T.to_csv(ROOT/'model_metrics.csv',encoding='utf-8-sig')
print('Da luu',len(list(OUTPUT.glob('*.png'))),'bieu do va cac bang ket qua')
