import pandas as pd, numpy as np, seaborn as sns, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight","axes.titleweight":"bold"})
df = pd.read_csv("crop.csv")
num = ["N","P","K","temperature","humidity","ph","rainfall"]
units = {"N":"N (kg/ha)","P":"P (kg/ha)","K":"K (kg/ha)","temperature":"Temperature (°C)","humidity":"Relative humidity (%)","ph":"Soil pH","rainfall":"Rainfall (mm)"}

# Fig1 class balance
fig,ax=plt.subplots(figsize=(8,5))
vc=df.label.value_counts().sort_index()
sns.barplot(x=vc.values,y=vc.index,color="#4C9F70",ax=ax)
ax.set(title="Records per crop (balanced classes)",xlabel="Number of records",ylabel="Crop")
fig.savefig("figs/fig1.png"); plt.close()

# Fig2 histograms
fig,axes=plt.subplots(2,4,figsize=(14,6.5)); axes=axes.ravel()
for a,c in zip(axes,num):
    sns.histplot(df[c],kde=True,bins=30,color="#2E7D5B",ax=a)
    a.set(title=f"{c}  (skew={df[c].skew():.2f})",xlabel=units[c],ylabel="Count")
axes[-1].axis("off")
fig.suptitle("Distribution of numeric features with KDE",fontweight="bold",y=1.01)
fig.tight_layout(); fig.savefig("figs/fig2.png"); plt.close()

# Fig3 K log transform
fig,axes=plt.subplots(1,2,figsize=(10,4))
sns.histplot(df.K,bins=30,kde=True,color="#C0392B",ax=axes[0]); axes[0].set(title=f"Original K (skew={df.K.skew():.2f})",xlabel="K (kg/ha)",ylabel="Count")
lk=np.log1p(df.K)
sns.histplot(lk,bins=30,kde=True,color="#2E7D5B",ax=axes[1]); axes[1].set(title=f"log(1+K) (skew={lk.skew():.2f})",xlabel="log(1 + K)",ylabel="Count")
fig.suptitle("Effect of log transformation on potassium",fontweight="bold",y=1.03)
fig.tight_layout(); fig.savefig("figs/fig3.png"); plt.close()

# Fig4 outlier boxplots (z-scored)
z=(df[num]-df[num].mean())/df[num].std()
fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=z,orient="h",palette="Greens",ax=ax)
ax.set_yticks(range(len(num))); ax.set_yticklabels([units[c] for c in num]); ax.axvline(0,color="grey",ls="--",lw=1)
ax.set(title="Standardised box plots: outlier check",xlabel="Z-score (standard deviations from mean)")
fig.savefig("figs/fig4.png"); plt.close()

# Fig5 correlation heatmap
fig,ax=plt.subplots(figsize=(7.5,6))
corr=df[num].corr()
sns.heatmap(corr,annot=True,fmt=".2f",cmap="RdBu_r",vmin=-1,vmax=1,mask=np.triu(np.ones_like(corr,bool),1),square=True,cbar_kws={"label":"Pearson r"},ax=ax)
ax.set_title("Correlation matrix of soil and climate features")
fig.savefig("figs/fig5.png"); plt.close()

# Fig6 rainfall by crop
order=df.groupby("label").rainfall.median().sort_values(ascending=False).index
fig,ax=plt.subplots(figsize=(10,6))
sns.boxplot(data=df,y="label",x="rainfall",order=order,color="#7FB8D9",fliersize=2,ax=ax)
ax.set(title="Rainfall requirement by crop (sorted by median)",xlabel="Rainfall (mm)",ylabel="Crop")
fig.savefig("figs/fig6.png"); plt.close()

# Fig7 humidity by crop
order=df.groupby("label").humidity.median().sort_values(ascending=False).index
fig,ax=plt.subplots(figsize=(10,6))
sns.boxplot(data=df,y="label",x="humidity",order=order,color="#F2B880",fliersize=2,ax=ax)
ax.set(title="Relative humidity by crop (sorted by median)",xlabel="Relative humidity (%)",ylabel="Crop")
fig.savefig("figs/fig7.png"); plt.close()

# Fig8 P vs K scatter
hi=df.label.isin(["apple","grapes"])
fig,ax=plt.subplots(figsize=(8,6))
sns.scatterplot(data=df[~hi],x="P",y="K",color="#9E9E9E",s=18,alpha=.6,label="Other 20 crops",ax=ax)
sns.scatterplot(data=df[hi],x="P",y="K",hue="label",palette={"apple":"#C0392B","grapes":"#6C3483"},s=30,ax=ax)
ax.annotate("Apple and grapes form a\nhigh-P, high-K cluster",xy=(133,200),xytext=(60,120),arrowprops=dict(arrowstyle="->"))
ax.set(title="Phosphorus vs potassium: a cluster drives the correlation",xlabel="P (kg/ha)",ylabel="K (kg/ha)")
ax.legend(title="Crop",loc="lower right")
fig.savefig("figs/fig8.png"); plt.close()

# Fig9 temp vs rainfall selected crops
sel=["rice","coconut","jute","muskmelon","chickpea","papaya","apple","lentil"]
fig,ax=plt.subplots(figsize=(9,6))
sns.scatterplot(data=df[df.label.isin(sel)],x="temperature",y="rainfall",hue="label",style="label",s=45,ax=ax)
ax.set(title="Temperature vs rainfall for eight selected crops",xlabel="Temperature (°C)",ylabel="Rainfall (mm)")
ax.legend(title="Crop",bbox_to_anchor=(1.02,1),loc="upper left")
fig.savefig("figs/fig9.png"); plt.close()

# Fig10 crop profile heatmap (z of crop means)
prof=df.groupby("label")[num].mean()
zp=(prof-prof.mean())/prof.std()
fig,ax=plt.subplots(figsize=(9,8))
sns.heatmap(zp,cmap="RdBu_r",center=0,annot=prof.round(0).astype(int),fmt="d",linewidths=.4,cbar_kws={"label":"Z-score of crop mean"},ax=ax)
ax.set(title="Crop requirement profiles (colour = z-score, number = mean)",xlabel="Feature",ylabel="Crop")
fig.savefig("figs/fig10.png"); plt.close()

# Fig11 ANOVA F
from scipy import stats
F=pd.Series({c:stats.f_oneway(*[g[c].values for _,g in df.groupby("label")]).statistic for c in num}).sort_values()
fig,ax=plt.subplots(figsize=(8,4.5))
ax.barh([units[c] for c in F.index],F.values,color="#4C9F70"); ax.set_xscale("log")
for i,v in enumerate(F.values): ax.text(v*1.1,i,f"{v:,.0f}",va="center")
ax.set(title="How well each feature separates crops (ANOVA F, log scale)",xlabel="F-statistic (higher = better separation)")
ax.set_xlim(right=F.max()*8)
fig.savefig("figs/fig11.png"); plt.close()

# Fig12 pairplot subset
sub=df[df.label.isin(["rice","maize","chickpea","coconut","banana","grapes"])]
g=sns.pairplot(sub,vars=["N","K","humidity","rainfall"],hue="label",corner=True,plot_kws=dict(s=14,alpha=.7),height=2.2)
g.fig.suptitle("Pair plot for six contrasting crops",fontweight="bold",y=1.02)
g.savefig("figs/fig12.png"); plt.close()

# numbers for text
print("PK corr all",df.P.corr(df.K).round(2),"excluding apple/grapes",df[~hi].P.corr(df[~hi].K).round(2))
df["NPK_total"]=df.N+df.P+df.K
print(df.groupby("label").NPK_total.mean().sort_values().round(0).iloc[[0,1,-2,-1]])
cv=(df.groupby("label")[num].std()/df.groupby("label")[num].mean()).round(2)
print(cv.mean().round(2))
print("ph range by crop"); print(df.groupby("label").ph.agg(["min","max"]).round(1).sort_values("max").iloc[[0,1,-2,-1]])
df["rain_band"]=pd.cut(df.rainfall,[0,50,100,150,300],labels=["<50","50-100","100-150",">150"])
print(df.rain_band.value_counts().sort_index())
print(df.groupby("rain_band",observed=True).label.agg(lambda s:s.value_counts().head(3).to_dict()))
print("temp>=35:",(df.temperature>=35).sum(), df[df.temperature>=35].label.value_counts().to_dict())
print("humidity<30:",(df.humidity<30).sum(), df[df.humidity<30].label.value_counts().to_dict())
print(df[hi].K.describe().round(0)); print((df.K>df.K.quantile(.75)+1.5*(df.K.quantile(.75)-df.K.quantile(.25))).sum(), df[df.K>80].label.value_counts().to_dict())
print(df.groupby('label').rainfall.median().sort_values().round(0).iloc[[0,1,-2,-1]])
print(df.groupby('label').humidity.median().sort_values().round(0).iloc[[0,1,2,-1]])
print("corr after log K, P:",np.log1p(df.K).corr(df.P).round(2))
