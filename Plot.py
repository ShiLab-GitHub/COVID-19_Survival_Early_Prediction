from Data_Analysis import *
import seaborn as sns
from numpy import mean

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# filePath = "./data/death data - s&f.xlsx"
filePath = "./data/recover data - box.xlsx"

data = pd.read_excel(filePath, index_col=None, header=0)
sig = np.array(pd.read_csv('./data/shap/01/fea.txt', index_col=None, header=None))
sig = sig.squeeze()
print(sig.shape)

x_data = data[sig]
print(np.shape(x_data))

y_data = data['Infection_status_code']
data = pd.concat([y_data, x_data], axis=1, join='outer')

title = ['NG,NG-Dimethyl-L-arginine', '4-Coumaryl alcohol', 'Pyridoxamine', 'N1,N12-Diacetylspermine', 'Coniferyl alcohol', '1-Phosphatidyl-1D-myo-inositol 3-phosphate', 'sn-Glycero-3-phosphocholine']
i = 0
plt.figure(dpi=300)
for ssig in sig:
    # sns.violinplot(x="Infection_status_code", y=sig, data=data12, palette="Reds", split=False, inner='quartile') #, linewidth=4
    sns.boxplot(x="Infection_status_code", y=ssig, data=data, palette="Reds") #, notch=True

    Means = data.groupby('Infection_status_code')[ssig].mean()
    plt.scatter(x=range(len(Means)), y=Means, c="b")
    for a, b in zip(range(len(Means)), Means):
        plt.text(a, b + 0.005, '%.4f' % b, ha='center', va='bottom', fontsize=14)

    # names = ['Survival', 'Death']
    names = ['COVID-19', 'Recovered']

    x = [i for i in range(2)]
    plt.xticks(x, names, size=14)
    plt.yticks(size=14)
    plt.title(title[i]+'\nm/z  '+str(ssig), fontsize=16, fontproperties='Times New Roman', fontweight='medium')
    plt.xlabel("")
    plt.ylabel("Normalized concentration", fontsize=14)

    # plt.savefig('./data/shap/01/plot/alive&death/82-{}.png'.format(i))
    plt.savefig('./data/shap/01/plot/recover/82-{}.png'.format(i))
    plt.cla()
    i += 1
