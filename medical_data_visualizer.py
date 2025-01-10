import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)


# 4
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat.rename(columns={'cardio': 'cardiovascular'}, inplace=True)
    df_cat['sum'] = 1
    df_cat = df_cat.groupby(['cardiovascular', 'variable', 'value'], as_index=False).count()
    fig = sns.catplot(x='variable', y='sum', data=df_cat, hue='value', kind='bar', col='cardiovascular').fig
    
    fig.savefig('catplot.png')
    return fig




# 10
def draw_heat_map():
    # 11

    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (
                df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 14
    n = len(corr.columns)
    plt.figure(figsize=(n, n))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
    )

    plt.savefig('heatmap.png')
    return 0
