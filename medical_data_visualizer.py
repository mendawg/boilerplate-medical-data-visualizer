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
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                     var_name='Parameter', value_name='Value')
    df_cat.rename(columns={'cardio': 'cardiovascular'}, inplace=True)

    # 6
    df_cat.groupby(['cardiovascular', 'Parameter', 'Value']).size().reset_index(name='Count')

    # 7
    t = sns.catplot(data=df_cat, x='Value', hue='cardiovascular', col='Parameter', kind='count', height=6, aspect=1)
    # 8
    fig = t.fig

    # 9
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
    mask = ~np.triu(np.ones_like(corr, dtype=bool))
    # 14
    n = len(corr.columns)
    plt.figure(figsize=(n,n))

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


draw_heat_map()
