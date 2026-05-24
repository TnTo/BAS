# %%
import pandas
import seaborn

# %%
df = pandas.concat([pandas.read_pickle(f"0{m}_res.pkl") for m in range(6,10)]).pivot(index=['Model', 'seed', 't'], columns='Var', values='Val').reset_index()
# %%
seaborn.lineplot(x="t", y="i", hue="Model", data=df)
# %%
seaborn.lineplot(x="t", y="i", hue="Model", data=df[df.t>10])
# %%
seaborn.lineplot(x="t", y="u", hue="Model", data=df)
# %%
seaborn.lineplot(x="t", y="u", hue="Model", data=df[df.t>25])
#%%
seaborn.lineplot(x="t", y="GDP", hue="Model", data=df)
#%%
seaborn.lineplot(x="t", y="MGini", hue="Model", data=df)
#%%
seaborn.lineplot(x="t", y="WGini", hue="Model", data=df)
#%%
seaborn.lineplot(x="t", y="WGini", hue="Model", data=df[df.t>25])
#%%
seaborn.lineplot(x="t", y="PubExpShare", hue="Model", data=df)
# %%
