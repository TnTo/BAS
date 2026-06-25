# %%
import pandas
import seaborn
from matplotlib.pyplot import savefig

# %%
df = (
    pandas.concat([pandas.read_pickle(f"{m}_res.pkl") for m in ['06','07','08','09','10']])
    .pivot(index=["Model", "seed", "t"], columns="Var", values="Val")
    .reset_index()
)
# %%
seaborn.lineplot(x="t", y="i", hue="Model", data=df)
# %%
seaborn.lineplot(x="t", y="i", hue="Model", data=df[df.t > 10])
savefig("plot/pol_i.pdf")
# %%
seaborn.lineplot(x="t", y="u", hue="Model", data=df)
# %%
seaborn.lineplot(x="t", y="u", hue="Model", data=df[df.t > 25])
savefig("plot/pol_u.pdf")
# %%
seaborn.lineplot(x="t", y="GDP", hue="Model", data=df)
savefig("plot/pol_gdp.pdf")
# %%
seaborn.lineplot(x="t", y="MGini", hue="Model", data=df)
savefig("plot/pol_mgini.pdf")
# %%
seaborn.lineplot(x="t", y="WGini", hue="Model", data=df)
# %%
seaborn.lineplot(x="t", y="WGini", hue="Model", data=df[df.t > 25])
savefig("plot/pol_wgini.pdf")
# %%
seaborn.lineplot(x="t", y="PubExpShare", hue="Model", data=df)
savefig("plot/pol_g.pdf")
# %%
seaborn.lineplot(x="t", y="GvtDebt", hue="Model", data=df[df.t>50])
# %%
seaborn.lineplot(x="t", y="GvtDeficit", hue="Model", data=df[df.t>50])
# %%
