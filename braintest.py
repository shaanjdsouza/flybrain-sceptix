import pandas as pd
g = pd.read_csv("banc_subset/all_groups.csv")
mn = g[g.group == "MN_leg"]
print(mn.groupby(["nerve", "soma_side"]).size().to_string())
print(mn.sub_class.value_counts().head(30).to_string())