import pandas as pd
import numpy as np 
from data_processing import data,X

#print(np.quantile(data['I_mean'],[0.0,0.25,0.5,0.75,1.0]))
table = {
    
}


table.setdefault("Feature",[])
table.setdefault("Min",[])
table.setdefault("1st Quartile",[])
table.setdefault("Median",[])
table.setdefault("Mean",[])
table.setdefault("3rd Quartile",[])
table.setdefault("Max",[])

for col in data.columns:
    table["Feature"].append(col)
    table["Min"].append(data[col].min())
    table["1st Quartile"].append(np.quantile(data[col],[0.25])[0])
    table["Median"].append(np.quantile(data[col],[0.50])[0])
    table["Mean"].append(data[col].mean())
    table["3rd Quartile"].append(np.quantile(data[col],[0.75])[0])
    table["Max"].append(data[col].max())
table = pd.DataFrame(table)
#print(table)
    