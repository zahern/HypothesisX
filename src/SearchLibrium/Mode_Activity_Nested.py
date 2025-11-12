
from SearchLibrium.multinomial_nested import NestedLogit
import pandas as pd

df=pd.read_csv("C:/Users/N11931396/OneDrive - Queensland University of Technology/Documents/QUT/Papers/Nested Logit/DATA_SYNTHE_ASC.csv")

# ALL VARIABLES (All levels)

varnames_all = [
    # Alternative_level
    "var_ICT_1","var_ICT_2","var_ICT_3","var_ICT_4",
    "var_NICT_1","var_NICT_2","var_NICT_3","var_NICT_4",
    "var_NA_1","var_NA_2","var_NA_3","var_NA_4",
    # Nest_level
    "var_NMT_1","var_NMT_2","var_NMT_3","var_NMT_4","var_NMT_5",
    "var_IPT_1","var_IPT_2","var_IPT_3","var_IPT_4","var_IPT_5",
    "var_PV_1","var_PV_2","var_PV_3","var_PV_4","var_PV_5",
    "var_BUS_1","var_BUS_2","var_BUS_3","var_BUS_4","var_BUS_5"
    
]

# SIGNIFICANT VARIABLES (Variables 4 and 5 are Non-Significant at the Nest-level and variable 4 are Non-Significant at the alternative-level, then are dropped)

varnames = [
    # Alternative_level
    "var_ICT_1","var_ICT_2","var_ICT_3",
    "var_NICT_1","var_NICT_2","var_NICT_3",
    "var_NA_1","var_NA_2","var_NA_3",
    # Nest_level
    "var_NMT_1","var_NMT_2","var_NMT_3",
    "var_IPT_1","var_IPT_2","var_IPT_3",
    "var_PV_1","var_PV_2","var_PV_3",
    "var_BUS_1","var_BUS_2","var_BUS_3"      
]

# ALL VARIABLES (Nest-level)
varnest_all = [
    # Nest_level
    "var_NMT_1","var_NMT_2","var_NMT_3","var_NMT_4","var_NMT_5",
    "var_IPT_1","var_IPT_2","var_IPT_3","var_IPT_4","var_IPT_5",
    "var_PV_1","var_PV_2","var_PV_3","var_PV_4","var_PV_5",
    "var_BUS_1","var_BUS_2","var_BUS_3","var_BUS_4","var_BUS_5"   
]

# SIGNIFICANT VARIABLES (Nest-level)
varnest = [
    # Nest_level
    "var_NMT_1","var_NMT_2","var_NMT_3",
    "var_IPT_1","var_IPT_2","var_IPT_3",
    "var_PV_1","var_PV_2","var_PV_3",
    "var_BUS_1","var_BUS_2","var_BUS_3"   
]

nests = {
    "NMT": [0,1,2],
    "IPT": [3,4,5],
    "PV": [6,7,8],
    "BUS": [9,10,11]
}
 
lambdas = {
    "NMT": 1,    
    "IPT": 1,
    "PV": 1,    
    "BUS": 1        
}


model = NestedLogit()
model.setup(
    X=df[varnames],
    X_nest=df[varnest],
    y=df['choice'],    
    isvars=[],
    varnames=varnames,    
    fit_intercept=True,
    alts=df['alternative'],
    ids=df['individual'],
    avail=None,
    nests=nests,
    lambdas=lambdas
)

model.fit()
model.summarise()