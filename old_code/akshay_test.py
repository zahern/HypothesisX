import numpy as np
from harmony import *
from siman import *
from threshold import *
from latent_class_mixed_model import LatentClassMixedModel
from latent_class_model import LatentClassModel
from mixed_logit import MixedLogit
from multinomial_logit import MultinomialLogit
import pandas as pd
from scipy import stats

##df_wide = pd.read_csv('dataMaaSInd_wide.csv')
##df_wide =df_wide[df_wide['CHOICE'] !=0]
##
##from xlogit.utils import wide_to_long
##df = wide_to_long(df_wide, id_col='indID', alt_list=[1, 2, 3, 4],
##                       varying=['pref', 'purchase', 'useWork', 'useFamily', 'useMaintenance', 'useSocial',
##                                'maasLocalPT-1', 'maasLDPT-1', 'maasTaxi-1', 'maasCarRental-1', 'maasCarshare-1', 'maasRideshare-1', 'maasBikeshare-1', 'maasTktInt-1', 'maasBkInt-1', 'maasRTInf-1', 'maasPers-1', 'maasPaymentType-1', 'maasCost-1',
##                                'maasLocalPT-2', 'maasLDPT-2', 'maasTaxi-2', 'maasCarRental-2', 'maasCarshare-2', 'maasRideshare-2', 'maasBikeshare-2', 'maasTktInt-2', 'maasBkInt-2', 'maasRTInf-2', 'maasPers-2', 'maasPaymentType-2', 'maasCost-2'], alt_name='alt', sep= "_")
##
##df = wide_to_long(df, id_col='indID', alt_list=[1, 2],
##                  varying=['maasLocalPT', 'maasLDPT', 'maasTaxi', 'maasCarRental', 'maasCarshare', 'maasRideshare', 'maasBikeshare', 'maasTktInt', 'maasBkInt', 'maasRTInf', 'maasPers', 'maasPaymentType', 'maasCost'], alt_name='alt2', sep= "-")
##
##df = df.fillna(0)
##
##df['CHOICE'] = 1 * (df['pref'] == df['alt2'] )
##
##df.to_csv("akshay_long.csv", index=False)

df =pd.read_csv('akshay_long_true.csv')


model = LatentClassModel()
varnames=['InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male', 'Children', 'Income', 'NDI',
          'LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
          'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
          'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'
          ]

X = df[varnames].values
y = df['CHOICE'].values
member_params_spec = np.array([['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                               ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                               ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male', 'Children', 'Income', 'NDI'],
                               ['_inter', 'InnerCity', 'InnerRegional', 'Under30', 'Over65', 'College', 'FullTime', 'PartTime', 'Male', 'Children', 'Income', 'NDI']],
                              dtype='object')

class_params_spec = np.array([['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
                               'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                               'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
                              ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
                               'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                               'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
                              ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
                               'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                               'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
                              ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
                               'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                               'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers'],
                              ['LocalPTPayG', 'LDPTPayG', 'TaxiPayG', 'CarRentalPayG', 'CarsharePayG', 'RidesharePayG', 'BikesharePayG',
                               'LocalPTUnl', 'LDPTUnl', 'TaxiUnl', 'CarRentalUnl', 'CarshareUnl', 'RideshareUnl', 'BikeshareUnl',
                               'Cost', 'TktInt', 'BkInt', 'RTInf', 'Pers']],
                              dtype='object')


init_class_thetas = np.array([-1.321318, -0.254239, -0.137624, -9.159877, 0.009594, 1.189211, -0.084255, 0.437849, 0.222736, -2.338727, -0.220732, 0.206103,
                              0.293479, 0.17829, -0.293836, -0.499868, -0.336, 0.588949, 0.0357, 0.393709, -0.215125, -0.28694, -0.264146, -0.871409,
                              -1.160788, 0.752398, -0.054771, 0.554518, -0.559022, 0.633359, -0.150176, 0.020715, -0.23028, 0.185878, -0.219888, -1.531753,
                              -0.833134, -0.168312, -2.27768, 1.136705, 0.093996, 1.672507, 1.29167, 1.49679, 0.423603, 0.249344, -0.832107, -2.778636])


init_class_betas = [np.array([0.441269, 0.448334, 0.288787, 0.35502, 0.216816, 0.198564, 0.069477,
                              0.346543, 0.233089, 0.323059, 0.333928, 0.149546, 0.124614, 0.0443181,
                              -0.00741137, 0.036144, -0.00298227, 0.140595, 0.046312]), #Class 1
                    np.array([0.801542, 0.483616, 0.546757, 0.498264, 0.206961, 0.367382, 0.00124702,
                              0.587733, 0.398037, 0.5319, 0.369294, 0.246564, -0.100532, -0.141248,
                              -0.019849, 0.038627, -0.104714, 0.173183, 0.0905047]), #Class 2
                    np.array([1.28245, 0.704765, 0.8016, 0.145479, 0.340825, 0.554092, -0.0942558,
                              12.6054, 83.2791, 27.7743, -14.1763, 26.7106, 21.6308, -2.87297,
                              -32.6663, 0.528885, 0.375195, 0.367734, 0.343927]), #Class 3
                    np.array([1.18916, 0.562234, 0.58024, -0.00850272, 0.122827, 0.619118, 0.0330975,
                              0.970455, 0.24954, 0.698946, 0.172871, 0.64793, -0.395843, 0.00472563,
                              -0.425557, 0.157351, 0.0453663, 0.194574, 0.0677801]), #Class 4
                    np.array([0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0])] #Class 5
                
bounds_thetas = ((-1.321318, -1.321318), (-0.254239, -0.254239), (-0.137624, -0.137624), (-9.159877, -9.159877), (0.009594, 0.009594), (1.189211, 1.189211), (-0.084255, -0.084255), (0.437849, 0.437849), (0.222736, 0.222736), (-2.338727,-2.338727), (-0.220732, -0.220732), (0.206103,0.206103),
                 (0.293479, 0.293479), (0.17829, 0.17829), (-0.293836, -0.293836), (-0.499868, -0.499868), (-0.336, -0.336), (0.588949, 0.588949), (0.0357, 0.0357), (0.393709, 0.393709), (-0.215125, -0.215125), (-0.28694,-0.28694), (-0.264146, -0.264146), (-0.871409,-0.871409),
                 (-1.160788, -1.160788), (0.752398, 0.752398), (-0.054771, -0.054771), (0.554518, 0.554518), (-0.559022, -0.559022), (0.633359, 0.633359), ( -0.150176,  -0.150176), (0.020715, 0.020715), (-0.23028, -0.23028), (0.185878,0.185878), (-0.219888, -0.219888), (-1.531753, -1.531753),
                 (-0.833134, -0.833134), (-0.168312, -0.168312), (-2.27768, -2.27768), (1.136705, 1.136705), (0.093996, 0.093996), (1.672507, 1.672507), (1.29167, 1.29167), (1.49679, 1.49679), (0.423603, 0.423603), (0.249344, 0.249344), (-0.832107, -0.832107), (-2.778636, -2.778636))
bounds_betas = [((-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) ,),
                ((-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) ,),
                ((-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) ,),
                ((-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) ,),
                ((-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100),
                 (-100, 100), (-100, 100) , (-100, 100) , (-100, 100) , (-100, 100) ,),]



model.setup(X, y, ids=df['CHID'].values, panels=df['indID'].values,
          varnames=varnames,
          num_classes=5,
          class_params_spec=class_params_spec,
          member_params_spec=member_params_spec,
            init_class_betas=init_class_betas,
            init_class_thetas = init_class_thetas,
          alts=[1,2],
          ftol_lccm=1e-4,
          gtol=1e-5,
          #verbose = 2
          )
model.reassign_penalty(0.05)
model.fit()
model.summarise()

