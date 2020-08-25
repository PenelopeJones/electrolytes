import numpy as np
import pandas as pd

import pdb


def main():
    # Track where the raw data is located
    sim_param = '30nq'
    type = 'type_0'
    num = 0
    
    no_sol = 'no_sol'
    if no_sol == 'no_sol':
        symbols_map = {'0.0': 'Cl', '1.0': 'Na'}
    else:
        symbols_map = {'0.0': 'Cl', '1.0': 'Na', '2.0': 'H'}

    ptd = '../data/3d_values/' + sim_param + '/'
    pts = '..data/xyz/' + sim_param + '/'
    
    """
        if type == 'type_1':
        num = 500
    else:
        num = 120
    """
    

    df = pd.read_csv(ptd + 'config_' + str(num), sep = '\s+', skiprows = 24,
                     usecols = [2,3,4,5], nrows = 18000, header = None,
                     lineterminator = '}')
    
    pdb.set_trace()

    df['symbols'] = df[5].astype(str).map(symbols_map)

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-2]
    df = df[cols]
    df = df[df['symbols'].notna()]

    N = df.shape[0]
    print(N)

    pdb.set_trace()

    df.to_csv(sim_param + '_' + type + '_' + no_sol + '.csv', index=False)

    pdb.set_trace()






if __name__ == "__main__":
    main()
