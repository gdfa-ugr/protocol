# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from datetime import datetime
from matplotlib import dates
import pandas as pd
from fdist.copula import Copula
from ttide.t_tide import t_predic
import scipy.stats as st
from fdist import ajuste_simple as ajs
from fdist import lognorm_genpar2, lognorm2, weib_genpar2, norm_genpar2, normenr, normtrun
from scipy.io import savemat

def simula_var(par, lsim, type_):
    """Funcion que genera las series temporales de probabilidad

    Args:
        par: Parametros de ajuste de VAR
        lsim: Longitud de la serie a simular
        type_: Tipo de distribucion U (multivariada normal)

    Returns:
        Una serie multivariada

    """

    dim = par['dim']
    ord_ = par['id-bic']
    zsim = np.zeros([dim, lsim])
    if type_ == 'normal':
        y = np.random.multivariate_normal(np.zeros(dim), par['Q'], lsim).T
    """
    elif TYPE == 'paretotails':
        U2 = PARU
        OBJ = list()
        for I in range(0,dim):
            OBJ[I] = paretotails(PAR.U[:,I],0.05,0.95)
            U2[:,I] = norminv(cdf(OBJ[I],PAR.U[:,I]))

        Q2 = np.cov(U2)
        Y2 = np.random.multivariate_normal(np.zeros([dim,1]),Q2,LSIM)
        Y = Y2
        for I in range(0,dim):
            Y[:,I] = icdf(OBJ[I],normcdf(Y2[:,I]))

    """

    zsim[:, 0:ord_] = y[:, 0:ord_]
    for i in range(ord_, lsim):
        z = np.fliplr(zsim[:, i-ord_:i])
        z1 = np.vstack((1, np.reshape(z.T, (ord_*dim, 1))))
        zsim[:, i] = np.dot(par['B'], z1).T + y[:, i]

    return zsim.T


def simulacion(anocomienzo, duracion, par, mod, nnorm, f_pro, vars_, df_dt, tide_param, *familia):
    """Funcion que simula las curvas de estados climaticos que caracterizan el oleaje y el viento.

    Args:
        - anocomienzo: Ano inicial de la simulacion
        - duracion: Numero de anos que se desean simular
        - par: lista con los parametros de los ajustes de las variables
        - mod: lista con los modos de los ajustes de las variables
        - nnorm: numero de normales del ajuste direccional
        - f_mix: lista con los modelos de probabilidad de cada variable
        - f_pro: lista con los funciones de probabilidad de los ajustes
        - vars_: lista con los nombres de las variables sobre las que usar la simulación
        - df_dt: dataframe con los parametros del modelo VAR
        - tide_param: parametros mareales

            * ``nameu``: nombre de la constituyentes principales
            * ``fu``: frecuencias de las constituyentes
            * ``tideconout``: amplitud de las constituyentes
            * ``mm_min``: valor mínimo del residuo meteorológico registrado

        - semilla: Semilla que se pasa a la funcion rng() para inicializar el analisis. Su valor puede ser cualquier
                 numero entero positivo para forzar un comportamiento deterministicos, o 'shuffle' para hacerlo no
                 deterministico. Si no se especifica, por defecto realiza una simulacion no deterministica
        - familia[opc]: familia de la copula que se usara para el ajuste univariado (clayton, gumbel, frank)

    Returns:
        Un dataframe con los datos simulados

    """

    ano2 = anocomienzo + duracion - 1
    date1 = datetime(anocomienzo, 1, 1, 0, 0, 0)
    date2 = datetime(ano2, 12, 31, 21, 0, 0)
    df_sim = pd.DataFrame(index=pd.date_range(start=date1, end=date2, freq='3H'))

    n_aux = df_sim.index.map(dates.date2num)
    n_anio = map(datetime, df_sim.index.year, np.ones(len(df_sim), dtype=int), np.ones(len(df_sim), dtype=int))
    n_anio = np.array(map(dates.date2num, n_anio))
    n_anio1 = map(datetime, df_sim.index.year + 1, np.ones(len(df_sim), dtype=int), np.ones(len(df_sim), dtype=int))
    n_anio1 = np.array(map(dates.date2num, n_anio1))
    nsim = (n_aux - n_anio)/(n_anio1 - n_anio) + (df_sim.index.year - anocomienzo)
#    nsim = ((df_sim.index.dayofyear + df_sim.index.hour/24. - 1)/pd.to_datetime(
#            {'year': df_sim.index.year, 'month': 12, 'day': 31, 'hour': 23}).dt.dayofyear).values + (
#        df_sim.index.year - anocomienzo)

    lsim = np.size(nsim)
    if not hasattr(familia, 'attr_name'):
        zsim = simula_var(df_dt, lsim, 'normal')
    else:
        cop = Copula()
        cop.theta, cop.tau, cop.pr, cop.sr = df_dt[1:]
        zsim = np.zeros(lsim)
        zsim[0] = np.random.rand(1)
        for i in range(1, lsim):
            zsim[i] = cop.generate_cond(zsim[i-1])

    for i, j in enumerate(vars_):
        if hasattr(f_pro[i], 'name'):
            df_sim[j] = ajs.inv(nsim, st.norm.cdf(zsim[:, i]), par[i], mod[i], f_pro[i])
        elif (('normenr' in f_pro[i].__name__) | ('normtrun' in f_pro[i].__name__)):
            df_sim[j], _ = f_pro[i].inv(nsim, st.norm.cdf(zsim[:, i]), par[i], mod[i], nnorm[i])
        else:
            df_sim[j] = f_pro[i].inv(nsim, st.norm.cdf(zsim[:, i]), par[i], mod[i])

    if 'mme' in vars_:
        nameu, fu, tideconout, mm_min, mm_mean = tide_param
        nsim = np.array(df_sim.index.map(dates.date2num), dtype=np.float64)
        df_sim['mas'] = t_predic(nsim, nameu, fu, tideconout, lat=36) + mm_mean
        df_sim['mme'] += mm_min

    if 'dh' in vars_ or 'DirM' in vars_:
        df_sim.dh = np.remainder(df_sim['DirM'], 360)

    if 'dv' in vars_ or 'DirV' in vars_:
        df_sim.dv = np.remainder(df_sim['DirM'], 360)

    return df_sim


def serie_simulaciones(nsim, semilla, anocomienzo, duracion, par, mod, nnorm, f_pro, vars_, df_dt, tide_param, name, savepy=True, savemat=False, *familia):

    if isinstance(semilla, int):
        np.random.seed(semilla)

    for ii in range(1, nsim+1, 1):
        sim = simulacion(anocomienzo, duracion, par, mod, nnorm, f_pro, vars_, df_dt, tide_param, *familia)

        if savepy:
            sim.to_pickle(name + '__' + '%02d' % (ii,) + '.pkl')

        if savemat:
            sim_dict = {col_name: sim[col_name].values for col_name in sim.columns.values}
            sim_dict['fecha'] = sim.index.map(dates.date2num) + 366
            savemat(name + '.mat', {'struct': sim_dict})
