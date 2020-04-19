#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)
from future.utils import bytes_to_native_str as nb  # to specify str type on both Python 2/3
# TODO cambiar nb por n

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import numpy as np
import pandas as pd
import os
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import dates
import time

from tqdm import tqdm

from climate import tests
from metoceandataframe.metoceandataframe import MetOceanDF
from preprocessing import missing_values
from climate.stats.forecasting import theoretical_fit
from climate.stats import empirical_distributions

from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo import analisis, simulacion
from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo.fdist import ajuste_simple, \
    normenr, normtrun, lognorm_genpar2, lognorm2, weib_genpar2, norm_genpar2
from climate.third_party.simulacion_clima_cobos.clima_maritimo.graficas import plot_analisis
from climate.stats import extremal, empirical_distributions, fitting


def test_onshore_simulation():
    # Modules activation and deactivation
    climate_analysis = False
    cdf_pdf_representation = False
    temporal_dependency = False
    climatic_simulation = True

    #%% Input data
    # Initial year, number of years, number of valid  data in a year
    anocomienzo, duracion, umbralano = (2018, 85, 0.8)
    # Type of fit (0-GUI, 1-stationary, 2-nonstationary)
    ant = [2, 2, 2, 2, 2, 2]
    # Fourier order for nonstationary analysis
    no_ord = [4, 4, 4, 4, 4, 4]
    # Number of simulations
    no_sim = 350
    # Type of fit functions
    fun = [st.lognorm, lognorm2, normtrun, st.weibull_min, normtrun, st.norm]
    # Number of normals
    no_norm = [False, False, 2, False, 2, False]

    #%% Read data
    # Import wave data (from propagated simar)
    data_path = os.path.join(tests.current_path, '..', '..', '..', '..', 'data', 'third_party',
                             'simulacion_clima_cobos', 'data')
    hm0_name = 'cadiz_hs_sim_emp.pkl'
    tp_name = 'cadiz_tp_sim_emp.pkl'
    dh_name = 'cadiz_dh_sim_emp.pkl'

    hm0 = pd.read_pickle(os.path.join(data_path, hm0_name))
    tp = pd.read_pickle(os.path.join(data_path, tp_name))
    dh = pd.read_pickle(os.path.join(data_path, dh_name))

    # Eliminate null values
    dh[dh < 220] = 220

    modf_wave = pd.concat([hm0, tp, dh], axis=1)
    # Hourly resample
    modf_wave = modf_wave.resample('H').interpolate()

    # # Import wave data (from simar offshore)
    # data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    # modf_file_name = 'guadalete_estuary_wave.modf'
    # path_name = os.path.join(data_path, modf_file_name)
    # modf_wave = MetOceanDF.read_file(path_name)
    # # Hourly resample
    # modf_wave = modf_wave.resample('H').interpolate()

    # Import wind data (from simar)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_wind.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_wind = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_wind = modf_wind.resample('H').interpolate()

    # Import sea level pressure (from era)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_sea_level_pressure.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_slp = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_slp = modf_slp.resample('H').interpolate()

    # Group into dataframe
    wave_wind = pd.concat([modf_wave, modf_wind, modf_slp], axis=1)
    wave_wind.columns = ['hs', 'tp', 'dh', 'vv', 'dv', 'slp']

    # Delete rows where with no common values
    wave_wind.dropna(how='any', inplace=True)

    #%% Preprocessing
    t_step = missing_values.find_timestep(wave_wind) # Find tstep
    data_gaps = missing_values.find_missing_values(wave_wind, t_step)
    wave_wind = missing_values.fill_missing_values(wave_wind, t_step, technique='interpolation', method='nearest',
                                                   limit=16*24, limit_direction='both')
    data_gaps_after = missing_values.find_missing_values(wave_wind, t_step)

    # Add noise for VAR
    noise = np.random.rand(wave_wind.shape[0], wave_wind.shape[1])*1e-2
    wave_wind = wave_wind + noise

    # Save_to_pickle
    wave_wind.to_pickle('wave_wind.p')

    # Group into list of dataframes
    df = list()
    df.append(pd.DataFrame(wave_wind['hs']))
    df.append(pd.DataFrame(wave_wind['tp']))
    df.append(pd.DataFrame(wave_wind['dh']))
    df.append(pd.DataFrame(wave_wind['vv']))
    df.append(pd.DataFrame(wave_wind['dv']))
    df.append(pd.DataFrame(wave_wind['slp']))

    #%% # ANALISIS CLIMATICO (0: PARA SALTARLO, 1: PARA HACERLO; LO MISMO PARA TODOS ESTOS IF)
    if climate_analysis:
        if cdf_pdf_representation:
            for i in range(len(df)):
                # DIBUJO LAS CDF Y PDF DE LOS REGISTROS
                plot_analisis.cdf_pdf_registro(df[i], df[i].columns[0])
                plt.pause(0.5)

        param0 = list()
        # Hs: Check goodness of fit
        #Theoretical fit
        data = wave_wind['hs']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[0].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'hs', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Tp
        # Initial parameters for mixture distribution
        data = wave_wind['tp']
        xlim = 7.5  # where the first pdf ends
        x1, x2 = np.log(data[data <= xlim]), np.log(data[data > xlim])
        alpha = float(len(x1)) / len(data)
        mu1, mu2, sig1, sig2 = np.mean(x1), np.mean(x2), np.std(x1), np.std(x2)
        param0.append(np.array([alpha, mu1, sig1, mu2, sig2]))
        mod = list()
        mod.append(np.array([0, 0, 0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(df[1]), np.max(df[1]), len(n))
        cdf = fun[1].cdf(n, x_cdf, param0[1], mod[0])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'tp', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Dir_wave
        data = wave_wind['dh']
        param0.append(np.array([(240 * np.pi) / 180, (6 * np.pi) / 180, (262 * np.pi) / 180, (5 * np.pi) / 180,
                                0.23261628]))
        mod = list()
        mod.append(np.array([0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(data), np.max(data), len(n))
        x_cdf_rad = (x_cdf * np.pi) / 180
        cdf = fun[2].cdf(n, x_cdf_rad, param0[2], mod[0], no_norm[2])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'dh', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Wind velocity
        data = wave_wind['vv']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[3].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'vv', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Wind direction
        data = wave_wind['dv']
        param0.append(np.array([(107 * np.pi) / 180, (42 * np.pi) / 180, (281 * np.pi) / 180, (40 * np.pi) / 180, 0.4]))
        mod = list()
        mod.append(np.array([0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(data), np.max(data), len(n))
        x_cdf_rad = (x_cdf * np.pi) / 180
        cdf = fun[4].cdf(n, x_cdf_rad, param0[4], mod[0], no_norm[4])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'dv', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Sea level pressure
        # Theoretical fit
        data = wave_wind['slp']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[5].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'slp', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # ANALISIS
        par, mod, f_mix, data_graph = list(), list(), list(), list()
        comienzo = time.time()
        for i in range(len(df)):
            df[i] = df[i].resample('3H').dropna()

            if df[i].columns == 'dh':
                # SE HAN SELECCIONADO LOS ULTIMOS 7 ANOS PARA QUE EL ANALISIS SEA MAS RAPIDO
                analisis_ = analisis.analisis(df[i][df[i].index.year > 2007], fun[i], ant[i], ordg=no_ord[i],
                                              nnorm=no_norm[i], par0=param0[i])
            else:
                analisis_ = analisis.analisis(df[i], fun[i], ant[i], ordg=no_ord[i],
                                              nnorm=no_norm[i], par0=param0[i])

            par.append(analisis_[0])
            mod.append(analisis_[1])
            f_mix.append(analisis_[2])
            data_graph.append(analisis_[3])

            # DIBUJO LOS RESULTADOS (HAY UNA GRAN GAMA DE FUNCIONES DE DIBUJO; VER MANUAL)
            plot_analisis.cuantiles_ne(*data_graph[i])
            plt.pause(0.5)

        # hs
        data = wave_wind['hs']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = ajuste_simple.cdf(n, x, par[0], mod[0], fun[0])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'hs', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # tp
        data = wave_wind['tp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = fun[1].cdf(n, x, par[1], mod[1])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'tp', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # dh
        data = wave_wind['dh']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = fun[2].cdf(n, x, par[2], mod[2], no_norm[2])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'dh', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # vv
        data = wave_wind['vv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = ajuste_simple.cdf(n, x, par[3], mod[3], fun[3])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'vv', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # dv
        data = wave_wind['dv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = fun[4].cdf(n, x, par[4], mod[4], no_norm[4])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'dv', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # slp
        data = wave_wind['slp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        n = np.linspace(0, 1, 100)
        x = np.linspace(np.min(data), np.max(data), len(n))
        cdf = ajuste_simple.cdf(n, x, par[5], mod[5], fun[5])
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit_nest(cdf, ecdf, wave_wind, 'slp', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                                  yqqplot_teo)

        # GUARDO LOS PARAMETROS
        np.save(os.path.join('output', 'analisis', 'parameter_onshore.npy'), par)
        np.save(os.path.join('output', 'analisis', 'mod_onshore.npy'), mod)
        np.save(os.path.join('output', 'analisis', 'f_mix_onshore.npy'), f_mix)

        fin = time.time()
        print
        'Duracion del analisis :  ' + str((fin - comienzo) / 60) + '  minutos'

    # DEPENDENCIA TEMPORAL
    if temporal_dependency:
        # CARGO PARÁMETROS
        df = pd.concat([df[0], df[1], df[2], df[3], df[4], df[5]], axis=1, join='inner').dropna()
        par = np.load(os.path.join('output', 'analisis', 'parameter_onshore.npy'))
        mod = np.load(os.path.join('output', 'analisis', 'mod_onshore.npy'))
        f_mix = np.load(os.path.join('output', 'analisis', 'f_mix_onshore.npy'))

        # SE UTILIZAN LOS PARAMETROS DE SALIDA DEL ANÁLISIS PREVIO
        (df_dt, cdf_) = analisis.dependencia_temporal(df, par, mod, no_norm, f_mix, fun)

        # SE GUARDAN LOS PARAMETROS DEL MODELO VAR
        df_dt.to_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_onshore.p'))
        cdf_.to_pickle(os.path.join('output', 'dependencia_temporal', 'cdf_onshore.p'))

    # SIMULACION CLIMÁTICA
    if climatic_simulation:
        # CARGO PARÁMETROS
        par = np.load(os.path.join('output', 'analisis', 'parameter_onshore.npy'))
        mod = np.load(os.path.join('output', 'analisis', 'mod_onshore.npy'))
        f_mix = np.load(os.path.join('output', 'analisis', 'f_mix_onshore.npy'))
        df_dt = pd.read_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_onshore.p'))

        vars_ = ['hs', 'tp', 'dh', 'vv', 'dv', 'slp']

        # Serie de 5 anos
        year = np.int(np.random.uniform(low=1960, high=2010, size=1))
        year_ini = pd.datetime(year, 1, 1)
        year_end = pd.datetime(year + 5, 1, 1)

        fig4, axes4 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
        axes4[0].plot(wave_wind.loc[year_ini:year_end, 'hs'])
        axes4[0].set_ylabel('Hm0 (m)')
        axes4[0].set_ylim([0, 5])
        axes4[0].grid()

        axes4[1].plot(wave_wind.loc[year_ini:year_end, 'tp'])
        axes4[1].set_ylabel('Tp (s)')
        axes4[1].set_ylim([0, 30])
        axes4[1].legend(['Hindcast', 'Forecast'])
        axes4[1].grid()

        axes4[2].plot(wave_wind.loc[year_ini:year_end, 'dh'])
        axes4[2].set_ylabel('DirW (deg)')
        axes4[2].set_ylim([200, 300])
        axes4[2].legend(['Hindcast', 'Forecast'])
        axes4[2].grid()

        axes4[3].plot(wave_wind.loc[year_ini:year_end, 'vv'])
        axes4[3].set_ylabel('Vv (m/s)')
        axes4[3].set_ylim([0, 30])
        axes4[3].legend(['Hindcast', 'Forecast'])
        axes4[3].grid()

        axes4[4].plot(wave_wind.loc[year_ini:year_end, 'dv'])
        axes4[4].set_ylabel('DirW (deg)')
        axes4[4].set_ylim([0, 360])
        axes4[4].legend(['Hindcast', 'Forecast'])
        axes4[4].grid()

        axes4[5].plot(wave_wind.loc[year_ini:year_end, 'slp'])
        axes4[5].set_ylabel('Slp (mbar)')
        axes4[5].set_ylim([980, 1050])
        axes4[5].set_xlabel('Time')
        axes4[5].legend(['Hindcast', 'Forecast'])
        axes4[5].grid()

        fig4.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_historico_5a_onshore.png'))

        # Serie de 1 anos
        year = np.int(np.random.uniform(low=1960, high=2010, size=1))
        year_ini = pd.datetime(year, 1, 1)
        year_end = pd.datetime(year + 1, 1, 1)

        fig5, axes5 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
        axes5[0].plot(wave_wind.loc[year_ini:year_end, 'hs'])
        axes5[0].set_ylabel('Hm0 (m)')
        axes5[0].set_ylim([0, 5])
        axes5[0].grid()

        axes5[1].plot(wave_wind.loc[year_ini:year_end, 'tp'])
        axes5[1].set_ylabel('Tp (s)')
        axes5[1].set_ylim([0, 30])
        axes5[1].legend(['Hindcast', 'Forecast'])
        axes5[1].grid()

        axes5[2].plot(wave_wind.loc[year_ini:year_end, 'dh'])
        axes5[2].set_ylabel('DirW (deg)')
        axes5[2].set_ylim([200, 300])
        axes5[2].legend(['Hindcast', 'Forecast'])
        axes5[2].grid()

        axes5[3].plot(wave_wind.loc[year_ini:year_end, 'vv'])
        axes5[3].set_ylabel('Vv (m/s)')
        axes5[3].set_ylim([0, 30])
        axes5[3].legend(['Hindcast', 'Forecast'])
        axes5[3].grid()

        axes5[4].plot(wave_wind.loc[year_ini:year_end, 'dv'])
        axes5[4].set_ylabel('DirW (deg)')
        axes5[4].set_ylim([0, 360])
        axes5[4].legend(['Hindcast', 'Forecast'])
        axes5[4].grid()

        axes5[5].plot(wave_wind.loc[year_ini:year_end, 'slp'])
        axes5[5].set_ylabel('Slp (mbar)')
        axes5[5].set_ylim([980, 1050])
        axes5[5].set_xlabel('Time')
        axes5[5].legend(['Hindcast', 'Forecast'])
        axes5[5].grid()

        fig5.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_historico_1a_onshore_.png'))

        # Figura de las cdf y pdf empiricas
        fig3, axes3 = plt.subplots(2, 6, figsize=(20, 7))

        for i in tqdm(range(1, no_sim)):

            df_sim = simulacion.simulacion(anocomienzo, duracion, par, mod, no_norm, f_mix, fun, vars_, df, df_dt,
                                           [0, 0, 0, 0, 0], semilla=int(np.random.rand(1)*1e6))
            # SE GUARDA LA SIMULACIÓN
            # df_sim.to_pickle(os.path.join('output', 'simulacion', 'clima_cadiz_simulado_onshore_' +
            #                               str(i).zfill(4) + '.p'))
            # df_sim.to_csv(os.path.join('output', 'simulacion', 'clima_cadiz_simulado_onshore_' +
            #                            str(i).zfill(4) + '.txt'))

            # Cdf Pdf
            data = df_sim['hs']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 0].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 0].plot(ecdf.index, ecdf, '--', color='0.75')

            data = df_sim['tp']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 1].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 1].plot(ecdf.index, ecdf, '--', color='0.75')

            data = df_sim['dh']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 2].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 2].plot(ecdf.index, ecdf, '--', color='0.75')

            data = df_sim['vv']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 3].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 3].plot(ecdf.index, ecdf, '--', color='0.75')

            data = df_sim['dv']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 4].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 4].plot(ecdf.index, ecdf, '--', color='0.75')

            data = df_sim['slp']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes3[0, 5].plot(epdf.index, epdf, '--', color='0.75')
            axes3[1, 5].plot(ecdf.index, ecdf, '--', color='0.75')

            if i <= 10:
                # Series temporales completas
                fig1, axes1 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
                axes1[0].plot(wave_wind.loc[:, 'hs'])
                axes1[0].plot(df_sim.loc[:, 'hs'])
                axes1[0].set_ylabel('Hm0 (m)')
                axes1[0].set_ylim([0, 5])
                axes1[0].legend(['Hindcast', 'Forecast'])
                axes1[0].grid()

                axes1[1].plot(wave_wind.loc[:, 'tp'])
                axes1[1].plot(df_sim.loc[:, 'tp'])
                axes1[1].set_ylabel('Tp (s)')
                axes1[1].set_ylim([0, 30])
                axes1[1].legend(['Hindcast', 'Forecast'])
                axes1[1].grid()

                axes1[2].plot(wave_wind.loc[:, 'dh'])
                axes1[2].plot(df_sim.loc[:, 'dh'])
                axes1[2].set_ylabel('DirW (deg)')
                axes1[2].set_ylim([200, 300])
                axes1[2].legend(['Hindcast', 'Forecast'])
                axes1[2].grid()

                axes1[3].plot(wave_wind.loc[:, 'vv'])
                axes1[3].plot(df_sim.loc[:, 'vv'])
                axes1[3].set_ylabel('Vv (m/s)')
                axes1[3].set_ylim([0, 30])
                axes1[3].legend(['Hindcast', 'Forecast'])
                axes1[3].grid()

                axes1[4].plot(wave_wind.loc[:, 'dv'])
                axes1[4].plot(df_sim.loc[:, 'dv'])
                axes1[4].set_ylabel('DirW (deg)')
                axes1[4].set_ylim([0, 360])
                axes1[4].legend(['Hindcast', 'Forecast'])
                axes1[4].grid()

                axes1[5].plot(wave_wind.loc[:, 'slp'])
                axes1[5].plot(df_sim.loc[:, 'slp'])
                axes1[5].set_ylabel('Slp (mbar)')
                axes1[5].set_ylim([980, 1050])
                axes1[5].set_xlabel('Time')
                axes1[5].legend(['Hindcast', 'Forecast'])
                axes1[5].grid()

                fig1.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_onshore_' +
                                          str(i).zfill(4) + '.png'))

                # Series de 5 anos
                year = np.int(np.random.uniform(low=2020, high=2090, size=1))
                year_ini = pd.datetime(year, 1, 1)
                year_end = pd.datetime(year+5, 1, 1)

                fig2, axes2 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
                axes2[0].plot(df_sim.loc[year_ini:year_end, 'hs'])
                axes2[0].set_ylabel('Hm0 (m)')
                axes2[0].set_ylim([0, 5])
                axes2[0].grid()

                axes2[1].plot(df_sim.loc[year_ini:year_end, 'tp'])
                axes2[1].set_ylabel('Tp (s)')
                axes2[1].set_ylim([0, 30])
                axes2[1].legend(['Hindcast', 'Forecast'])
                axes2[1].grid()

                axes2[2].plot(df_sim.loc[year_ini:year_end, 'dh'])
                axes2[2].set_ylabel('DirW (deg)')
                axes2[2].set_ylim([200, 300])
                axes2[2].legend(['Hindcast', 'Forecast'])
                axes2[2].grid()

                axes2[3].plot(df_sim.loc[year_ini:year_end, 'vv'])
                axes2[3].set_ylabel('Vv (m/s)')
                axes2[3].set_ylim([0, 30])
                axes2[3].legend(['Hindcast', 'Forecast'])
                axes2[3].grid()

                axes2[4].plot(df_sim.loc[year_ini:year_end, 'dv'])
                axes2[4].set_ylabel('DirW (deg)')
                axes2[4].set_ylim([0, 360])
                axes2[4].legend(['Hindcast', 'Forecast'])
                axes2[4].grid()

                axes2[5].plot(df_sim.loc[year_ini:year_end, 'slp'])
                axes2[5].set_ylabel('Slp (mbar)')
                axes2[5].set_ylim([980, 1050])
                axes2[5].set_xlabel('Time')
                axes2[5].legend(['Hindcast', 'Forecast'])
                axes2[5].grid()

                fig2.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_5a_onshore_' +
                                          str(i).zfill(4) + '.png'))

                # Series de 1 anos
                year = np.int(np.random.uniform(low=2020, high=2090, size=1))
                year_ini = pd.datetime(year, 1, 1)
                year_end = pd.datetime(year + 1, 1, 1)

                fig6, axes6 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
                axes6[0].plot(df_sim.loc[year_ini:year_end, 'hs'])
                axes6[0].set_ylabel('Hm0 (m)')
                axes6[0].set_ylim([0, 5])
                axes6[0].grid()

                axes6[1].plot(df_sim.loc[year_ini:year_end, 'tp'])
                axes6[1].set_ylabel('Tp (s)')
                axes6[1].set_ylim([0, 30])
                axes6[1].legend(['Hindcast', 'Forecast'])
                axes6[1].grid()

                axes6[2].plot(df_sim.loc[year_ini:year_end, 'dh'])
                axes6[2].set_ylabel('DirW (deg)')
                axes6[2].set_ylim([200, 300])
                axes6[2].legend(['Hindcast', 'Forecast'])
                axes6[2].grid()

                axes6[3].plot(df_sim.loc[year_ini:year_end, 'vv'])
                axes6[3].set_ylabel('Vv (m/s)')
                axes6[3].set_ylim([0, 30])
                axes6[3].legend(['Hindcast', 'Forecast'])
                axes6[3].grid()

                axes6[4].plot(df_sim.loc[year_ini:year_end, 'dv'])
                axes6[4].set_ylabel('DirW (deg)')
                axes6[4].set_ylim([0, 360])
                axes6[4].legend(['Hindcast', 'Forecast'])
                axes6[4].grid()

                axes6[5].plot(df_sim.loc[year_ini:year_end, 'slp'])
                axes6[5].set_ylabel('Slp (mbar)')
                axes6[5].set_ylim([980, 1050])
                axes6[5].set_xlabel('Time')
                axes6[5].legend(['Hindcast', 'Forecast'])
                axes6[5].grid()

                fig6.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_1a_onshore_' +
                                          str(i).zfill(4) + '.png'))

        data = wave_wind['hs']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 0].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 0].plot(ecdf.index, ecdf, 'r', lw=2)

        data = wave_wind['tp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 1].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 1].plot(ecdf.index, ecdf, 'r', lw=2)

        data = wave_wind['dh']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 2].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 2].plot(ecdf.index, ecdf, 'r', lw=2)

        data = wave_wind['vv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 3].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 3].plot(ecdf.index, ecdf, 'r', lw=2)

        data = wave_wind['dv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 4].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 4].plot(ecdf.index, ecdf, 'r', lw=2)

        data = wave_wind['slp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes3[0, 5].plot(epdf.index, epdf, 'r', lw=2)
        axes3[1, 5].plot(ecdf.index, ecdf, 'r', lw=2)

        fig3.savefig(os.path.join('output', 'simulacion', 'graficas', 'pdf_cdf_clima_cadiz_simulado_onshore_.png'))


def test_offshore_simulation():
    # Modules activation and deactivation
    climate_analysis = True
    cdf_pdf_representation = False
    temporal_dependency = False
    plot_results_analysis = False
    climatic_simulation = False

    #%% Input data
    # Initial year, number of years, number of valid  data in a year
    anocomienzo, duracion, umbralano = (2018, 85, 0.5)
    # Type of fit (0-GUI, 1-stationary, 2-nonstationary)
    ant = [2, 2, 2, 2, 2, 2]
    # Fourier order for nonstationary analysis
    no_ord = [4, 4, 4, 4, 4, 4]
    # Number of simulations
    no_sim = 350
    # Type of fit functions
    fun = [st.lognorm, lognorm2, normtrun, st.weibull_min, normtrun, st.norm]
    # Number of normals
    no_norm = [False, False, 2, False, 2, False]

    #%% Read data

    # Import wave data (from simar offshore)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_wave.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_wave = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_wave = modf_wave.resample('H').interpolate()

    # Import wind data (from simar)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_wind.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_wind = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_wind = modf_wind.resample('H').interpolate()

    # Import sea level pressure (from era)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_sea_level_pressure.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_slp = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_slp = modf_slp.resample('H').interpolate()

    # Group into dataframe
    wave_wind = pd.concat([modf_wave, modf_wind, modf_slp], axis=1)
    wave_wind.columns = ['hs', 'tp', 'dh', 'vv', 'dv', 'slp']

    # Delete rows where with no common values
    wave_wind.dropna(how='any', inplace=True)

    #%% Preprocessing
    t_step = missing_values.find_timestep(wave_wind)  # Find tstep
    data_gaps = missing_values.find_missing_values(wave_wind, t_step)
    wave_wind = missing_values.fill_missing_values(wave_wind, t_step, technique='interpolation', method='nearest',
                                                   limit=16*24, limit_direction='both')
    data_gaps_after = missing_values.find_missing_values(wave_wind, t_step)

    # Add noise for VAR
    noise = np.random.rand(wave_wind.shape[0], wave_wind.shape[1])*1e-2
    wave_wind = wave_wind + noise

    # Save_to_pickle
    wave_wind.to_pickle('wave_wind_offshore.p')

    # Group into list of dataframes
    df = list()
    df.append(pd.DataFrame(wave_wind['hs']))
    df.append(pd.DataFrame(wave_wind['tp']))
    df.append(pd.DataFrame(wave_wind['dh']))
    df.append(pd.DataFrame(wave_wind['vv']))
    df.append(pd.DataFrame(wave_wind['dv']))
    df.append(pd.DataFrame(wave_wind['slp']))

    #%% # ANALISIS CLIMATICO (0: PARA SALTARLO, 1: PARA HACERLO; LO MISMO PARA TODOS ESTOS IF)
    if climate_analysis:
        if cdf_pdf_representation:
            for i in range(len(df)):
                # DIBUJO LAS CDF Y PDF DE LOS REGISTROS
                plot_analisis.cdf_pdf_registro(df[i], df[i].columns[0])
                plt.pause(0.5)

        param0 = list()

        # # Hs: Check goodness of fit
        data = wave_wind['hs']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[0].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'hs', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # data = wave_wind['hs']
        # # QQ Plot lo locate tails
        # st.probplot(data, dist="norm", plot=plt)
        # plt.grid()
        #
        # # Fit GPD to the upper tail to extract the parameter xi
        # threshold_percentile = 95
        # minimum_interarrival_time = pd.Timedelta('3 days')
        # minimum_cycle_length = pd.Timedelta('3 hours')
        # interpolation = True
        # interpolation_method = 'linear'
        # interpolation_freq = '1min'
        # truncate = False
        # extra_info = False
        #
        # threshold = np.percentile(wave_wind.loc[:, 'hs'], threshold_percentile)
        # cycles, calm_periods = extremal.extreme_events(wave_wind, 'hs', threshold, minimum_interarrival_time,
        #                                                minimum_cycle_length, interpolation, interpolation_method,
        #                                                interpolation_freq, truncate, extra_info)
        # # Peaks over threshold
        # peaks_over_thres = extremal.events_max(cycles)
        # # POT Empirical distribution
        # ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
        # n_peaks_year = len(peaks_over_thres) / len(wave_wind['hs'].index.year.unique())
        # ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)
        # # Fit POT to Scipy-GPD
        # (param, x_gpd, y_gpd, y_gpd_rp) = extremal.extremal_distribution_fit(data=wave_wind, var_name='hs',
        #                                                                      sample=peaks_over_thres,
        #                                                                      threshold=threshold, fit_type='gpd',
        #                                                                      x_min=0.90 * min(peaks_over_thres),
        #                                                                      x_max=1.5 * max(peaks_over_thres),
        #                                                                      n_points=1000,
        #                                                                      cumulative=True)
        # plt.figure()
        # ax = plt.axes()
        # ax.plot(x_gpd, y_gpd, label='GPD Fit')
        # ax.plot(ecdf_pot.index, ecdf_pot, '.r', label='POT ECDF')
        # plt.xlabel('Return Period (years)')
        # plt.ylabel('Hm0 (m)')
        # ax.legend()
        # plt.grid()
        #
        # plt.figure()
        # ax = plt.axes()
        # ax.semilogx(y_gpd_rp, x_gpd, label='GPD Fit')
        # ax.semilogx(ecdf_pot_rp, ecdf_pot_rp.index, '.r', label='POT ECDF')
        # plt.xlim(0, 500)
        # plt.xlabel('Return Period (years)')
        # plt.ylabel('Hm0 (m)')
        # ax.legend(['GPD Fit', 'POT ECDF'])
        # plt.grid()
        #
        # xi = param[0]
        #
        # # Fit lognormal to data to extract mu and sigma
        # (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type='lognorm', x_min=np.min(data),
        #                                                         x_max=1.1 * np.max(data), n_points=1000)
        # mu = param[1]
        # sigma = param[2]
        #
        # data = np.log(wave_wind['hs'])
        # param0.append(np.array([np.mean(data), np.std(data), xi, -0.8, 3.6]))
        # # param0.append(np.array([-0.63092061, 0.15189784, 0.19899842, -0.4940286, 4.45337391]))
        # # param0.append(np.array([-0.1452, 0.6639, xi, 0.01, 2.01]))
        # mod = list()
        # mod.append(np.array([0, 0, 0]))
        #
        # # To check the goodness of fit, the cdf is calculated
        # n = np.linspace(0, 1, 100)
        # x_cdf = np.linspace(np.min(df[0]), np.max(df[0]), len(n))
        # cdf = fun[0].cdf(n, x_cdf, param0[0], mod[0])
        #
        # data = wave_wind['hs']
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # # PP - Plot values
        # (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # # QQ - Plot values
        # (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # # Plot Goodness of fit
        # theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'hs', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
        #                                      yqqplot_teo)
        #
        # # Plot figure
        # x = df[0]
        # x = x.dropna(axis=0, how='all')
        # yp, xp = np.histogram(x, bins=25, normed=True)
        # xp = np.diff(xp) / 2 + xp[0:-1]
        #
        # fig1, ax1 = plt.subplots(figsize=(20, 20))
        # Yp = np.cumsum(yp) / np.cumsum(yp)[-1]
        # plt.plot(xp, Yp, 'gray', label='Registro')
        # plt.plot(x_cdf, cdf, 'blue', label='Ajuste')
        # plt.ylabel('cdf', fontweight='bold')
        # plt.legend(loc='best', numpoints=1)
        # plt.gcf().subplots_adjust(bottom=0.2, left=0.15, right=0.85)
        # fig1.savefig(os.path.join('output', 'analisis', 'graficas', 'goodness_of_fit_' + 'hs' + '.png'))

        # Tp
        # Initial parameters for mixture distribution
        data = wave_wind['tp']
        xlim = 7.5  # where the first pdf ends
        x1, x2 = np.log(data[data <= xlim]), np.log(data[data > xlim])
        alpha = float(len(x1)) / len(data)
        mu1, mu2, sig1, sig2 = np.mean(x1), np.mean(x2), np.std(x1), np.std(x2)
        param0.append(np.array([alpha, mu1, sig1, mu2, sig2]))
        mod = list()
        mod.append(np.array([0, 0, 0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(df[1]), np.max(df[1]), len(n))
        cdf = fun[1].cdf(n, x_cdf, param0[1], mod[0])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'tp', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Dir_wave
        data = wave_wind['dh']
        param0.append(np.array([(125 * np.pi) / 180, (25 * np.pi) / 180, (275 * np.pi) / 180, (25 * np.pi) / 180,
                                0.23]))
        mod = list()
        mod.append(np.array([0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(data), np.max(data), len(n))
        x_cdf_rad = (x_cdf * np.pi) / 180
        cdf = fun[2].cdf(n, x_cdf_rad, param0[2], mod[0], no_norm[2])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'dh', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Wind velocity
        data = wave_wind['vv']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[3].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'vv', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Wind direction
        data = wave_wind['dv']
        param0.append(np.array([(107 * np.pi) / 180, (42 * np.pi) / 180, (281 * np.pi) / 180, (40 * np.pi) / 180, 0.4]))
        mod = list()
        mod.append(np.array([0, 0]))

        # To check the goodness of fit, the cdf is calculated
        n = np.linspace(0, 1, 100)
        x_cdf = np.linspace(np.min(data), np.max(data), len(n))
        x_cdf_rad = (x_cdf * np.pi) / 180
        cdf = fun[4].cdf(n, x_cdf_rad, param0[4], mod[0], no_norm[4])

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x_cdf, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x_cdf, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'dv', x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Sea level pressure
        # Theoretical fit
        data = wave_wind['slp']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data, fit_type=fun[5].name, x_min=np.min(data),
                                                                x_max=1.1 * np.max(data), n_points=1000)
        param0.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, wave_wind, 'slp', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # ANALISIS
        par, mod, f_mix, data_graph = list(), list(), list(), list()
        comienzo = time.time()
        for i in range(len(df)):

            if df[i].columns == 'hs':
                df[i] = df[i].resample('3H').dropna()
                # SE HAN SELECCIONADO LOS ULTIMOS 7 ANOS PARA QUE EL ANALISIS SEA MAS RAPIDO
                analisis_ = analisis.analisis(df[i], fun[i], ant[i], ordg=no_ord[i],
                                              nnorm=no_norm[i])
            else:
                df[i] = df[i].resample('3H').dropna()
                analisis_ = analisis.analisis(df[i], fun[i], ant[i], ordg=no_ord[i],
                                              nnorm=no_norm[i], par0=param0[i])

            par.append(analisis_[0])
            mod.append(analisis_[1])
            f_mix.append(analisis_[2])

            aux = list(analisis_[3])
            aux[5] = i
            aux = tuple(aux)
            data_graph.append(aux)

            # DIBUJO LOS RESULTADOS (HAY UNA GRAN GAMA DE FUNCIONES DE DIBUJO; VER MANUAL)
            plot_analisis.cuantiles_ne(fun, *data_graph[i])
            plt.pause(0.5)

        # GUARDO LOS PARAMETROS
        np.save(os.path.join('output', 'analisis', 'parameter_offshore.npy'), par)
        np.save(os.path.join('output', 'analisis', 'mod_offshore.npy'), mod)
        np.save(os.path.join('output', 'analisis', 'f_mix_offshore.npy'), f_mix)

        fichero = os.path.join('output', 'analisis', 'data_graph_offshore.p')
        with open(fichero, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(data_graph, filehandle)

        # with open(fichero, 'rb') as filehandle:
        #     # store the data as binary data stream
        #     data_graph = pickle.load(filehandle)


        fin = time.time()
        print
        'Duracion del analisis :  ' + str((fin - comienzo) / 60) + '  minutos'

    # DEPENDENCIA TEMPORAL
    if temporal_dependency:
        # CARGO PARÁMETROS
        df = pd.concat([df[0], df[1], df[2], df[3], df[4], df[5]], axis=1, join='inner').dropna()
        par = np.load(os.path.join('output', 'analisis', 'parameter_offshore.npy'))
        mod = np.load(os.path.join('output', 'analisis', 'mod_offshore.npy'))
        f_mix = np.load(os.path.join('output', 'analisis', 'f_mix_offshore.npy'))

        # SE UTILIZAN LOS PARAMETROS DE SALIDA DEL ANÁLISIS PREVIO
        (df_dt, cdf_) = analisis.dependencia_temporal(df, par, mod, no_norm, f_mix, fun)

        # SE GUARDAN LOS PARAMETROS DEL MODELO VAR
        df_dt.to_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_offshore.p'))
        cdf_.to_pickle(os.path.join('output', 'dependencia_temporal', 'cdf_offshore.p'))

    # REPRESENTACION RESULTADOS ANALISIS
    if plot_results_analysis:
        data_graph = np.load(os.path.join('output', 'analisis', 'parameter_offshore.npy'))
        for i in range(len(df)):
            plot_analisis.cuantiles_ne(*data_graph[i])

    # SIMULACION CLIMÁTICA
    if climatic_simulation:
        # CARGO PARÁMETROS
        par = np.load(os.path.join('output', 'analisis', 'parameter_offshore.npy'))
        mod = np.load(os.path.join('output', 'analisis', 'mod_offshore.npy'))
        f_mix = np.load(os.path.join('output', 'analisis', 'f_mix_offshore.npy'))
        df_dt = pd.read_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_offshore.p'))

        vars_ = ['hs', 'tp', 'dh', 'vv', 'dv', 'slp']

        # Serie de 5 anos
        year = np.int(np.random.uniform(low=1960, high=2010, size=1))
        year_ini = pd.datetime(year, 1, 1)
        year_end = pd.datetime(year + 5, 1, 1)

        # fig4, axes4 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
        # axes4[0].plot(wave_wind.loc[year_ini:year_end, 'hs'])
        # axes4[0].set_ylabel('Hm0 (m)')
        # axes4[0].set_ylim([0, 5])
        # axes4[0].grid()
        #
        # axes4[1].plot(wave_wind.loc[year_ini:year_end, 'tp'])
        # axes4[1].set_ylabel('Tp (s)')
        # axes4[1].set_ylim([0, 30])
        # axes4[1].legend(['Hindcast', 'Forecast'])
        # axes4[1].grid()
        #
        # axes4[2].plot(wave_wind.loc[year_ini:year_end, 'dh'])
        # axes4[2].set_ylabel('DirW (deg)')
        # axes4[2].set_ylim([200, 300])
        # axes4[2].legend(['Hindcast', 'Forecast'])
        # axes4[2].grid()
        #
        # axes4[3].plot(wave_wind.loc[year_ini:year_end, 'vv'])
        # axes4[3].set_ylabel('Vv (m/s)')
        # axes4[3].set_ylim([0, 30])
        # axes4[3].legend(['Hindcast', 'Forecast'])
        # axes4[3].grid()
        #
        # axes4[4].plot(wave_wind.loc[year_ini:year_end, 'dv'])
        # axes4[4].set_ylabel('DirW (deg)')
        # axes4[4].set_ylim([0, 360])
        # axes4[4].legend(['Hindcast', 'Forecast'])
        # axes4[4].grid()
        #
        # axes4[5].plot(wave_wind.loc[year_ini:year_end, 'slp'])
        # axes4[5].set_ylabel('Slp (mbar)')
        # axes4[5].set_ylim([980, 1050])
        # axes4[5].set_xlabel('Time')
        # axes4[5].legend(['Hindcast', 'Forecast'])
        # axes4[5].grid()
        #
        # fig4.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_historico_5a_offshore.png'))

        # Serie de 1 anos
        year = np.int(np.random.uniform(low=1960, high=2010, size=1))
        year_ini = pd.datetime(year, 1, 1)
        year_end = pd.datetime(year + 1, 1, 1)

        # fig5, axes5 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
        # axes5[0].plot(wave_wind.loc[year_ini:year_end, 'hs'])
        # axes5[0].set_ylabel('Hm0 (m)')
        # axes5[0].set_ylim([0, 10])
        # axes5[0].grid()
        #
        # axes5[1].plot(wave_wind.loc[year_ini:year_end, 'tp'])
        # axes5[1].set_ylabel('Tp (s)')
        # axes5[1].set_ylim([0, 30])
        # axes5[1].legend(['Hindcast', 'Forecast'])
        # axes5[1].grid()
        #
        # axes5[2].plot(wave_wind.loc[year_ini:year_end, 'dh'])
        # axes5[2].set_ylabel('DirW (deg)')
        # axes5[2].set_ylim([0, 360])
        # axes5[2].legend(['Hindcast', 'Forecast'])
        # axes5[2].grid()
        #
        # axes5[3].plot(wave_wind.loc[year_ini:year_end, 'vv'])
        # axes5[3].set_ylabel('Vv (m/s)')
        # axes5[3].set_ylim([0, 30])
        # axes5[3].legend(['Hindcast', 'Forecast'])
        # axes5[3].grid()
        #
        # axes5[4].plot(wave_wind.loc[year_ini:year_end, 'dv'])
        # axes5[4].set_ylabel('DirW (deg)')
        # axes5[4].set_ylim([0, 360])
        # axes5[4].legend(['Hindcast', 'Forecast'])
        # axes5[4].grid()
        #
        # axes5[5].plot(wave_wind.loc[year_ini:year_end, 'slp'])
        # axes5[5].set_ylabel('Slp (mbar)')
        # axes5[5].set_ylim([980, 1050])
        # axes5[5].set_xlabel('Time')
        # axes5[5].legend(['Hindcast', 'Forecast'])
        # axes5[5].grid()
        #
        # fig5.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_historico_1a_offshore_.png'))

        # Figura de las cdf y pdf empiricas
        fig3, axes3 = plt.subplots(2, 6, figsize=(80, 10))
        plt.rcParams.update({'font.size': 22})

        for i in tqdm(range(1, no_sim)):

            df_sim = simulacion.simulacion(anocomienzo, duracion, par, mod, no_norm, f_mix, fun, vars_, df, df_dt,
                                           [0, 0, 0, 0, 0], semilla=int(np.random.rand(1)*1e6))

            df_sim = df_sim.resample('1H').interpolate()
            # SE GUARDA LA SIMULACIÓN
            # df_sim.to_pickle(os.path.join('output', 'simulacion', 'clima_cadiz_simulado_onshore_' +
            #                               str(i).zfill(4) + '.p'))
            # df_sim.to_csv(os.path.join('output', 'simulacion', 'clima_cadiz_simulado_onshore_' +
            #                            str(i).zfill(4) + '.txt'))

            # Cdf Pdf
            # data = df_sim['hs']
            # paso = 0.1
            # bins = np.max(data) / (paso * 2.0)
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 0].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 0].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[0, 0].set_ylabel('PDF', fontsize=16)
            # axes3[0, 0].set_xticklabels([])
            # axes3[1, 0].set_ylabel('CDF', fontsize=16)
            # axes3[1, 0].set_xlabel('$H_{m0} (m)$', fontsize=16)
            # axes3[1, 0].set_xticks([0, 5, 10])
            #
            # axes3[0, 0].set_xticks([0, 5, 10])
            # axes3[0, 0].set_yticks([0, 0.5, 1])
            # axes3[0, 0].set_xticklabels([])
            # axes3[0, 0].grid(True)
            # axes3[1, 0].grid(True)
            # axes3[0, 0].set_xlim([0, 10])
            # axes3[0, 0].set_ylim([0, 1])
            # axes3[1, 0].set_xlim([0, 10])
            # axes3[1, 0].set_ylim([0, 1.05])
            #
            # data = df_sim['tp']
            # paso = 0.25
            # bins = np.max(data) / (paso * 2.0)
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 1].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 1].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[1, 1].set_xlabel('$T_{p} (s)$', fontsize=16)
            # axes3[0, 1].set_xticklabels([])
            # axes3[1, 1].set_yticklabels([])
            # axes3[1, 1].set_xticks([0, 15, 30])
            #
            # axes3[0, 1].set_xticks([0, 15, 30])
            # axes3[0, 1].set_yticks([0, 0.07, 0.14])
            # axes3[0, 1].set_xticklabels([])
            # axes3[0, 1].grid(True)
            # axes3[1, 1].grid(True)
            # axes3[0, 1].set_xlim([0, 30])
            # axes3[0, 1].set_ylim([0, 0.14])
            # axes3[1, 1].set_xlim([0, 30])
            # axes3[1, 1].set_ylim([0, 1.05])
            #
            # data = df_sim['dh']
            # paso = 1
            # bins = np.max(data) / (paso * 2.0)
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 2].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 2].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[1, 2].set_xlabel('$w_{theta} ()$', fontsize=16)
            # axes3[0, 2].set_xticklabels([])
            # axes3[1, 2].set_yticklabels([])
            # axes3[1, 2].set_xticks([0, 180, 360])
            #
            # axes3[0, 2].set_xticks([0, 180, 360])
            # axes3[0, 2].set_yticks([0, 0.01, 0.02])
            # axes3[0, 2].set_xticklabels([])
            # axes3[0, 2].grid(True)
            # axes3[1, 2].grid(True)
            # axes3[0, 2].set_xlim([0, 360])
            # axes3[0, 2].set_ylim([0, 0.02])
            # axes3[1, 2].set_xlim([0, 360])
            # axes3[1, 2].set_ylim([0, 1.05])
            #
            # data = df_sim['vv']
            # paso = 0.1
            # bins = np.max(data) / (paso * 2.0)
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 3].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 3].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[1, 3].set_xlabel('$u_{10} (m/s)$', fontsize=16)
            # axes3[0, 3].set_xticklabels([])
            # axes3[1, 3].set_yticklabels([])
            # axes3[1, 3].set_xticks([0, 15, 30])
            #
            # axes3[0, 3].set_xticks([0, 15, 30])
            # axes3[0, 3].set_yticks([0, 0.08, 0.16])
            # axes3[0, 3].set_xticklabels([])
            # axes3[0, 3].grid(True)
            # axes3[1, 3].grid(True)
            # axes3[0, 3].set_xlim([0, 30])
            # axes3[0, 3].set_ylim([0, 0.16])
            # axes3[1, 3].set_xlim([0, 30])
            # axes3[1, 3].set_ylim([0, 1.05])
            #
            # data = df_sim['dv']
            # bins = 60
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 4].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 4].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[1, 4].set_xlabel('$u_{\\theta} ()$', fontsize=16)
            # axes3[0, 4].set_xticklabels([])
            # axes3[1, 4].set_yticklabels([])
            # axes3[1, 4].set_xticks([0, 180, 360])
            #
            # axes3[0, 4].set_xticks([0, 180, 360])
            # axes3[0, 4].set_yticks([0, 0.003, 0.006])
            # axes3[0, 4].set_xticklabels([])
            # axes3[0, 4].grid(True)
            # axes3[1, 4].grid(True)
            # axes3[0, 4].set_xlim([0, 360])
            # axes3[0, 4].set_ylim([0, 0.006])
            # axes3[1, 4].set_xlim([0, 360])
            # axes3[1, 4].set_ylim([0, 1.05])
            #
            # data = df_sim['slp']
            # paso = 1
            # bins = np.max(data) / (paso * 2.0)
            # ecdf = empirical_distributions.ecdf_histogram(data)
            # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
            # axes3[0, 5].plot(epdf.index, epdf, '--', color='0.75')
            # axes3[1, 5].plot(ecdf.index, ecdf, '--', color='0.75')
            # axes3[1, 5].set_xlabel('$slp (mbar)$', fontsize=16)
            # axes3[0, 5].set_xticklabels([])
            # axes3[1, 5].set_yticklabels([])
            # axes3[1, 5].set_xticks([980, 1015, 1050])
            #
            # axes3[0, 5].set_xticks([980, 1015, 1050])
            # axes3[0, 5].set_yticks([0, 0.06, 0.12])
            # axes3[0, 5].set_xticklabels([])
            # axes3[0, 5].grid(True)
            # axes3[1, 5].grid(True)
            # axes3[0, 5].set_xlim([980, 1050])
            # axes3[0, 5].set_ylim([0, 0.12])
            # axes3[1, 5].set_xlim([980, 1050])
            # axes3[1, 5].set_ylim([0, 1.05])
            #
            # if i <= 10:
            #     # Series temporales completas
            #     fig1, axes1 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
            #     axes1[0].plot(wave_wind.loc[:, 'hs'])
            #     axes1[0].plot(df_sim.loc[:, 'hs'])
            #     axes1[0].set_ylabel('Hm0 (m)')
            #     axes1[0].set_ylim([0, 10])
            #     axes1[0].legend(['Hindcast', 'Forecast'])
            #     axes1[0].grid()
            #
            #     axes1[1].plot(wave_wind.loc[:, 'tp'])
            #     axes1[1].plot(df_sim.loc[:, 'tp'])
            #     axes1[1].set_ylabel('Tp (s)')
            #     axes1[1].set_ylim([0, 30])
            #     axes1[1].legend(['Hindcast', 'Forecast'])
            #     axes1[1].grid()
            #
            #     axes1[2].plot(wave_wind.loc[:, 'dh'])
            #     axes1[2].plot(df_sim.loc[:, 'dh'])
            #     axes1[2].set_ylabel('DirW (deg)')
            #     axes1[2].set_ylim([0, 360])
            #     axes1[2].legend(['Hindcast', 'Forecast'])
            #     axes1[2].grid()
            #
            #     axes1[3].plot(wave_wind.loc[:, 'vv'])
            #     axes1[3].plot(df_sim.loc[:, 'vv'])
            #     axes1[3].set_ylabel('Vv (m/s)')
            #     axes1[3].set_ylim([0, 30])
            #     axes1[3].legend(['Hindcast', 'Forecast'])
            #     axes1[3].grid()
            #
            #     axes1[4].plot(wave_wind.loc[:, 'dv'])
            #     axes1[4].plot(df_sim.loc[:, 'dv'])
            #     axes1[4].set_ylabel('DirW (deg)')
            #     axes1[4].set_ylim([0, 360])
            #     axes1[4].legend(['Hindcast', 'Forecast'])
            #     axes1[4].grid()
            #
            #     axes1[5].plot(wave_wind.loc[:, 'slp'])
            #     axes1[5].plot(df_sim.loc[:, 'slp'])
            #     axes1[5].set_ylabel('Slp (mbar)')
            #     axes1[5].set_ylim([980, 1050])
            #     axes1[5].set_xlabel('Time')
            #     axes1[5].legend(['Hindcast', 'Forecast'])
            #     axes1[5].grid()
            #
            #     fig1.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_offshore_' +
            #                               str(i).zfill(4) + '.png'))
            #
            #     # Series de 5 anos
            #     year = np.int(np.random.uniform(low=2020, high=2090, size=1))
            #     year_ini = pd.datetime(year, 1, 1)
            #     year_end = pd.datetime(year+5, 1, 1)
            #
            #     fig2, axes2 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
            #     axes2[0].plot(df_sim.loc[year_ini:year_end, 'hs'])
            #     axes2[0].set_ylabel('Hm0 (m)')
            #     axes2[0].set_ylim([0, 10])
            #     axes2[0].grid()
            #
            #     axes2[1].plot(df_sim.loc[year_ini:year_end, 'tp'])
            #     axes2[1].set_ylabel('Tp (s)')
            #     axes2[1].set_ylim([0, 30])
            #     axes2[1].legend(['Hindcast', 'Forecast'])
            #     axes2[1].grid()
            #
            #     axes2[2].plot(df_sim.loc[year_ini:year_end, 'dh'])
            #     axes2[2].set_ylabel('DirW (deg)')
            #     axes2[2].set_ylim([0, 360])
            #     axes2[2].legend(['Hindcast', 'Forecast'])
            #     axes2[2].grid()
            #
            #     axes2[3].plot(df_sim.loc[year_ini:year_end, 'vv'])
            #     axes2[3].set_ylabel('Vv (m/s)')
            #     axes2[3].set_ylim([0, 30])
            #     axes2[3].legend(['Hindcast', 'Forecast'])
            #     axes2[3].grid()
            #
            #     axes2[4].plot(df_sim.loc[year_ini:year_end, 'dv'])
            #     axes2[4].set_ylabel('DirW (deg)')
            #     axes2[4].set_ylim([0, 360])
            #     axes2[4].legend(['Hindcast', 'Forecast'])
            #     axes2[4].grid()
            #
            #     axes2[5].plot(df_sim.loc[year_ini:year_end, 'slp'])
            #     axes2[5].set_ylabel('Slp (mbar)')
            #     axes2[5].set_ylim([980, 1050])
            #     axes2[5].set_xlabel('Time')
            #     axes2[5].legend(['Hindcast', 'Forecast'])
            #     axes2[5].grid()
            #
            #     fig2.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_5a_offshore_' +
            #                               str(i).zfill(4) + '.png'))
            #
            #     # Series de 1 anos
            #     year = np.int(np.random.uniform(low=2020, high=2090, size=1))
            #     year_ini = pd.datetime(year, 1, 1)
            #     year_end = pd.datetime(year + 1, 1, 1)
            #
            #     fig6, axes6 = plt.subplots(6, 1, sharex=True, figsize=(20, 20))
            #     axes6[0].plot(df_sim.loc[year_ini:year_end, 'hs'])
            #     axes6[0].set_ylabel('Hm0 (m)')
            #     axes6[0].set_ylim([0, 10])
            #     axes6[0].grid()
            #
            #     axes6[1].plot(df_sim.loc[year_ini:year_end, 'tp'])
            #     axes6[1].set_ylabel('Tp (s)')
            #     axes6[1].set_ylim([0, 30])
            #     axes6[1].legend(['Hindcast', 'Forecast'])
            #     axes6[1].grid()
            #
            #     axes6[2].plot(df_sim.loc[year_ini:year_end, 'dh'])
            #     axes6[2].set_ylabel('DirW (deg)')
            #     axes6[2].set_ylim([0, 360])
            #     axes6[2].legend(['Hindcast', 'Forecast'])
            #     axes6[2].grid()
            #
            #     axes6[3].plot(df_sim.loc[year_ini:year_end, 'vv'])
            #     axes6[3].set_ylabel('Vv (m/s)')
            #     axes6[3].set_ylim([0, 30])
            #     axes6[3].legend(['Hindcast', 'Forecast'])
            #     axes6[3].grid()
            #
            #     axes6[4].plot(df_sim.loc[year_ini:year_end, 'dv'])
            #     axes6[4].set_ylabel('DirW (deg)')
            #     axes6[4].set_ylim([0, 360])
            #     axes6[4].legend(['Hindcast', 'Forecast'])
            #     axes6[4].grid()
            #
            #     axes6[5].plot(df_sim.loc[year_ini:year_end, 'slp'])
            #     axes6[5].set_ylabel('Slp (mbar)')
            #     axes6[5].set_ylim([980, 1050])
            #     axes6[5].set_xlabel('Time')
            #     axes6[5].legend(['Hindcast', 'Forecast'])
            #     axes6[5].grid()
            #
            #     fig6.savefig(os.path.join('output', 'simulacion', 'graficas', 'clima_cadiz_simulado_1a_offshore_' +
            #                               str(i).zfill(4) + '.png'))

            # Save the file
            df_sim.to_csv(os.path.join('output', 'simulacion', 'series_temporales', 'wave_wind_slp_offshore_500',
                                       'wave_wind_slp_guadalete_offshore_sim_' + str(i).zfill(4) + '.txt'),
                          sep=nb(b'\t'), float_format='%.3f')

        # data = wave_wind['hs']
        # paso = 0.1
        # bins = np.max(data) / (paso * 2.0)
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 0].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 0].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # data = wave_wind['tp']
        # paso = 1
        # bins = np.max(data) / (paso * 2.0)
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 1].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 1].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # data = wave_wind['dh']
        # paso = 1
        # bins = np.max(data) / (paso * 2.0)
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 2].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 2].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # data = wave_wind['vv']
        # paso = 0.1
        # bins = np.max(data) / (paso * 2.0)
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 3].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 3].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # data = wave_wind['dv']
        # bins = 60
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 4].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 4].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # data = wave_wind['slp']
        # paso = 1
        # bins = np.max(data) / (paso * 2.0)
        # ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        # axes3[0, 5].plot(epdf.index, epdf, 'r', lw=2)
        # axes3[1, 5].plot(ecdf.index, ecdf, 'r', lw=2)
        #
        # fig3.savefig(os.path.join('output', 'simulacion', 'graficas', 'pdf_cdf_clima_cadiz_simulado_offshore_.png'))