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
from tqdm import tqdm

from climate import tests
from metoceandataframe.metoceandataframe import MetOceanDF
from preprocessing import missing_values
from climate.stats import empirical_distributions

from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo import analisis, simulacion
from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo.fdist import ajuste_simple, \
    normenr, normtrun, lognorm_genpar2, lognorm2, weib_genpar2, norm_genpar2
from climate.third_party.simulacion_clima_cobos.clima_maritimo.graficas import plot_analisis


def test_figuras_cuantiles_ajuste_no_estacionario():

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

    d_frame = pd.DataFrame(wave_wind)

    fichero = os.path.join('output', 'analisis', 'data_graph_offshore.p')
    with open(fichero, 'rb') as filehandle:
        # store the data as binary data stream
        data_graph = pickle.load(filehandle)

    # DIBUJO LOS RESULTADOS (HAY UNA GRAN GAMA DE FUNCIONES DE DIBUJO; VER MANUAL)
    plot_analisis.cuantiles_ne_paper(fun, *data_graph)


def test_pdf_cdf_simulaciones():
    #%% Input data
    # Number of simulations
    no_sim = 100
    # Preparo la figura
    plt.rcParams.update({'font.size': 12})
    fig3, axes3 = plt.subplots(3, 3, figsize=(12, 10))
    plt.delaxes(axes3[2, 1])
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

    # Import sea level pressure (from era)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_astronomical_tide.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_at = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_at = modf_at.resample('H').interpolate()
    at_hindcast_df = pd.DataFrame(modf_at)

    # Import sea level pressure (from era)
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_astronomical_tide_forecast.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_at_fc = MetOceanDF.read_file(path_name)
    # Hourly resample
    modf_at_fc = modf_at_fc.resample('H').interpolate()
    at_forecast_df = pd.DataFrame(modf_at_fc)

    # Group into dataframe
    wave_wind = pd.concat([modf_wave, modf_wind, modf_slp], axis=1)
    wave_wind.columns = ['hs', 'tp', 'dh', 'vv', 'dv', 'slp']

    # Delete rows where with no common values
    wave_wind.dropna(how='any', inplace=True)

    # Lectura de descarga fluvial
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_river_discharge.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_rd = MetOceanDF.read_file(path_name)

    # Group into dataframe
    river_discharge = pd.DataFrame(modf_rd)
    # Delete rows where with no common values
    river_discharge.dropna(how='any', inplace=True)

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


    for i in tqdm(range(1, no_sim)):
        file_name_simar_sim = os.path.join('output', 'simulacion', 'series_temporales', 'wave_wind_slp_offshore_500',
                                       'wave_wind_slp_guadalete_offshore_sim_' + str(i).zfill(4) + '.txt')

        file_name_rd_sim = os.path.join('output', 'simulacion', 'series_temporales', 'descarga_fluvial_500',
                                          'descarga_fluvial_guadalete_sim_' + str(i).zfill(4) + '.txt')

        df_simar_sim = pd.read_table(file_name_simar_sim, index_col=0)
        df_rd_sim = pd.read_table(file_name_rd_sim, index_col=0)



        # Cdf Pdf
        data = df_simar_sim['hs']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[0, 0].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[0, 0].set_ylabel('CDF', fontsize=16)
        axes3[0, 0].set_xlabel('$H_{m0} (m)$', fontsize=16)
        axes3[0, 0].set_xticks([0, 5, 10])
        axes3[0, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[0, 0].grid(True)
        axes3[0, 0].set_xlim([0, 10])
        axes3[0, 0].set_ylim([0, 1.05])


        data = df_simar_sim['tp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[0, 1].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[0, 1].set_ylabel('', fontsize=16)
        axes3[0, 1].set_xlabel('$T_{p} (s)$', fontsize=16)
        axes3[0, 1].set_xticks([0, 12, 24])
        axes3[0, 1].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[0, 1].set_yticklabels([])
        axes3[0, 1].grid(True)
        axes3[0, 1].set_xlim([0, 24])
        axes3[0, 1].set_ylim([0, 1.05])

        data = df_simar_sim['dh']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[0, 2].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[0, 2].set_ylabel('', fontsize=16)
        axes3[0, 2].set_xlabel('$w_{theta} (^\circ)$', fontsize=16)
        axes3[0, 2].set_xticks([0, 180, 360])
        axes3[0, 2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[0, 2].set_yticklabels([])
        axes3[0, 2].grid(True)
        axes3[0, 2].set_xlim([0, 360])
        axes3[0, 2].set_ylim([0, 1.05])

        data = df_simar_sim['vv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[1, 0].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[1, 0].set_ylabel('CDF', fontsize=16)
        axes3[1, 0].set_xlabel('$u_{10} (m/s)$', fontsize=16)
        axes3[1, 0].set_xticks([0, 15, 30])
        axes3[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[1, 0].grid(True)
        axes3[1, 0].set_xlim([0, 30])
        axes3[1, 0].set_ylim([0, 1.05])

        data = df_simar_sim['dv']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[1, 1].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[1, 1].set_ylabel('', fontsize=16)
        axes3[1, 1].set_xlabel('$u_{\\theta} (^\circ)$', fontsize=16)
        axes3[1, 1].set_xticks([0, 180, 360])
        axes3[1, 1].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[1, 1].set_yticklabels([])
        axes3[1, 1].grid(True)
        axes3[1, 1].set_xlim([0, 360])
        axes3[1, 1].set_ylim([0, 1.05])

        data = df_simar_sim['slp']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[1, 2].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[1, 2].set_ylabel('', fontsize=16)
        axes3[1, 2].set_xlabel('$slp (mbar)$', fontsize=16)
        axes3[1, 2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[1, 2].set_yticklabels([])
        axes3[1, 2].set_xticks([980, 1015, 1050])
        axes3[1, 2].grid(True)
        axes3[1, 2].set_xlim([980, 1050])
        axes3[1, 2].set_ylim([0, 1.05])


        data = at_forecast_df['Eta']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[2, 0].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[2, 0].set_ylabel('CDF', fontsize=16)
        axes3[2, 0].set_xlabel('$A_{AT} (m)$', fontsize=16)
        axes3[2, 0].set_xticks([-2, 0, 2])
        axes3[2, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[2, 0].grid(True)
        axes3[2, 0].set_xlim([-2, 2.])
        axes3[2, 0].set_ylim([0, 1.05])

        axes3[2, 1].set_xticklabels([])
        axes3[2, 1].set_yticklabels([])



        data = river_discharge['Q']
        ecdf = empirical_distributions.ecdf_histogram(data)
        # epdf = empirical_distributions.epdf_histogram(data, bins=bins)
        axes3[2, 2].plot(ecdf.index, ecdf, color='tab:orange', linestyle=':', lw=3)
        axes3[2, 2].set_ylabel('', fontsize=16)
        axes3[2, 2].set_xlabel('$Q (m^{3}/s)$', fontsize=16)
        axes3[2, 2].set_xticks([0, 250, 500])
        axes3[2, 2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes3[2, 2].set_yticklabels([])
        axes3[2, 2].grid(True)
        axes3[2, 2].set_xlim([0, 500])
        axes3[2, 2].set_ylim([0, 1.05])


    data = wave_wind['hs']
    paso = 0.1
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[0, 0].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = wave_wind['tp']
    paso = 1
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[0, 1].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = wave_wind['dh']
    paso = 12
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[0, 2].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = wave_wind['vv']
    paso = 0.1
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[1, 0].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = wave_wind['dv']
    bins = 8
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[1, 1].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = wave_wind['slp']
    paso = 1
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    epdf = empirical_distributions.epdf_histogram(data, bins=bins)
    axes3[1, 2].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = at_forecast_df['Eta']
    ecdf = empirical_distributions.ecdf_histogram(data)
    axes3[2, 0].plot(ecdf.index, ecdf, color='tab:blue', lw=2)

    data = river_discharge['Q']
    paso = 1
    bins = np.max(data) / (paso * 2.0)
    ecdf = empirical_distributions.ecdf_histogram(data)
    axes3[2, 2].plot(ecdf.index, ecdf, color='tab:blue', lw=2)
    plt.tight_layout()

    fig3.savefig(os.path.join('output', 'analisis', 'graficas', 'ecdf_historico_simulacion.pdf'))
    fig3.savefig(os.path.join('output', 'analisis', 'graficas', 'ecdf_historico_simulacion.png'))