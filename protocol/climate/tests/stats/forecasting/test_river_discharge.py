#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import time
import random
import os
import datetime
import pickle
import pandas as pd
import numpy as np
from future.utils import bytes_to_native_str as n  # to specify str type on both Python 2/3

from tqdm import tqdm

from climate import tests, analysis, read
from climate.stats import extremal, empirical_distributions, fitting
import statsmodels.api as sm
import matplotlib.pyplot as plt
from metoceandataframe.metoceandataframe import MetOceanDF
from preprocessing import missing_values
from climate.stats.forecasting import theoretical_fit
import scipy.stats as st
from input.util import dates
from scipy.stats import exponweib, poisson, expon, norm

from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo import analisis, simulacion
from climate.third_party.simulacion_clima_cobos.clima_maritimo.clima_maritimo.fdist import ajuste_simple, \
    normenr, normtrun, lognorm_genpar2, lognorm2, weib_genpar2, norm_genpar2
from climate.third_party.simulacion_clima_cobos.clima_maritimo.graficas import plot_analisis


def test_river_discharge_simulation():
    # Modules activation and deactivation
    # analysis = False
    # cdf_pdf_representation = False
    # temporal_dependency = False
    # climatic_events_fitting = True
    # threshold_checking_for_simulation = False
    # simulation_cycles = True
    analysis = True
    cdf_pdf_representation = False
    temporal_dependency = False
    climatic_events_fitting = True
    threshold_checking_for_simulation = False
    simulation_cycles = True


    #%% Input data
    # Initial year, number of years, number of valid  data in a year
    anocomienzo, duracion, umbralano = (2018, 10, 0.8)
    # Type of fit (0-GUI, 1-stationary, 2-nonstationary)
    ant = [2]
    # Fourier order for nonstationary analysis
    no_ord_cycles = [2]
    no_ord_calms = [2]
    # Number of simulations
    no_sim = 1
    # Type of fit functions
    fun_cycles = [st.exponweib]
    fun_calms = [st.norm]
    # Number of normals
    no_norm_cycles = [False]
    no_norm_calms = [False]
    f_mix_cycles = [False]
    mod_cycles = [[0, 0, 0, 0]]

    # Cycles River discharge
    threshold_cycles = 25
    # minimum_interarrival_time = pd.Timedelta('250 days')
    # minimum_cycle_length = pd.Timedelta('5 days')
    minimum_interarrival_time = pd.Timedelta('7 days')
    minimum_cycle_length = pd.Timedelta('2 days')

    # Cycles SPEI
    threshold_spei = 0
    minimum_interarrival_time_spei = pd.Timedelta('150 days')
    minimum_cycle_length_spei = pd.Timedelta('150 days')

    interpolation = True
    interpolation_method = 'linear'
    interpolation_freq = '1min'
    truncate = True
    extra_info = True

    #%% Read data
    # Import river discharge data when all dams were active
    data_path = os.path.join(tests.current_path, '..', '..', 'inputadapter', 'tests', 'output', 'modf')
    modf_file_name = 'guadalete_estuary_river_discharge.modf'
    path_name = os.path.join(data_path, modf_file_name)
    modf_rd = MetOceanDF.read_file(path_name)

    # Group into dataframe
    river_discharge = pd.DataFrame(modf_rd)

    # Delete rows where with no common values
    river_discharge.dropna(how='any', inplace=True)

    # Import complete rive discharge historic data
    # All historic river discharge
    data_path = os.path.join(tests.current_path, '..', '..', '..', '..', 'data', 'solar_flux_nao_index_spei')
    modf_file_name = 'caudales.txt'
    path_name = os.path.join(data_path, modf_file_name)
    modf_all = pd.read_table(path_name, header=None, delim_whitespace=True)
    date_col = dates.extract_date(modf_all.iloc[:, 0:4])
    modf_all.index = date_col
    modf_all.drop(modf_all.columns[0:4], axis=1, inplace=True)
    modf_all.columns = ['Q']

    #%% Preprocessing
    t_step = missing_values.find_timestep(river_discharge) # Find tstep
    data_gaps = missing_values.find_missing_values(river_discharge, t_step)
    river_discharge = missing_values.fill_missing_values(river_discharge, t_step, technique='interpolation', method='nearest',
                                                   limit=16*24, limit_direction='both')
    data_gaps_after = missing_values.find_missing_values(river_discharge, t_step)

    # Add noise for VAR
    noise = np.random.rand(river_discharge.shape[0], river_discharge.shape[1])*1e-2
    river_discharge = river_discharge + noise

    # Save_to_pickle
    river_discharge.to_pickle('river_discharge.p')

    # Group into list of dataframes
    df = list()
    df.append(pd.DataFrame(river_discharge['Q']))
    
    #%% Cycles and calms calculation 
    cycles, calm_periods, info = extremal.extreme_events(river_discharge, 'Q', threshold_cycles,
                                                         minimum_interarrival_time,
                                                         minimum_cycle_length, interpolation, interpolation_method,
                                                         interpolation_freq, truncate, extra_info)
    # Calculate duration of the cycles
    dur_cycles = extremal.events_duration(cycles)
    dur_cycles_description = dur_cycles.describe()

    sample_cycles = pd.DataFrame(info['data_cycles'].iloc[:, 0])
    noise = np.random.rand(sample_cycles.shape[0], sample_cycles.shape[1]) * 1e-2
    sample_cycles = sample_cycles + noise

    sample_calms = pd.DataFrame(info['data_calm_periods'])
    noise = np.random.rand(sample_calms.shape[0], sample_calms.shape[1]) * 1e-2
    sample_calms = sample_calms + noise

    #%% CLIMATIC INDICES
    # Sunspots
    data_path = os.path.join(tests.current_path, '..', '..', '..', '..', 'data', 'solar_flux_nao_index_spei')
    modf_file_name = 'sunspot.csv'
    path_name = os.path.join(data_path, modf_file_name)
    sunspot = pd.read_csv(path_name, header=None, delim_whitespace=True, parse_dates=[[0, 1]], index_col=0)
    sunspot = sunspot.drop([2, 4, 5], axis=1)

    # SPEI
    data_path = os.path.join(tests.current_path, '..', '..', '..', '..', 'data', 'solar_flux_nao_index_spei')
    modf_file_name = 'spei_cadiz.csv'
    path_name = os.path.join(data_path, modf_file_name)
    spei = pd.read_csv(path_name, sep=',')
    spei.index = sunspot.index[2412:3233]

    # Calculate cycles over SPEI
    spei = pd.DataFrame(spei.loc[:, 'SPEI_12'] * 100).dropna()
    cycles_spei, calm_periods_spei, info_spei = extremal.extreme_events(spei, 'SPEI_12', threshold_spei,
                                                                        minimum_interarrival_time_spei,
                                                                        minimum_cycle_length_spei, interpolation,
                                                                        interpolation_method,
                                                                        interpolation_freq, truncate, extra_info)
    peaks_over_thres_spei = extremal.events_max(cycles_spei)

    # Plot peaks
    peaks_over_thres = extremal.events_max(cycles)

    # Represent cycles
    fig1 = plt.figure(figsize=(20, 20))
    ax = plt.axes()
    ax.plot(river_discharge)
    ax.axhline(threshold_cycles, color='lightgray')
    ax.plot(spei.loc[:, 'SPEI_12'] * 100, color='0.75', linewidth=2)
    # Plot cycles
    # for cycle in cycles_all:
    #     ax.plot(cycle, 'sandybrown', marker='.', markersize=5)
    #     # ax.plot(cycle.index[0], cycle[0], 'gray', marker='.', markersize=10)
    #     # ax.plot(cycle.index[-1], cycle[-1], 'black', marker='.', markersize=10)
    for cycle in cycles:
        ax.plot(cycle, 'g', marker='.', markersize=5)
        # ax.plot(cycle.index[0], cycle[0], 'gray', marker='.', markersize=10)
        # ax.plot(cycle.index[-1], cycle[-1], 'black', marker='.', markersize=10)
    for cycle in cycles_spei:
        ax.plot(cycle, 'k', marker='.', markersize=5, linewidth=2)
        ax.plot(cycle.index[0], cycle[0], 'gray', marker='.', markersize=15)
        ax.plot(cycle.index[-1], cycle[-1], 'black', marker='.', markersize=15)
    ax.plot(peaks_over_thres, '.r', markersize=15)
    ax.plot(peaks_over_thres_spei, '.c', markersize=15)
    ax.grid()
    ax.set_xlim([datetime.date(1970, 01, 01), datetime.date(2018, 04, 11)])
    ax.set_ylim([-5, 500])
    fig1.savefig(os.path.join('output', 'analisis', 'graficas', 'ciclos_river_discharge_spei.png'))

    #%% # ANALISIS CLIMATICO (0: PARA SALTARLO, 1: PARA HACERLO; LO MISMO PARA TODOS ESTOS IF)
    if analysis:
        if cdf_pdf_representation:
            for i in range(len(df)):
                # DIBUJO LAS CDF Y PDF DE LOS REGISTROS
                plot_analisis.cdf_pdf_registro(df[i], df[i].columns[0])
                plt.pause(0.5)

        #%%  THEORETICAL FIT CYCLES
        data_cycles = sample_cycles['Q']

        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data_cycles)
        # Fit the variable to an extremal distribution
        (param, x, cdf_expwbl, pdf_expwbl) = theoretical_fit.fit_distribution(data_cycles,
                                                                              fit_type=fun_cycles[0].name,
                                                                              x_min=min(data_cycles),
                                                                              x_max=2 * max(data_cycles),
                                                                              n_points=1000)
        par0_cycles = list()
        par0_cycles.append(np.asarray(param))
        # GUARDO LOS PARAMETROS
        np.save(os.path.join('output', 'analisis', 'parameter_river_discharge_cycles.npy'), par0_cycles)

        # Check the goodness of the fit
        fig1 = plt.figure(figsize=(20, 20))
        ax = plt.axes()
        ax.plot(ecdf.index, ecdf, '.')
        ax.plot(x, cdf_expwbl)
        ax.set_xlabel('Q (m3/s)')
        ax.set_ylabel('CDF')
        ax.legend(['ECDF', 'Exponweib Fit', ])
        ax.grid()
        ax.set_xlim([0, 500])
        fig1.savefig(os.path.join('output', 'analisis', 'graficas', 'cdf_fit_ciclos_river_discharge.png'))

        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf_expwbl, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf_expwbl, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf_expwbl, ecdf, river_discharge, 'Q', x, yppplot_emp, yqqplot_emp,
                                             yppplot_teo, yqqplot_teo)

        # Non-stationary fit for calms
        par_cycles, mod_cycles, f_mix_cycles, data_graph_cycles = list(), list(), list(), list()
        df = list()
        df.append(data_cycles)
        for i in range(len(df)):
            # SE HAN SELECCIONADO LOS ULTIMOS 7 ANOS PARA QUE EL ANALISIS SEA MAS RAPIDO
            analisis_ = analisis.analisis(df[i], fun_cycles[i], ant[i], ordg=no_ord_cycles[i], nnorm=no_norm_cycles[i],
                                          par0=par0_cycles[i])

            par_cycles.append(analisis_[0])
            mod_cycles.append(analisis_[1])
            f_mix_cycles.append(analisis_[2])

            aux = list(analisis_[3])
            aux[5] = i
            aux = tuple(aux)
            data_graph_cycles .append(aux)

            # DIBUJO LOS RESULTADOS (HAY UNA GRAN GAMA DE FUNCIONES DE DIBUJO; VER MANUAL)
            plot_analisis.cuantiles_ne(*data_graph_cycles[i])
            plt.pause(0.5)



        fig2 = plt.figure(figsize=(20, 20))
        plt.plot(x, pdf_expwbl)
        _ = plt.hist(data_cycles, bins=np.linspace(0, 500, 100), normed=True, alpha=0.5);
        plt.xlim([0, 400])
        fig2.savefig(os.path.join('output', 'analisis', 'graficas', 'pdf_fit_ciclos_river_discharge.png'))

        # %%  THEORETICAL FIT CALMS
        param0_calms = list()
        data_calms = sample_calms['Q']
        (param, x, cdf, pdf) = theoretical_fit.fit_distribution(data_calms, fit_type=fun_calms[0].name,
                                                                x_min=np.min(data_calms),
                                                                x_max=1.1 * np.max(data_calms), n_points=1000)
        param0_calms.append(np.asarray(param))
        # Empirical cdf
        ecdf = empirical_distributions.ecdf_histogram(data_calms)
        epdf = empirical_distributions.epdf_histogram(data_calms, bins=0)
        # PP - Plot values
        (yppplot_emp, yppplot_teo) = theoretical_fit.pp_plot(x, cdf, ecdf)
        # QQ - Plot values
        (yqqplot_emp, yqqplot_teo) = theoretical_fit.qq_plot(x, cdf, ecdf)
        # Plot Goodness of fit
        theoretical_fit.plot_goodness_of_fit(cdf, ecdf, sample_calms, 'Q', x, yppplot_emp, yqqplot_emp, yppplot_teo,
                                             yqqplot_teo)

        # Non-stationary fit for calms
        par_calms, mod_calms, f_mix_calms, data_graph_calms = list(), list(), list(), list()
        df = list()
        df.append(data_calms)
        for i in range(len(df)):
            # SE HAN SELECCIONADO LOS ULTIMOS 7 ANOS PARA QUE EL ANALISIS SEA MAS RAPIDO
            analisis_ = analisis.analisis(df[i], fun_calms[i], ant[i], ordg=no_ord_calms[i], nnorm=no_norm_calms[i],
                                          par0=param0_calms[i])

            par_calms.append(analisis_[0])
            mod_calms.append(analisis_[1])
            f_mix_calms.append(analisis_[2])
            data_graph_calms.append(analisis_[3])

            # DIBUJO LOS RESULTADOS (HAY UNA GRAN GAMA DE FUNCIONES DE DIBUJO; VER MANUAL)
            plot_analisis.cuantiles_ne(*data_graph_calms[i])
            plt.pause(0.5)

        # Guardo parametros
        np.save(os.path.join('output', 'analisis', 'parameter_river_discharge_calms.npy'), par_calms)
        np.save(os.path.join('output', 'analisis', 'mod_river_discharge_calms.npy'), mod_calms)
        np.save(os.path.join('output', 'analisis', 'f_mix_river_discharge_calms.npy'), f_mix_calms)

    #%% TEMPORAL DEPENDENCY
    if temporal_dependency:
        # SE UTILIZAN LOS PARAMETROS DE SALIDA DEL ANÁLISIS PREVIO
        # Lectura de datos
        par_cycles = np.load(os.path.join('output', 'analisis', 'parameter_river_discharge_cycles.npy'))
        par_calms = np.load(os.path.join('output', 'analisis', 'parameter_river_discharge_calms.npy'))
        mod_calms = np.load(os.path.join('output', 'analisis', 'mod_river_discharge_calms.npy'))
        f_mix_calms = np.load(os.path.join('output', 'analisis', 'f_mix_river_discharge_calms.npy'))

        (df_dt_cycles, cdf_) = analisis.dependencia_temporal(sample_cycles, par_cycles, mod_cycles, no_norm_cycles,
                                                             f_mix_cycles, fun_cycles)

        # SE GUARDAN LOS PARAMETROS DEL MODELO VAR
        df_dt_cycles.to_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_river_discharge_cycles.p'))

        (df_dt_calms, cdf_) = analisis.dependencia_temporal(sample_calms, par_calms, mod_calms, no_norm_calms,
                                                            f_mix_calms, fun_calms)

        # SE GUARDAN LOS PARAMETROS DEL MODELO VAR
        df_dt_calms.to_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_river_discharge_calms.p'))

    if climatic_events_fitting:
        #%% FIT NUMBER OF EVENTS DURING WET CYCLES
        events_wet_cycle = pd.Series([5, 2, 1, 3, 2, 2, 0, 6, 1])
        ecdf_events_wet_cycle = empirical_distributions.ecdf_histogram(events_wet_cycle)

        mu = np.mean(events_wet_cycle)
        simulated_number_events = pd.Series(poisson.rvs(mu, loc=0, size=100, random_state=None))
        ecdf_simulated_events_wet_cycle = empirical_distributions.ecdf_histogram(simulated_number_events)
        x_poisson = np.linspace(0, 10, 100)
        cdf_poisson = poisson.cdf(x_poisson, mu, loc=0)

        plt.figure()
        ax = plt.axes()
        ax.plot(ecdf_events_wet_cycle.index, ecdf_events_wet_cycle, '.')
        ax.plot(ecdf_simulated_events_wet_cycle.index, ecdf_simulated_events_wet_cycle, '.')
        ax.plot(x_poisson, cdf_poisson)
        ax.legend(['ECDF', 'ECDF Sim', 'Poisson Fit'])
        ax.grid()

        #%% FIT TIME BETWEEN WET CYCLES
        t_wet_cycles = peaks_over_thres_spei.index.to_series().diff().dropna().astype('m8[s]').astype(np.float32)
        ecdf_t_wet_cycle = empirical_distributions.ecdf_histogram(t_wet_cycles)

        norm_param = norm.fit(t_wet_cycles, loc=0)
        simulated_t_wet_cycles = pd.Series(norm.rvs(*norm_param, size=100, random_state=None))
        ecdf_simulated_t_wet_cycles = empirical_distributions.ecdf_histogram(simulated_t_wet_cycles)
        x_norm = np.linspace(0, 2*max(t_wet_cycles), 100)
        cdf_norm = norm.cdf(x_norm, *norm_param)

        plt.figure()
        ax = plt.axes()
        ax.plot(ecdf_t_wet_cycle.index, ecdf_t_wet_cycle, '.')
        ax.plot(ecdf_simulated_t_wet_cycles.index, ecdf_simulated_t_wet_cycles, '.')
        ax.plot(x_norm, cdf_norm)
        ax.legend(['ECDF', 'ECDF Sim', 'Exponential Fit'])
        ax.grid()

        simulated_t_wet_cycles_days = simulated_t_wet_cycles.astype('m8[s]')
        # Elimino valores negativos
        simulated_t_wet_cycles_days = simulated_t_wet_cycles_days[
            simulated_t_wet_cycles_days.values > datetime.timedelta(days=1)]

        #%% FIT TIME BETWEEN EVENTS DURING WET CYCLES
        t_between_events = peaks_over_thres.index.to_series().diff().dropna()
        t_between_events = t_between_events[t_between_events < datetime.timedelta(days=400)]
        t_between_events = t_between_events.astype('m8[s]').astype(np.float32)
        ecdf_t_between_events = empirical_distributions.ecdf_histogram(t_between_events)

        lambda_par = expon.fit(t_between_events, loc=0)
        simulated_t_between_events = pd.Series(expon.rvs(scale=lambda_par[1], size=100, random_state=None))
        ecdf_simulated_t_between_events = empirical_distributions.ecdf_histogram(simulated_t_between_events)
        x_expon = np.linspace(0, 2*max(t_between_events), 100)
        cdf_expon = expon.cdf(x_expon, scale=lambda_par[1], loc=0)

        plt.figure()
        ax = plt.axes()
        ax.plot(ecdf_t_between_events.index, ecdf_t_between_events, '.')
        ax.plot(ecdf_simulated_t_between_events.index, ecdf_simulated_t_between_events, '.')
        ax.plot(x_expon, cdf_expon)
        ax.legend(['ECDF', 'ECDF Sim', 'Exponential Fit'])
        ax.grid()

        simulated_t_between_events_days = simulated_t_between_events.astype('m8[s]')

        #%% FIT TIME BETWEEN ALL EVENTS
        # Fit time between events (without considering wet cycles) 2 method
        t_between_events_2method = peaks_over_thres.index.to_series().diff().dropna()
        t_between_events_2method = t_between_events_2method.astype('m8[s]').astype(np.float32)
        ecdf_t_between_events_2method = empirical_distributions.ecdf_histogram(t_between_events_2method)

        lambda_par = expon.fit(t_between_events_2method, loc=0)
        simulated_t_between_events_2method = pd.Series(expon.rvs(scale=lambda_par[1], size=100, random_state=None))
        ecdf_simulated_t_between_events_2method = empirical_distributions.ecdf_histogram(
            simulated_t_between_events_2method)
        x_expon = np.linspace(0, 2*np.max(t_between_events_2method), 100)
        cdf_expon = expon.cdf(x_expon, scale=lambda_par[1], loc=0)

        plt.figure()
        ax = plt.axes()
        ax.plot(ecdf_t_between_events_2method.index, ecdf_t_between_events_2method, '.')
        ax.plot(ecdf_simulated_t_between_events_2method.index, ecdf_simulated_t_between_events_2method, '.')
        ax.plot(x_expon, cdf_expon)
        ax.legend(['ECDF', 'ECDF Sim', 'Exponential Fit'])
        ax.grid()

        simulated_t_between_events_2method_days = simulated_t_between_events.astype('m8[s]')
        # nul_values = simulated_t_between_events_2method_days.values > datetime.timedelta(days=2000)

    #%% SIMULACION CLIMÁTICA CHEQUEO UMBRAL OPTIMO PARA AJUSTAR DURACIONES
    if threshold_checking_for_simulation:
        # CARGO PARÁMETROS
        par_cycles = np.load(os.path.join('output', 'analisis', 'parameter_river_discharge_cycles.npy'))
        df_dt_cycles = pd.read_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_river_discharge_cycles.p'))
        vars_ = ['Q']

        # Cargo el SPEI Index para ajustar tiempo entre ciclos humedos, numero de eventos por ciclo humedo
        # tiempo entre eventos dentro de ciclo humedo

        # Figura de las cdf y pdf empiricas
        fig1, axes1 = plt.subplots(1, 2, figsize=(20, 7))

        cont = 0
        iter = 0
        while cont < no_sim:
            df_sim = simulacion.simulacion(anocomienzo, duracion, par_cycles, mod_cycles, no_norm_cycles, f_mix_cycles,
                                           fun_cycles, vars_, sample_cycles, df_dt_cycles, [0, 0, 0, 0, 0],
                                           semilla=int(np.random.rand(1)*1e6))

            iter += 1

            # Primero filtro si hay valores mayores que el umbral,en cuyo caso descarto la serie
            if np.max(df_sim).values <= np.max(sample_cycles['Q']) * 1.25:
                # Representacion de la serie
                plt.figure()
                ax = plt.axes()
                ax.plot(df_sim)
                ax.plot(sample_cycles, '.')
                ax.plot(df_sim * 0 + max(sample_cycles['Q']), 'r')
                ax.grid()

                # Cdf Pdf
                data = df_sim['Q']
                ecdf = empirical_distributions.ecdf_histogram(data)
                epdf = empirical_distributions.epdf_histogram(data, bins=0)
                axes1[0].plot(epdf.index, epdf, '--', color='0.75')
                axes1[1].plot(ecdf.index, ecdf, '--', color='0.75')

                # Extract cycles from data for different thresholds to fix the duration
                fig2, axes2 = plt.subplots(1, 2, figsize=(20, 7))
                if cont == 0:
                    dur_cycles = dur_cycles.astype('m8[s]').astype(np.float32)  # Convierto a segundos y flotante
                ecdf_dur = empirical_distributions.ecdf_histogram(dur_cycles)
                epdf_dur = empirical_distributions.epdf_histogram(dur_cycles, bins=0)
                axes2[0].plot(epdf_dur.index, epdf_dur, 'r', lw=2)
                axes2[1].plot(ecdf_dur.index, ecdf_dur, 'r', lw=2)

                threshold = np.arange(20, 110, 10)
                color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
                for j, th in enumerate(threshold):
                    minimum_interarrival_time = pd.Timedelta('1 hour')
                    minimum_cycle_length = pd.Timedelta('2 days')
                    cycles, calm_periods, info = extremal.extreme_events(df_sim, 'Q', th,
                                                                         minimum_interarrival_time,
                                                                         minimum_cycle_length, interpolation,
                                                                         interpolation_method,
                                                                         interpolation_freq, truncate, extra_info)

                    # Calculate duration of the cycles
                    dur_cycles_sim = extremal.events_duration(cycles)
                    dur_cycles_sim_description = dur_cycles_sim.describe()

                    # Represent cycles
                    fig3 = plt.figure(figsize=(20, 20))
                    ax = plt.axes()
                    ax.plot(df_sim)
                    ax.axhline(th, color='lightgray')
                    ax.grid()
                    ax.legend(['Threshold: ' + str(th) + ' (m3/s)' + '/ Dur_min ' + str(
                        dur_cycles_description['min']) + ' - ' + str(
                        dur_cycles_sim_description['min']) + '/ Dur_mean ' + str(
                        dur_cycles_description['mean']) + ' - ' + str(
                        dur_cycles_sim_description['mean']) + '/ Dur_max ' + str(
                        dur_cycles_description['max']) + ' - ' + str(dur_cycles_sim_description['max'])])

                    for cycle in cycles:
                        ax.plot(cycle, 'g', marker='.', markersize=5)
                        ax.plot(cycle.index[0], cycle[0], 'gray', marker='.', markersize=10)
                        ax.plot(cycle.index[-1], cycle[-1], 'black', marker='.', markersize=10)
                    ax.set_xlim([datetime.date(2018, 04, 01), datetime.date(2030, 01, 01)])
                    ax.set_ylim([0, 600])

                    fig_name = 'ciclos_sim_' + str(cont) + '_threshold_' + str(th) + '.png'
                    fig3.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial',
                                              'umbral_optimo', fig_name))

                    # Calculate the cdf and pdf of the cycle duration
                    dur_cycles_sim = dur_cycles_sim.astype('m8[s]').astype(np.float32)
                    ecdf_dur_sim = empirical_distributions.ecdf_histogram(dur_cycles_sim)
                    epdf_dur_sim = empirical_distributions.epdf_histogram(dur_cycles_sim, bins=0)
                    axes2[0].plot(epdf_dur_sim.index, epdf_dur_sim, '--', color=color_sequence[j],
                                  label=['Threshold: ' + str(threshold[j])])
                    axes2[1].plot(ecdf_dur_sim.index, ecdf_dur_sim, '--', color=color_sequence[j],
                                  label=['Threshold: ' + str(threshold[j])])
                    axes2[0].legend()
                    axes2[1].set_xlim([0, 5000000])
                    axes2[0].set_xlim([0, 5000000])

                fig_name = 'ciclos_dur_sim_' + str(cont) + '.png'
                fig2.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial',
                                          'umbral_optimo', fig_name))

                cont += 1

            data = sample_cycles['Q']
            ecdf = empirical_distributions.ecdf_histogram(data)
            epdf = empirical_distributions.epdf_histogram(data, bins=0)
            axes1[0].plot(epdf.index, epdf, 'r', lw=2)
            axes1[1].plot(ecdf.index, ecdf, 'r', lw=2)

        fig_name = 'pdf_cdf_descarga_fluvial.png'
        fig1.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial',
                                  'umbral_optimo', fig_name))

    #%% SIMULACION CLIMATICA
    threshold = 50
    minimum_interarrival_time = pd.Timedelta('1 hour')
    minimum_cycle_length = pd.Timedelta('2 days')
    if simulation_cycles:
        # CARGO PARÁMETROS
        par_cycles = np.load(os.path.join('output', 'analisis', 'parameter_river_discharge_cycles.npy'))
        par_calms = np.load(os.path.join('output', 'analisis', 'parameter_river_discharge_calms.npy'))
        mod_calms = np.load(os.path.join('output', 'analisis', 'mod_river_discharge_calms.npy'))
        f_mix_calms = np.load(os.path.join('output', 'analisis', 'f_mix_river_discharge_calms.npy'))

        df_dt_cycles = pd.read_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_river_discharge_cycles.p'))
        df_dt_calms = pd.read_pickle(os.path.join('output', 'dependencia_temporal', 'df_dt_river_discharge_calms.p'))
        vars_ = ['Q']

        # Figura de las cdf y pdf empiricas
        fig2, axes1 = plt.subplots(1, 2, figsize=(20, 7))

        cont = 0
        iter = 0
        while cont < no_sim:
            df_sim = simulacion.simulacion(anocomienzo, duracion, par_cycles, mod_cycles, no_norm_cycles, f_mix_cycles,
                                           fun_cycles, vars_, sample_cycles, df_dt_cycles, [0, 0, 0, 0, 0],
                                           semilla=int(np.random.rand(1)*1e6))

            iter += 1

            # Primero filtro si hay valores mayores que el umbral,en cuyo caso descarto la serie
            if np.max(df_sim).values <= np.max(sample_cycles['Q']) * 1.25:
                df_sim = df_sim.resample('1H').interpolate()

                # Extract cycles from data for different thresholds to fix the duration
                if cont == 0:
                    dur_cycles = dur_cycles.astype('m8[s]').astype(np.float32)  # Convierto a segundos y flotante
                # Calculate cycles
                cycles, calm_periods, info = extremal.extreme_events(df_sim, 'Q', threshold,
                                                                     minimum_interarrival_time,
                                                                     minimum_cycle_length, interpolation,
                                                                     interpolation_method,
                                                                     interpolation_freq, truncate, extra_info)

                # # Represent cycles
                # fig3 = plt.figure(figsize=(20, 20))
                # ax = plt.axes()
                # ax.plot(df_sim)
                # ax.axhline(threshold, color='lightgray')
                # ax.grid()
                #
                # for cycle in cycles:
                #     ax.plot(cycle, 'g', marker='.', markersize=5)
                #     ax.plot(cycle.index[0], cycle[0], 'gray', marker='.', markersize=10)
                #     ax.plot(cycle.index[-1], cycle[-1], 'black', marker='.', markersize=10)
                # ax.set_xlim([datetime.date(2018, 01, 01), datetime.date(2021, 01, 01)])
                # ax.set_ylim([0, 600])
                # fig3.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial',
                #                           'ciclos_cadiz_simulado_' + str(cont).zfill(4) + '.png'))

                # Start to construct the time series
                indices = pd.date_range(start='2018', end='2100', freq='1H')
                df_simulate = pd.DataFrame(np.zeros((len(indices), 1))+25, dtype=float, index=indices, columns=['Q'])

                # The start is in wet cycles
                cont_wet_cicles = 0
                cont_df_events = 1
                t_ini = datetime.datetime(2018, 01, 01)
                t_end = datetime.datetime(2018, 01, 01)
                while t_end < datetime.datetime(2090, 01, 01):
                    if cont_wet_cicles != 0:
                        t_ini = t_end + simulated_t_wet_cycles_days[cont_wet_cicles]
                        year = t_ini.year
                    else:
                        year = 2018

                    # Select the number of events during wet cycle
                    n_events = simulated_number_events[cont_wet_cicles] - 1
                    cont_wet_cicles += 1

                    if n_events != 0:

                        # for j in range(0, n_events):
                        cont_df_events_in_wet_cycles = 0
                        while cont_df_events_in_wet_cycles <= n_events:
                            if cont_df_events_in_wet_cycles != 0:
                                # Time between events
                                year = year + 1

                            # Select the event
                            cycle = cycles[cont_df_events]

                            if np.max(cycle) >= 150:
                                # Simulate date
                                month1 = [random.randint(1, 3), random.randint(10, 12)]
                                rand_pos = random.randint(0, 1)
                                month = month1[rand_pos]
                                day = random.randint(1, 28)
                                hour = random.randint(0, 23)
                            else:
                                # Simulate date
                                month = random.randint(1, 12)
                                day = random.randint(1, 28)
                                hour = random.randint(0, 23)
                            t_ini = datetime.datetime(year, month, day, hour)
                            pos_ini = np.where(df_simulate.index == t_ini)[0][0]
                            pos_end = pos_ini + cycle.shape[0]

                            # Insert cycle
                            df_simulate.iloc[pos_ini:pos_end, 0] = cycle.values
                            t_end = df_simulate.index[pos_end]
                            year = df_simulate.index[pos_end].to_datetime().year
                            cont_df_events += 1
                            cont_df_events_in_wet_cycles += 1

                    else:
                        t_end = t_ini

                # Simulation of calm periods
                df_sim_calms = simulacion.simulacion(anocomienzo, 85, par_calms, mod_calms, no_norm_calms,
                                                    f_mix_calms, fun_calms, vars_, sample_calms, df_dt_calms,
                                                    [0, 0, 0, 0, 0], semilla=int(np.random.rand(1) * 1e6))
                
                # Remove negative values
                df_sim_calms[df_sim_calms < 0] = np.random.randint(1, 5)
                
                # Combine both dataframes with cycles and calms
                pos_cycles = df_simulate >= 50
                df_river_discharge = df_sim_calms
                df_river_discharge[pos_cycles] = df_simulate

                # Hourly interpolation
                df_river_discharge = df_river_discharge.resample('H').interpolate()
                
                # Representation of results
                fig1 = plt.figure(figsize=(20, 10))
                ax = plt.axes()
                ax.plot(river_discharge)
                ax.plot(df_river_discharge)
                ax.legend('Hindcast', 'Forecast')
                ax.grid()
                ax.set_ylim([-5, 500])
                fig1.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial', 
                                          'descarga_fluvial_cadiz_simulado_' + str(cont).zfill(4) + '.png'))

                # Cdf Pdf
                data = df_river_discharge['Q']
                ecdf = empirical_distributions.ecdf_histogram(data)
                epdf = empirical_distributions.epdf_histogram(data, bins=0)
                axes1[0].plot(epdf.index, epdf, '--', color='0.75')
                axes1[1].plot(ecdf.index, ecdf, '--', color='0.75')

                # Guardado de ficheros
                df_river_discharge.to_csv(os.path.join('output', 'simulacion', 'series_temporales', 'descarga_fluvial_500',
                                          'descarga_fluvial_guadalete_sim_' + str(cont).zfill(4) + '.txt'),
                                          sep=n(b'\t'))
                cont += 1

        data = river_discharge['Q']
        ecdf = empirical_distributions.ecdf_histogram(data)
        epdf = empirical_distributions.epdf_histogram(data, bins=0)
        axes1[0].plot(epdf.index, epdf, 'r', lw=2)
        axes1[1].plot(ecdf.index, ecdf, 'r', lw=2)
        fig_name = 'pdf_cdf_descarga_fluvial.png'
        fig2.savefig(os.path.join('output', 'simulacion', 'graficas', 'descarga_fluvial', fig_name))


