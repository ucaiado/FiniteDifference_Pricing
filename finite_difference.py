#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the explicit finite difference method to calculate the price of
various options

@author: ucaiado

Created on 07/03/2016
"""

# import libraries
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import time
from scipy.optimize import minimize

'''
Begin help functions
'''


class STABILITY_ERROR(Exception):
    '''
    STABILITY_ERROR is raised by the init method of the Grid class
    '''
    pass


class UNSUCCESSFUL_ERROR(Exception):
    '''
    UNSUCCESSFUL_ERROR is raised by the static hedhing minimization
    '''
    pass


def get_d1_and_d2(f_St, f_sigma, f_time, f_r, f_K):
    '''
    Calculate the d1 and d2 parameter used in Digital and call options
    '''
    f_d2 = (np.log(f_St/f_K) - (f_r - 0.5 * f_sigma ** 2)*f_time)
    f_d2 /= (f_sigma * f_time**0.5)
    f_d1 = f_d2 + f_sigma*f_time**0.5
    return f_d1, f_d2


def bilinear_interpolation(f_S, f_time, df):
    '''
    Get information from simulations matrix using bilinear interpolation
    :param f_S: float. asset price
    :param f_time: float. time in years
    :param df: dataframe. information to be interpolated
    '''
    # encontro linhas e colunas relevantes
    f_col1 = df.columns[df.columns < f_time][-1]
    f_col2 = df.columns[df.columns >= f_time][0]
    f_row1 = df.index[df.index < f_S][-1]
    f_row2 = df.index[df.index >= f_S][0]
    # defino pontos e areas
    l_V = [df.loc[f_row1, f_col1], df.loc[f_row1, f_col2],
           df.loc[f_row2, f_col2], df.loc[f_row2, f_col1]]
    l_A = [(f_row2 - f_S) * (f_col2 - f_time),
           (f_row2 - f_S) * (f_time - f_col1),
           (f_S - f_row1) * (f_time - f_col1),
           (f_S - f_row1) * (f_col2 - f_time)]
    # interpolo valores
    return sum(np.array(l_V)*np.array(l_A))/sum(np.array(l_A))


'''
End help functions
'''


class GridNode(object):
    '''
    A representation of a Node of a Grid
    '''
    def __init__(self, i, k):
        '''
        Initialize a GridNode object
        :param k: integer. the time index
        :param i: integer. the asset index
        '''
        # inicia variaveis de controle
        self.i = i  # linhas sao os passos do asset
        self.k = k  # colunas sao os passos no tempo

        self.node_idx = '{:.0f},{:.0f}'.format(i, k)
        # inicia variaveis para precificacao
        self.f_asset_value = 0
        self.f_option_value = 0
        self.f_delta = 0
        self.f_gamma = 0
        self.f_theta = 0
        # inicia variaveis para guardar valores analiticos
        self.f_option_value_anlt = 0
        self.f_delta_anlt = 0
        self.f_gamma_anlt = 0

    def __str__(self):
        '''
        Return node_idx
        '''
        return self.node_idx

    def __repr__(self):
        '''
        Return the node_idx
        '''
        return self.node_idx

    def __eq__(self, other):
        '''
        Return if a node has different node_idx from the other
        :param other: node object. Node to be compared
        '''
        return self.node_idx == other.node_idx

    def __ne__(self, other):
        '''
        Return if a node has the same node_idx from the other
        :param other: node object. Node to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the node object be used as a key in a hash
        table
        '''
        return self.node_idx.__hash__()


class Grid(object):
    '''
    A general representation of a Grid to be used by Derivative classes in the
    discretization of their domains
    '''
    def __init__(self, f_vol, f_value, f_time, i_nas, i_nts=None):
        '''
        Initialize a Grid object. Save all parameters as attributes
        :param f_vol: float. Volatility of the underlying instrument
        :param f_val: float. The reference value to calculate the grid length
        :param f_time: float. time to be used in the grid
        :param i_nas: integer. Number of asset steps
        :*param i_nts: integer. Number of time steps
        '''
        # inicia variaveis e usa vol para garantir estabilidade
        self.f_nas = 1. * i_nas
        # 'infinito' eh duas vezes o valor
        self.dS = 2 * f_value / self.f_nas
        # como o wilmott garantiu estabilidade
        self.dt = 0.9 / f_vol**2. / self.f_nas**2.
        self.i_nts = int(f_time/self.dt) + 1
        if i_nts:
            if i_nts <= self.i_nts:
                self.i_nts = i_nts-1
            else:
                s_err = 'The maximum of time steps is {}'
                raise STABILITY_ERROR(s_err.format(self.i_nts))
        self.dt = f_time / (self.i_nts * 1.)
        # inicia grid. O ponto do tempo inicial eh o final, na verdade
        self.grid = {}
        for k in xrange(int(self.i_nts) + 1):
            for i in xrange(int(self.f_nas) + 1):
                node = GridNode(i, k)
                self.grid[node] = node

    def __call__(self, i, k):
        '''
        Allow direct access to the nodes of the object
        :param k: integer. the time index
        :param i: integer. the asset index
        '''
        node_idx = GridNode(i, k)
        return self.grid[node_idx]

    def __str__(self):
        '''
        A string representation of the node
        '''
        s_aux = ''
        df_rtn = pd.DataFrame(np.zeros([int(self.f_nas),
                                        int(self.i_nts)]))
        for k in xrange(int(self.i_nts) + 1):
            for i in xrange(int(self.f_nas) + 1):
                valid_node = self(i, k)
                df_rtn.ix[i, k] = valid_node
        return str(df_rtn)


class Derivative(object):
    '''
    A general representation of a Derivative contract.
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K=None,
                 i_nts=None, f_sigmam=None):
        '''
        Initialize a Derivative object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :*param f_K: float. The strike, if applyable
        :*param i_nas: integer. Number of time steps
        :*param f_sigmam: float. The minimum volatility observed. If it is set
            fs_sigma is the maximum volatility observed
        '''
        # inicia variaveis
        self.s_name = "General"
        self.f_St = f_St
        self.f_K = f_K
        self.f_r = f_r
        self.f_sigma = f_sigma
        self.use_UV = False
        if f_sigmam:
            self.f_sigmaM = f_sigma
            self.f_sigmam = f_sigmam
            self.f_sigma = (f_sigma + f_sigmam)/2.
            self.use_UV = True
        self.f_time = f_time
        # inica grid
        self.grid = Grid(f_vol=f_sigma,
                         f_value=f_St,
                         f_time=f_time,
                         i_nas=i_nas,
                         i_nts=i_nts)

    def get_information(self, f_S, f_time, s_info):
        '''
        :param f_S: float. asset price
        :param f_time: float. time in years
        :param s_info: string. information desired. delta, gamma, price,
              delta_anlt, gamma_anlt, price_anlt
        '''
        # define dataframe desejado
        if s_info == 'price':
            df = self.df_opt_prices
        elif s_info == 'price_anlt':
            df = self.df_opt_prices_anlt
        elif s_info == 'delta':
            df = self.df_delta
        elif s_info == 'delta_anlt':
            df = self.df_delta_anlt
        elif s_info == 'gamma':
            df = self.df_gamma
        elif s_info == 'gamma_anlt':
            df = self.df_gamma_anlt
        # interpola informacao
        return bilinear_interpolation(f_S, f_time, df)

    def compare_to_analytical_solutions(self, l_S, f_time):
        '''
        Plot charts comparing the price, delta and gamma measure by the finitte
        difference and by the analytical solution
        l_S: list. asset price list
        f_time. float. the time step to measure the outputs
        '''
        d_price = {u'analítico': [], u'diferenças finitas': []}
        d_delta = {u'analítico': [], u'diferenças finitas': []}
        d_gamma = {u'analítico': [], u'diferenças finitas': []}
        l_prices = l_S
        for f_S in l_prices:
            # calcula precos
            f_aux = self.get_information(f_S, f_time, 'price_anlt')
            d_price[u'analítico'].append(f_aux)
            f_aux = self.get_information(f_S, f_time, 'price')
            d_price[u'diferenças finitas'].append(f_aux)
            # calcula delta
            f_aux = self.get_information(f_S, f_time, 'delta_anlt')
            d_delta[u'analítico'].append(f_aux)
            f_aux = self.get_information(f_S, f_time, 'delta')
            d_delta[u'diferenças finitas'].append(f_aux)
            # calcula gamma
            f_aux = self.get_information(f_S, f_time, 'gamma_anlt')
            d_gamma[u'analítico'].append(f_aux)
            f_aux = self.get_information(f_S, f_time, 'gamma')
            d_gamma[u'diferenças finitas'].append(f_aux)
        # plota resultados
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        fig.set_size_inches(12, 4)

        l_title = [u'Preços\n', u'$\Delta$\n', u'$\Gamma$\n']
        for d_aux, ax, s_title in zip([d_price, d_delta, d_gamma],
                                      [ax1, ax2, ax3], l_title):

            s_col = u'diferenças finitas'
            df_plot = pd.DataFrame(d_aux[s_col], index=l_prices)
            df_plot.columns = [s_col]
            df_plot.plot(ax=ax)
            s_col = u'analítico'
            df_plot = pd.DataFrame(d_aux[s_col], index=l_prices)
            df_plot.columns = [s_col]
            df_plot.plot(style='--', ax=ax)

            # df_plot =  pd.DataFrame(d_aux, index=l_prices)
            # df_plot.plot(ax=ax)
            ax.set_xlabel(u'Preço do Subjacente')
            ax.set_title(s_title)

        ax1.set_ylabel(u'Valor')
        s_prep = u"Comparação de Resultados para {}\n"
        fig.suptitle(s_prep.format(self.s_name), fontsize=16, y=1.03)
        fig.tight_layout()

    def _set_final_condition(self):
        '''
        Set up the final condition in the grid, the payoff
        '''
        # apenas o valor final do ativo eh necessario aqui
        for i in xrange(int(self.grid.f_nas) + 1):
            f_S = i * 1. * self.grid.dS
            self.grid(i, 0).f_asset_value = f_S
            self.grid(i, 0).f_option_value = self._get_payoff(f_S)
            self.grid(i, 0).f_option_value_anlt = self._get_payoff(f_S)
        # preencho ultimo valor de todas as colunas (tempo)
        for j in xrange(int(self.grid.i_nts) + 1):
            f_S = i * 1. * self.grid.dS
            self.grid(i, j).f_asset_value = f_S

    def _set_all_matrix(self):
        '''
        Create attributes to hold the get_matrix information
        '''
        d_rtn = self._get_matrix()
        self.df_asset_prices = d_rtn['asset']
        self.df_opt_prices = d_rtn['opt_prices']
        self.df_delta = d_rtn['delta']
        self.df_gamma = d_rtn['gamma']
        self.df_theta = d_rtn['theta']
        self.df_opt_prices_anlt = d_rtn['opt_prices_anlt']
        self.df_delta_anlt = d_rtn['delta_anlt']
        self.df_gamma_anlt = d_rtn['gamma_anlt']

    def _go_backwards(self):
        '''
        work backwards in time to calculate the option value
        '''
        # inicia variaveis que serao utilizadas
        dS = self.grid.dS
        dt = self.grid.dt
        f_r = self.f_r
        f_vol = self.f_sigma
        i_nas = int(self.grid.f_nas)
        # seta condicao final
        self._set_final_condition()
        # comeco o loop depois do primeiro passo de cada dimensao
        for k in xrange(1, int(self.grid.i_nts) + 1):
            for i in xrange(1, int(self.grid.f_nas)):
                # calcula valores auxiliares
                f_S = i * 1. * dS
                self(i, k).f_asset_value = f_S
                f_V_ip1_km1 = self(i+1, k-1).f_option_value
                f_V_im1_km1 = self(i-1, k-1).f_option_value
                f_V_i_km1 = self(i, k-1).f_option_value
                # calcula gregas por diferenca central
                f_delta = (f_V_ip1_km1 - f_V_im1_km1) / (2. * dS)
                f_gamma = (f_V_ip1_km1 - 2 * f_V_i_km1 + f_V_im1_km1) / (dS**2)
                # treat UV if is set
                f_vol = self.f_sigma
                if self.use_UV:
                    if f_gamma < 0:
                        f_vol = self.f_sigmaM
                    elif f_gamma > 0:
                        f_vol = self.f_sigmam
                # calcula theta Vki−Vk+1iδt
                f_theta = f_r * f_V_i_km1 - f_r * f_S * f_delta
                f_theta -= 0.5 * f_gamma * f_vol**2 * f_S**2
                # guarda as gregas e novo preco
                self(i, k).f_delta = f_delta
                self(i, k).f_gamma = f_gamma
                self(i, k).f_theta = f_theta
                f_option_value = f_V_i_km1 - dt * f_theta
                # aplica exercicio antecipado, se definido
                f_option_value = self._early_exercise(f_option_value, f_S)
                # guarda valor
                self(i, k).f_option_value = f_option_value
                # guarda valores analiticos
                f_price_anlt = self._get_analytical_price(f_S, k*dt)
                f_delta_anlt = self._get_analytical_delta(f_S, k*dt)
                f_gamma_anlt = self._get_analytical_gamma(f_S, k*dt)
                self(i, k).f_option_value_anlt = f_price_anlt
                self(i, k).f_delta_anlt = f_delta_anlt
                self(i, k).f_gamma_anlt = f_gamma_anlt
            # aplica condicoes de contorno
            f_aux1, f_aux2 = self._apply_boundary_conditions(k)
            self(0, k).f_option_value = f_aux1
            self(i_nas, k).f_option_value = f_aux2
            # guarda valores analiticos
            for i_step in [0, i_nas]:
                f_S = i_step * 1. * dS
                f_price_anlt = self._get_analytical_price(f_S, k*dt)
                f_delta_anlt = self._get_analytical_delta(f_S, k*dt)
                f_gamma_anlt = self._get_analytical_gamma(f_S, k*dt)
                self(i_step, k).f_option_value_anlt = f_price_anlt
                self(i_step, k).f_delta_anlt = f_delta_anlt
                self(i_step, k).f_gamma_anlt = f_gamma_anlt

    def _apply_boundary_conditions(self, k):
        '''
        Apply boundary conditions
        :param k: integer. The time k step
        '''
        # para S = 0
        dt = self.grid.dt
        i_nas = int(self.grid.f_nas)
        f_rtn1 = self(0, k - 1).f_option_value * (1 - self.f_r * dt)
        # para S=inf
        f_rtn2 = 2 * self(i_nas - 1, k).f_option_value
        f_rtn2 -= self(i_nas - 2, k).f_option_value
        return f_rtn1, f_rtn2

    def _get_matrix(self):
        '''
        Return a matrix of prices, deltas, gammas and thetas calculated
        '''
        # cria listas de preco e tempo
        # l_time = ['{:.3f}'.format(self.grid.dt * i)
        #           for i in xrange(int(self.grid.i_nts) + 1)]
        l_time = [self.grid.dt * i for i in xrange(int(self.grid.i_nts) + 1)]
        l_price = [self.grid.dS * i for i in xrange(int(self.grid.f_nas) + 1)]
        # inicia dataframes
        d_rtn = {}
        l_col = ['asset', 'opt_prices', 'delta', 'gamma', 'theta',
                 'opt_prices_anlt', 'delta_anlt', 'gamma_anlt']
        for s_key in l_col:
            d_rtn[s_key] = pd.DataFrame(np.zeros([int(self.grid.f_nas),
                                                  int(self.grid.i_nts)]))

        # extrain informacoes
        for k in xrange(int(self.grid.i_nts) + 1):
            for i in xrange(int(self.grid.f_nas) + 1):
                valid_node = self(i, k)
                d_rtn['asset'].ix[i, k] = valid_node.f_asset_value
                d_rtn['opt_prices'].ix[i, k] = valid_node.f_option_value
                d_rtn['delta'].ix[i, k] = valid_node.f_delta
                d_rtn['gamma'].ix[i, k] = valid_node.f_gamma
                d_rtn['theta'].ix[i, k] = valid_node.f_theta
                f_aux = valid_node.f_option_value_anlt
                d_rtn['opt_prices_anlt'].ix[i, k] = f_aux
                d_rtn['delta_anlt'].ix[i, k] = valid_node.f_delta_anlt
                d_rtn['gamma_anlt'].ix[i, k] = valid_node.f_gamma_anlt

        # arruma index
        for s_key in l_col:
            d_rtn[s_key].index = l_price
            d_rtn[s_key].columns = l_time

        return d_rtn

    def _early_exercise(self, f_value, f_S):
        '''
        Modify the derivative value if it is subject to early exercise
        '''
        return f_value

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        raise NotImplementedError

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        raise NotImplementedError

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        raise NotImplementedError

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        raise NotImplementedError()

    def __call__(self, i, k):
        '''
        Allow direct access to the nodes in the grid
        :param k: integer. the time index
        :param i: integer. the asset index
        '''
        node = self.grid(i, k)
        return node


class EuropianCall(Derivative):
    '''
    A representation of a europian Call Option
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K, i_nts=None,
                 f_sigmam=None):
        '''
        Initialize a EuropianCall object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param f_K: float. The strike
        :*param i_nas: integer. Number of asset steps
        '''
        # inicia variaveis de Derivativo
        super(EuropianCall, self).__init__(f_St=f_St,
                                           f_sigma=f_sigma,
                                           f_time=f_time,
                                           f_r=f_r,
                                           i_nas=i_nas,
                                           f_K=f_K,
                                           i_nts=i_nts,
                                           f_sigmam=f_sigmam)
        self.s_name = 'Call Europeia'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        exp_r_t = np.exp(-self.f_r * self.f_time)
        S_cdf_d1 = f_S * stats.norm.cdf(f_d1, 0., 1.)
        K_cdf_d2 = self.f_K * stats.norm.cdf(f_d2, 0., 1.)

        return S_cdf_d1 - K_cdf_d2 * exp_r_t

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        cdf_d1 = stats.norm.cdf(f_d1, 0., 1.)
        return cdf_d1

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        pdf_d1 = stats.norm.pdf(f_d1, 0., 1.)
        S_gima_sqrt_t = f_S * self.f_sigma * (f_time**0.5)
        return pdf_d1/S_gima_sqrt_t

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        return max(0, f_asset_price - self.f_K)


class LogContract(Derivative):
    '''
    A representation of a Log Contract
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K=None, i_nts=None,
                 f_sigmam=None):
        '''
        Initialize a LogContract object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param f_K: float. The strike
        :*param i_nas: integer. Number of asset steps
        '''
        # inicia variaveis de Derivativo
        super(LogContract, self).__init__(f_St=f_St,
                                          f_sigma=f_sigma,
                                          f_time=f_time,
                                          f_r=f_r,
                                          i_nas=i_nas,
                                          f_K=f_K,
                                          i_nts=i_nts,
                                          f_sigmam=f_sigmam)
        self.s_name = 'Contrato Log'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''

        exp_r_t = np.exp(-1 * self.f_r * f_time)
        ln_S = np.log(f_S)
        r_var_t = (self.f_r - (self.f_sigma**2.)/2.) * f_time
        ln_S_r_var_t = ln_S + r_var_t
        return exp_r_t * ln_S_r_var_t

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        exp_r_t = np.exp(-1. * self.f_r * f_time)
        return exp_r_t / f_S

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        exp_r_t = np.exp(-1. * self.f_r * f_time)
        return -1 * exp_r_t / f_S**2.

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        if f_asset_price == 0:
            return 0.
        return np.log(f_asset_price)


class SquaredLogContract(Derivative):
    '''
    A representation of a Squared Log Contract
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K=None, i_nts=None,
                 f_sigmam=None):
        '''
        Initialize a SquaredLogContract object. Save all parameters as
        attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param f_K: float. The strike
        :*param i_nas: integer. Number of asset steps
        '''
        # inicia variaveis de Derivativo
        super(SquaredLogContract, self).__init__(f_St=f_St,
                                                 f_sigma=f_sigma,
                                                 f_time=f_time,
                                                 f_r=f_r,
                                                 i_nas=i_nas,
                                                 f_K=f_K,
                                                 i_nts=i_nts,
                                                 f_sigmam=f_sigmam)
        self.s_name = 'Contrato Log Quadratico'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        exp_r_t = np.exp(-1*self.f_r*f_time)
        ln_S_r_var_t_sq = (np.log(f_S) + (self.f_r -
                           (self.f_sigma**2.)/2.) * f_time)**2.
        var_t = self.f_sigma**2. * f_time
        return exp_r_t * (ln_S_r_var_t_sq + var_t)

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        two_exp_r_t_over_S = 2 * np.exp(-1*self.f_r*f_time) / f_S
        ln_S_r_var_t = (np.log(f_S) + (self.f_r -
                        (self.f_sigma**2)/2) * f_time)
        return two_exp_r_t_over_S * ln_S_r_var_t

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        ln_S = np.log(f_S)
        r_t = self.f_r * f_time
        if f_S == 0:
            ln_S = 0
        exp_r_t = np.exp(-1. * r_t)
        sigma_sqr_t = self.f_sigma**2 * f_time
        f_rtn = exp_r_t / f_S**2. * (2 + sigma_sqr_t - 2 * ln_S - 2 * r_t)
        return f_rtn

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        if f_asset_price == 0:
            return 0.
        return np.log(f_asset_price) ** 2


class SquaredExotic(Derivative):
    '''
    A representation of a exotic suqared contract. The Strike is given
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K=None, i_nts=None,
                 f_sigmam=None):
        '''
        Initialize a SquaredExotic object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param f_K: float. The strike
        :*param i_nas: integer. Number of asset steps
        '''
        # inicia variaveis de Derivativo
        super(SquaredExotic, self).__init__(f_St=f_St,
                                            f_sigma=f_sigma,
                                            f_time=f_time,
                                            f_r=f_r,
                                            i_nas=i_nas,
                                            f_K=f_K,
                                            i_nts=i_nts,
                                            f_sigmam=f_sigmam)
        self.s_name = 'Exotico Quadratico'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        exp_r_var_t = np.exp((self.f_r + self.f_sigma**2)*f_time)
        S_sq_exp_r_var_t = f_S**2 * exp_r_var_t
        K_sq_exp_r_t = self.f_K**2 * np.exp(-self.f_r * f_time)
        two_S_K = 2 * f_S * self.f_K
        return S_sq_exp_r_var_t - two_S_K + K_sq_exp_r_t

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        exp_r_var_t = np.exp((self.f_r + self.f_sigma**2)*f_time)
        two_K = 2. * self.f_K

        return 2. * f_S * exp_r_var_t - two_K

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        return 2 * np.exp((self.f_r + self.f_sigma**2)*f_time)

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        if f_asset_price == 0:
            return 0.
        return (f_asset_price - self.f_K) ** 2

    def _apply_boundary_conditions(self, k):
        '''
        Apply boundary conditions
        :param k: integer. The time k step
        '''
        # para S = 0
        dt = self.grid.dt
        dS = self.grid.dS
        i_nas = int(self.grid.f_nas)
        f_rtn1 = self(0, k - 1).f_option_value * (1 - self.f_r * dt)
        # para S=inf
        f_rtn2 = 2 * self(i_nas - 1, k).f_option_value
        f_rtn2 -= self(i_nas - 2, k).f_option_value
        # adaptando condicao ao contrato
        f_rtn2 += (2 * dS**2)
        return f_rtn1, f_rtn2


class DigitalOption(Derivative):
    '''
    A representation of a Digital Option.
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, f_K, i_nts=None,
                 f_sigmam=None):
        '''
        Initialize a DigitalOption object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param f_K: float. The strike
        :*param i_nas: integer. Number of asset steps
        '''
        # inicia variaveis de Derivativo
        super(DigitalOption, self).__init__(f_St=f_St,
                                            f_sigma=f_sigma,
                                            f_time=f_time,
                                            f_r=f_r,
                                            i_nas=i_nas,
                                            f_K=f_K,
                                            i_nts=i_nts,
                                            f_sigmam=f_sigmam)
        self.s_name = 'Opcao Digital'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        exp_r_t = np.exp(-self.f_r * f_time)
        cdf_d2 = stats.norm.cdf(f_d2, 0., 1.)
        return exp_r_t * cdf_d2

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        exp_r_t = np.exp(-self.f_r * self.f_time)
        pdf_d2 = stats.norm.pdf(f_d2, 0., 1.)
        sig_S_sqtr_t = self.f_sigma * self.f_St * (self.f_time**0.5)
        return exp_r_t * pdf_d2 / sig_S_sqtr_t

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                   self.f_K)
        exp_r_t = np.exp(-self.f_r * f_time)
        sig_S_sqtr_t = self.f_sigma * f_S * (f_time**0.5)
        cdf_d2 = stats.norm.cdf(f_d2, 0., 1.)
        S_sqr_var_t = f_S**2 * self.f_sigma**2 * f_time
        return (-1. * exp_r_t * f_d1 * cdf_d2) / S_sqr_var_t

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        return 1. * (f_asset_price > self.f_K)


class EuropianCallButterfly(Derivative):
    '''
    A representation of a europian Call Butterfly strategy with legs expiring
    at the same maturity
    '''
    def __init__(self, f_St, f_sigma, f_time, f_r, i_nas, l_K, l_Q,
                 f_sigmam=None):
        '''
        Initialize a EuropianCall object. Save all parameters as attributes
        :param f_St: float. The price of the underline asset
        :param f_sigma: float. A non negative underline volatility
        :param f_time: float. The time remain until the expiration
        :param f_r: float. The free intereset rate
        :param i_nas: integer. Number of asset steps
        :param l_K: list. The strikes of the strategy
        :param l_Q: float. The normalized qtys of the strategy
        :*param f_sigmam: float. The minimum volatility observed. If it is set
            fs_sigma is the maximum volatility observed

        '''
        # inicia variaveis de Derivativo
        self.l_K = l_K
        self.l_Q = l_Q
        super(EuropianCallButterfly, self).__init__(f_St=f_St,
                                                    f_sigma=f_sigma,
                                                    f_time=f_time,
                                                    f_r=f_r,
                                                    i_nas=i_nas,
                                                    f_K=max(l_K),
                                                    i_nts=None,
                                                    f_sigmam=f_sigmam)
        self.s_name = 'Fly de Call Europeia'
        self._go_backwards()
        self._set_all_matrix()

    def _get_analytical_price(self, f_S, f_time):
        '''
        Return the price of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_price = 0.
        for f_q, f_K in zip(self.l_Q, self.l_K):
            f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                       f_K)
            exp_r_t = np.exp(-self.f_r * self.f_time)
            S_cdf_d1 = f_S * stats.norm.cdf(f_d1, 0., 1.)
            K_cdf_d2 = f_K * stats.norm.cdf(f_d2, 0., 1.)
            f_aux = S_cdf_d1 - K_cdf_d2 * exp_r_t
            f_price += (f_aux * f_q)
        return f_price

    def _get_analytical_delta(self, f_S, f_time):
        '''
        Return the delta of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_delta = 0.
        for f_q, f_K in zip(self.l_Q, self.l_K):
            f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                       f_K)
            cdf_d1 = stats.norm.cdf(f_d1, 0., 1.)
            f_aux = cdf_d1
            f_delta += (f_aux * f_q)
        return f_delta

    def _get_analytical_gamma(self, f_S, f_time):
        '''
        Return the gamma of the instrument using its analytical solution
        :param f_S: float. the asset price
        :param f_time: float.time to expiration
        '''
        f_gamma = 0.
        for f_q, f_K in zip(self.l_Q, self.l_K):
            f_d1, f_d2 = get_d1_and_d2(f_S, self.f_sigma, f_time, self.f_r,
                                       f_K)
            pdf_d1 = stats.norm.pdf(f_d1, 0., 1.)
            S_gima_sqrt_t = f_S * self.f_sigma * (f_time**0.5)
            f_aux = pdf_d1/S_gima_sqrt_t
            f_gamma += (f_aux * f_q)
        return f_gamma

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        f_payoff = 0.
        for f_q, f_K in zip(self.l_Q, self.l_K):
            f_payoff += max(0, f_asset_price - f_K)*f_q
        return f_payoff


class Derivative_UVM(object):
    '''
    A general representation of a Derivative contract precifed using UVM
    '''
    def __init__(self, f_volm, f_volM, f_intrate, s_ptype, l_strike, l_qty,
                 f_expiration, f_NAS, b_worstcase=True):
        '''
        Initialize a Derivative_UVM object. Save all parameters as attributes
        :param f_volm: float. The minimum underline volatility to be observed
        :param f_volM: float. The maximum underline volatility to be observed
        :param f_intrate: float. The risk free intereset rate
        :param s_ptype: string. Pur ot Call. 'P' or 'C'
        :param l_strike: list. Strikes in a array (1,3)
        :param l_qty: list. Normalized Qty in a array (1,3)
        :param f_expiration: float. The time remain until the expiration
        :param f_NAS: float. Number of asset steps
        :*param b_worstcase: boolean. if should use the worst or best case
        '''
        self.f_volm = f_volm
        self.f_volM = f_volM
        self.f_intrate = f_intrate
        self.s_ptype = s_ptype
        self.l_strike = l_strike
        self.f_strike = self.l_strike[1]
        self.l_qty = l_qty
        self.f_expiration = f_expiration
        self.f_NAS = f_NAS
        self.b_worstcase = b_worstcase
        self.s_name = 'Generico'
        self.OPTION = None

    def get_information(self, f_S, f_time, s_info):
        '''
        Recover information desired, interpolating values and not available
        :param f_S: float. asset price
        :param f_time: float. time in years
        :param s_info: string. information desired. delta, gamma, price
        '''
        # define dataframe desejado
        if s_info == 'price':
            df = self.df_price
        elif s_info == 'gamma':
            df = self.df_gamma
        elif s_info == 'delta':
            df = self.df_delta
        # interpola informacao
        return bilinear_interpolation(f_S, f_time, df)

    def get_optimized_satic_hedging(self, f_S0, na_V0,
                                    bounds=((-1.5, -0.), (-1.5, -0.))):
        '''
        Return the best Static hedging with traded options to improve prices,
        given boundaries and constrains to thr minimizer
        :param f_S0: numpy array. qtaties of the fisrt and last leg
        :param na_V0: numpy array. qtaties of the fisrt and last leg
        :param bounds : tuple. the boundaries to minimization
        '''
        # define the parameters
        d = {'f_volm': self.f_volm,
             'f_volM': self.f_volM,
             'f_intrate': self.f_intrate,
             's_ptype': self.s_ptype,
             'l_strike': self.l_strike,
             'l_qty': self.l_qty,
             'f_expiration': self.f_expiration,
             'f_NAS': self.f_NAS}
        # define if it is a minimization or maximization problem
        f_max = -1.
        f_middle = 1.
        if self.l_qty[1] < 0:
            f_middle = -1.
            f_max = 1.
            bounds = ((0., 1.5), (0., 1.5))
        # make sure that the sum of legs is <=1.5 and >= 0.3
        cons = ({'type': 'ineq',
                 'fun': lambda x: np.array([1.5 - abs(x[1] + x[0])])},
                {'type': 'ineq',
                 'fun': lambda x: np.array([abs(x[1] + x[0]) - 0.3])})
        # minimize what is desired
        l_x = np.array([-0.5, -0.5]) * f_middle
        res = minimize(self._best_price, l_x, tol=10e-6,
                       args=(f_middle, d, f_S0, na_V0, f_max),
                       bounds=bounds,
                       constraints=cons)
        if not res.success:
            raise UNSUCCESSFUL_ERROR
        l_x = list(res.x)
        return [l_x[0], f_middle, l_x[1]]

    def get_middle_price(self, f_S0, na_V0, f_time):
        '''
        Return the price to be traded at the middle
        :param f_S0: numpy array. price of the spot
        :param na_V0: numpy array. prices of fisrt and third leg
        '''
        # initialize and calculate the prices using OPTION
        na_x = np.array([self.l_qty[0], self.l_qty[2]])
        f_V = self.get_information(f_S0, f_time, 'price')
        f_already_paid = sum(na_V0 * na_x)
        return (f_V - f_already_paid) * self.l_qty[1]

    def plot_all_solutions(self, l_S, f_time=10e-6):
        '''
        Plot charts all solutions obtained by the method to the price, delta
        and gamma measure by the finitte difference using UVM
        :param l_S: list. asset price list
        :*param f_time. float. the time step to measure the outputs
        '''
        d_price = {u'diferenças finitas': []}
        d_delta = {u'diferenças finitas': []}
        d_gamma = {u'diferenças finitas': []}
        l_prices = l_S
        for f_S in l_prices:
            # calcula precos
            f_aux = self.get_information(f_S, f_time, 'price')
            d_price[u'diferenças finitas'].append(f_aux)
            # calcula delta
            f_aux = self.get_information(f_S, f_time, 'delta')
            d_delta[u'diferenças finitas'].append(f_aux)
            # calcula gamma
            f_aux = self.get_information(f_S, f_time, 'gamma')
            d_gamma[u'diferenças finitas'].append(f_aux)
        # plota resultados
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        fig.set_size_inches(12, 4)

        l_title = [u'Preços\n', u'$\Delta$\n', u'$\Gamma$\n']
        for d_aux, ax, s_title in zip([d_price, d_delta, d_gamma],
                                      [ax1, ax2, ax3], l_title):

            s_col = u'diferenças finitas'
            df_plot = pd.DataFrame(d_aux[s_col], index=l_prices)
            df_plot.columns = [s_col]
            df_plot.plot(ax=ax, legend=False)

            # df_plot =  pd.DataFrame(d_aux, index=l_prices)
            # df_plot.plot(ax=ax)
            ax.set_xlabel(u'Preço do Subjacente')
            ax.set_title(s_title)

        ax1.set_ylabel(u'Valor')
        s_prep = u'Soluções Obtidas para {}\n'
        fig.suptitle(s_prep.format(self.s_name), fontsize=16, y=1.03)
        fig.tight_layout()

    def _best_price(self, na_x, f_middle, d_param, f_S0, na_V0, f_max):
        '''
        Return the best price fo the strategy evaluated, minimize the
        quantities
        :param na_x: numpy array. qtaties of the fisrt and last leg
        :param f_middle: float. qty of the midle leg
        :param d_param: dictinary. Parameters of the function
        :param f_S0: numpy array. qtaties of the fisrt and last leg
        :param na_V0: numpy array. qtaties of the fisrt and last leg
        :param d_opt_param: dictionary. parameters to be used in option eval
        :param f_max: float. -1 to find the maximum and 1 to find the minimum
        '''
        # define the parameters
        d = d_param.copy()
        # redefine qtties
        l_qty = list(na_x)[:]
        l_qty.insert(1, f_middle)
        d['l_qty'] = l_qty
        # initialize and calculate the prices using OPTION
        my_uvmoption = self.OPTION(**d)
        f_V = my_uvmoption.get_information(f_S0, d['f_expiration'], 'price')
        f_already_paid = sum(na_V0 * np.array(na_x))
        return (f_V - f_already_paid) * f_middle * f_max

    def _early_exercise(self, na_V, na_payoff):
        '''
        Modify the derivative value if it is subject to early exercise
        :param na_V: numpy array. Option prices
        :param na_payoff: numpy array. Payoffs
        '''
        return na_V

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        raise NotImplementedError()

    def _go_backwards(self):
        '''
        Work backwards in time to calculate the option value. Use Wilmott
        implementation of finite difference method
        '''
        # recover variables
        f_volm = self.f_volm
        f_volM = self.f_volM
        f_intrate = self.f_intrate
        s_ptype = self.s_ptype
        f_expiration = self.f_expiration
        f_NAS = self.f_NAS
        b_worstcase = self.b_worstcase
        # initiate vaiables
        i_NAS = int(f_NAS)
        #   infinity is twice the strike
        dS = 2 * self.f_strike / f_NAS
        #   for stability
        dt = 0.9 / f_volM ** 2 / f_NAS**2
        #   number of time steps
        i_NTS = int(f_expiration/dt) + 1
        dt = f_expiration / float(i_NTS)
        f_sigma = (f_volm+f_volM)/2.

        # initiate the arrays used in solution
        na_S = np.zeros(i_NAS)
        na_payoff = np.zeros(i_NAS)
        na_V = np.zeros([i_NAS, i_NTS])
        na_Delta = np.zeros([i_NAS, i_NTS])
        na_Gamma = np.zeros([i_NAS, i_NTS])

        # define the terminal value
        for i in xrange(i_NAS):
            na_S[i] = i * dS
            f_rtn = self._get_payoff(na_S[i])
            # calculate the payoff at the end of the maturity
            na_V[i][0] = f_rtn
            # store payoff
            na_payoff[i] = na_V[i][0]
        # time loop
        for k in xrange(1, i_NTS):
            # asset loop
            for i in xrange(1, i_NAS-1):
                f_delta = (na_V[i+1][k-1] - na_V[i-1][k-1]) / (2.*dS)
                f_gamma = (na_V[i+1][k-1] - 2 * na_V[i][k-1] + na_V[i-1][k-1])
                f_gamma /= dS**2
                f_vol = f_sigma
                if b_worstcase:
                    # worst case scenario
                    if f_gamma < 0.:
                        f_vol = f_volM
                    elif f_gamma > 0.:
                        f_vol = f_volm
                else:
                    # best case scenario
                    if f_gamma > 0.:
                        f_vol = f_volM
                    elif f_gamma < 0.:
                        f_vol = f_volm
                f_theta = -0.5*f_vol**2 * na_S[i]**2 * f_gamma
                f_theta -= f_intrate*na_S[i]*f_delta
                f_theta += f_intrate * na_V[i][k-1]
                na_Delta[i][k] = f_delta
                na_Gamma[i][k] = f_gamma
                na_V[i][k] = na_V[i][k-1] - dt * f_theta
            # boundaty codition at S=0
            na_V[0][k] = na_V[0][k-1] * (1. - f_intrate * dt)
            # boundary condition at S=infinity
            na_V[i_NAS-1][k] = 2 * na_V[i_NAS-2][k] - na_V[i_NAS-3][k]
            # checking for early exercise
            na_V.T[k] = self._early_exercise(na_V.T[k], na_payoff)

        df_price = pd.DataFrame(na_V)
        df_gamma = pd.DataFrame(na_Gamma)
        df_delta = pd.DataFrame(na_Delta)
        for df in [df_price, df_gamma, df_delta]:
            df.index = na_S
            na_time = np.arange(i_NTS) * dt
            na_time[-1] = f_expiration
            df.columns = na_time

        return df_price, df_delta, df_gamma


class EuropianVanillaUvm(Derivative_UVM):
    '''
    A representation of a strategy of europian options that can use
    three legs at most
    '''
    def __init__(self, f_volm, f_volM, f_intrate, s_ptype, l_strike, l_qty,
                 f_expiration, f_NAS):
        '''
        Initiate a EuropianVanillaUvm object. Sale all parametes as attrubutes
        :param f_volm: float. The minimum underline volatility to be observed
        :param f_volM: float. The maximum underline volatility to be observed
        :param f_intrate: float. The risk free intereset rate
        :param s_ptype: string. Pur ot Call. 'P' or 'C'
        :param l_strike: list. Strikes in a array (1,3)
        :param l_qty: list. Normalized Qty in a array (1,3)
        :param f_expiration: float. The time remain until the expiration
        :param f_NAS: float. Number of asset steps
        '''
        super(EuropianVanillaUvm, self).__init__(f_volm=f_volm,
                                                 f_volM=f_volM,
                                                 f_intrate=f_intrate,
                                                 s_ptype=s_ptype,
                                                 l_strike=l_strike,
                                                 l_qty=l_qty,
                                                 f_expiration=f_expiration,
                                                 f_NAS=f_NAS,
                                                 b_worstcase=True)
        s_type = 'Put'
        if s_ptype == 'C':
            s_type = 'Call'
        self.s_name = 'Estrat. com {} Europeia: '.format(s_type)
        self.s_name += '[{:0.1f}, {:0.1f}, {:0.1f}]\n'
        self.s_name = self.s_name.format(*l_qty)
        self.df_price, self.df_delta, self.df_gamma = self._go_backwards()
        self.OPTION = EuropianVanillaUvm

    def _early_exercise(self, na_V, na_payoff):
        '''
        Modify the derivative value if it is subject to early exercise
        '''
        return na_V

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        # test for call or put
        q = 1.
        if self.s_ptype == 'P':
            q = -1.
        # calculate the payoff
        f_rtn = 0.
        for f_qty, f_K in zip(self.l_qty, self.l_strike):
            f_rtn += (max(q * (f_asset_price - f_K), 0) * f_qty)
        return f_rtn


class AmericanVanillaUvm(Derivative_UVM):
    '''
    A representation of a strategy of amarican options that can use
    three legs at most
    '''
    def __init__(self, f_volm, f_volM, f_intrate, s_ptype, l_strike, l_qty,
                 f_expiration, f_NAS):
        '''
        Initiate a AmericanVanillaUvm object. Sale all parametes as attrubutes
        :param f_volm: float. The minimum underline volatility to be observed
        :param f_volM: float. The maximum underline volatility to be observed
        :param f_intrate: float. The risk free intereset rate
        :param s_ptype: string. Pur ot Call. 'P' or 'C'
        :param l_strike: list. Strikes in a array (1,3)
        :param l_qty: list. Normalized Qty in a array (1,3)
        :param f_expiration: float. The time remain until the expiration
        :param f_NAS: float. Number of asset steps
        '''
        super(AmericanVanillaUvm, self).__init__(f_volm=f_volm,
                                                 f_volM=f_volM,
                                                 f_intrate=f_intrate,
                                                 s_ptype=s_ptype,
                                                 l_strike=l_strike,
                                                 l_qty=l_qty,
                                                 f_expiration=f_expiration,
                                                 f_NAS=f_NAS,
                                                 b_worstcase=True)
        s_type = 'Put'
        if s_ptype == 'C':
            s_type = 'Call'
        self.s_name = 'Estrat. com {} Americana: '.format(s_type)
        self.s_name += '[{:0.1f}, {:0.1f}, {:0.1f}]\n'
        self.s_name = self.s_name.format(*l_qty)
        self.df_price, self.df_delta, self.df_gamma = self._go_backwards()
        self.OPTION = AmericanVanillaUvm

    def _early_exercise(self, na_V, na_payoff):
        '''
        Modify the derivative value if it is subject to early exercise
        '''
        return np.amax((np.array([na_V, na_payoff]).T), axis=1)

    def _get_payoff(self, f_asset_price):
        '''
        Get the payoff of the contract
        :param f_asset_price: float. The base asset price
        '''
        # test for call or put
        q = 1.
        if self.s_ptype == 'P':
            q = -1.
        # calculate the payoff
        f_rtn = 0.
        for f_qty, f_K in zip(self.l_qty, self.l_strike):
            f_rtn += (max(q * (f_asset_price - f_K), 0) * f_qty)
        return f_rtn
