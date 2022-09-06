# Todo: imports. Have the curve formulas in a separate file? Meaning need a folder for curve_fitting?
# Todo: In this file also have a .py file that is just a dictionary connecting curve fit formula names to functions
from scipy.optimize import curve_fit as curve_fit_scipy
from importlib import import_module
import numpy as np
import pandas as pd


class CurveFit():

    def __init__(self, growth_df):
        """
        The class is initialized with a dataframe of stand-level biomass growth data that will be used for curve
        fitting.
        Todo: Write the specification of the dataframe

        Parameters
        ----------
        growth_df
        """
        self.growth_df = growth_df

    def set_maxes(self, max_df):
        """
        The growth data will be paired with each maximum biomass in the input dataframe for a separate curve fit.
        Todo: Spec of dataframe; include columns for data source, percentile if applicable (e.g. Walker), and number
        Todo: should this be part of the __init__ function?

        Parameters
        ----------
        max_df
        """
        self.max_df = max_df

    def fit_curve(self, curve_formula, curve_fit_params=None, set_params=None):
        """
        Fit a separate curve for each value of y_max

        Parameters
        ----------
        curve_formula: string, name of function which has curve fit formula. One of
            'chapman_richards_set_ymax'
            'chapman_richards_set_ymax_and_p'

        curve_fit_params: dictionary with keyword parameters for scipy.optimize.curve_fit, other than f, xdata,
        and ydata, which are set here.
        For default arguments see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        set_params: if specifying other parameters, e.g. p of Chapman-Richards

        """
        # Set curve fit params if they aren't input: curve formula, ydata for curve_fit, get age for xdata
        if curve_fit_params is None:
            curve_fit_params = {}

        #Import the curve formula function
        curve_formula_module = import_module(name='.curve_formulas', package='model_utilities.curve_fitting')
        curve_fit_params['f'] = getattr(curve_formula_module, curve_formula)
        #Save for future use
        self.curve_formula = curve_formula

        #Set ydata, get age to create xdata
        curve_fit_params['ydata'] = self.growth_df['agb_bgb_tCO2e_ha']
        age = np.array(self.growth_df['age']).reshape((self.growth_df['age'].shape[0], 1))

        # Lists to hold parameter estimates and covariance matrices for each curve fit
        self.params = []
        self.covars = []

        # Loop through array of y_max values, set xdata, do curve fits and store results
        for row_counter in range(self.max_df.shape[0]):
            # print(self.max_df.iloc[row_counter, 2])
            y_max_array = np.ones_like(age) * self.max_df.iloc[row_counter, 2]
            curve_fit_params['xdata'] = np.concatenate((age, y_max_array), axis=1)
            #Todo:if setting p, need to concatenate another column on here with that value
            # print(curve_fit_params)
            this_params, this_covar = curve_fit_scipy(**curve_fit_params)
            self.params.append(this_params)
            self.covars.append(this_covar)

        #Store function, xdata, and ydata along with other curve fit parameters
        self.curve_fit_params = curve_fit_params

    def predict(self, params, y_max_agb_bgb, years_predict=100):
        """

        Parameters
        ----------
        params
        y_max_agb_bgb
        years_predict

        Returns
        -------

        """
        age_years = np.arange(1, years_predict + 1, 1).reshape((years_predict, 1))
        y_max_array_plot = np.ones_like(age_years) * y_max_agb_bgb
        x_data_plot = np.concatenate((age_years, y_max_array_plot), axis=1)

        if self.curve_formula == 'chapman_richards_set_ymax':
            predictions = self.curve_fit_params['f'](x=x_data_plot, k=params[0], p=params[1])
        elif self.curve_formula == 'chapman_richards_set_ymax_and_p':
            # Todo:if setting p, need to concatenate another column on here with that value
            predictions = self.curve_fit_params['f'](x=x_data_plot, k=params[0])

        return predictions, age_years

    def predictions_with_monte_carlo(self, years_predict=100, n_mc=1000):
        """

        Parameters
        ----------
        years_predict
        n_mc: If > 0, specifies the number of members in the Monte Carlo ensemble for each y_max

        Returns
        -------

        """

        # The Monte Carlo dataframe list will have a dataframe for each y_max, identified with a label
        self.monte_carlo_dfs = {}

        # Loop through array of y_max values, set xdata, do curve fits and store results
        for row_counter in range(self.max_df.shape[0]):
            #Get predictions
            predictions, age_years = self.predict(params=self.params[row_counter],
                                                  y_max_agb_bgb=self.max_df.iloc[row_counter, 2],
                                                  years_predict=years_predict)
            #Create a label to identify the simulation, using the y_max source and detail
            label = self.max_df.iloc[row_counter, 0] + '_' + self.max_df.iloc[row_counter, 1]
            #Append the column of predictions
            if row_counter == 0:
                # Create dataframe to store the predictions from optimal parameter estimates
                # with Age column
                self.prediction_df = pd.DataFrame({'Age': age_years.reshape(age_years.shape[0],)})
            self.prediction_df[label] = predictions

            #Monte Carlo ensemble
            if n_mc > 0:
                # Make Monte Carlo ensemble, get median and 95% CI bounds
                param_sample = np.random.default_rng().multivariate_normal(mean=self.params[row_counter],
                                                                           cov=self.covars[row_counter],
                                                                           size=n_mc).T
                series_list = []
                # print(param_sample.shape)
                for param_ix in range(param_sample.shape[1]):
                    predictions_mc, age_years = self.predict(params=param_sample[:,param_ix],
                                                             y_max_agb_bgb=self.max_df.iloc[row_counter, 2],
                                                             years_predict=100)
                    series_list.append(pd.Series(data=predictions_mc, name=f'sim_{param_ix}'))

                self.monte_carlo_dfs[label] = pd.concat(series_list, axis=1)

