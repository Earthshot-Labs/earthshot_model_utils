from scipy.optimize import curve_fit as curve_fit_scipy
from importlib import import_module
import numpy as np
import pandas as pd


class GrowthCurveFit():

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

    def fit_curve(self, curve_formula, curve_fit_params=None, set_params=None, response_variable_name='agb_bgb_tCO2e_ha'):
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

        set_params: dictionary, if specifying other parameters, e.g. p of Chapman-Richards

        """
        self.response_variable_name = response_variable_name

        # Set curve fit params if they aren't input
        if curve_fit_params is None:
            curve_fit_params = {}

        #Import the curve formula function
        curve_formula_module = import_module(name='.curve_formulas', package='model_utilities.curve_fitting')
        curve_fit_params['f'] = getattr(curve_formula_module, curve_formula)
        #Save for future use
        self.curve_formula = curve_formula

        #Set ydata, get age to create xdata
        curve_fit_params['ydata'] = self.growth_df[response_variable_name]
        age = np.array(self.growth_df['age']).reshape((self.growth_df['age'].shape[0], 1))

        # Lists to hold parameter estimates and covariance matrices for each curve fit
        #TODO: add to __init__?
        self.params = []
        self.covars = []

        # Loop through array of y_max values, set xdata, do curve fits and store results
        max_list = self.max_df[response_variable_name].tolist()
        for this_max in max_list:
            y_max_array = np.ones_like(age) * this_max
            if curve_formula == 'chapman_richards_set_ymax':
                curve_fit_params['xdata'] = np.concatenate((age, y_max_array), axis=1)
            elif curve_formula == 'chapman_richards_set_ymax_and_p':
                p_array = np.ones_like(age) * set_params['p']
                curve_fit_params['xdata'] = np.concatenate((age, y_max_array, p_array), axis=1)
                self.set_params = set_params

            this_params, this_covar = curve_fit_scipy(**curve_fit_params)
            self.params.append(this_params)
            self.covars.append(this_covar)

        #Store function, xdata, and ydata along with other curve fit parameters
        # TODO: add to __init__?
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
        age_years = np.arange(0, years_predict, 1).reshape((years_predict, 1))
        y_max_array = np.ones_like(age_years) * y_max_agb_bgb

        if self.curve_formula == 'chapman_richards_set_ymax':
            x_data = np.concatenate((age_years, y_max_array), axis=1)
            predictions = self.curve_fit_params['f'](x=x_data, k=params[0], p=params[1])
        elif self.curve_formula == 'chapman_richards_set_ymax_and_p':
            p_array = np.ones_like(age_years) * self.set_params['p']
            x_data = np.concatenate((age_years, y_max_array, p_array), axis=1)
            predictions = self.curve_fit_params['f'](x=x_data, k=params[0])

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
        max_list = self.max_df[self.response_variable_name].tolist()
        for row_counter in range(self.max_df.shape[0]):
            #Get predictions
            predictions, age_years = self.predict(params=self.params[row_counter],
                                                  y_max_agb_bgb=max_list[row_counter],
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
                for param_ix in range(param_sample.shape[1]):
                    predictions_mc, age_years = self.predict(params=param_sample[:,param_ix],
                                                             y_max_agb_bgb=self.max_df.iloc[row_counter, 2],
                                                             years_predict=100)
                    series_list.append(pd.Series(data=predictions_mc, name=f'sim_{param_ix}'))

                self.monte_carlo_dfs[label] = pd.concat(series_list, axis=1)


# get 95% CI from entire ensemble
def ensemble_ci(growth_curve_fit):
    # combine dict of dfs to single df
    entire_ensemble = pd.concat(growth_curve_fit.monte_carlo_dfs.values(), axis = 1, ignore_index=True)
    #  track number of ensemble memebers before filtering
    n_members_orig = entire_ensemble.shape[1]
    #filter out unrealistic sims
    entire_ensemble = entire_ensemble.loc[:, entire_ensemble.iloc[0] < entire_ensemble.iloc[99]] 
    # track number of ensemble members filtered
    print('number ensemble members removed: ', entire_ensemble.shape[1] - n_members_orig)

    ensemble_deciles = pd.DataFrame({})
    ensemble_deciles[[0.025,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.975]] = entire_ensemble.quantile(q=[0.025,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.975], axis=1).transpose()
    ensemble_deciles['Age'] = ensemble_deciles.index
    
    return ensemble_deciles


# eventually should add ensemble_ci to class so you don't pass it to function separately
def extract_ensemble_member(growth_curve_fit, ensemble_ci, pctile, thresh=np.nan, use_thresh=False):
    import math

    # make df that is distance between ensemble member and ensemble_ci for each yr first 30 yrs
    entire_ensemble = pd.concat(growth_curve_fit.monte_carlo_dfs.values(), axis = 1, ignore_index=True)
    entire_ensemble = entire_ensemble.loc[:, entire_ensemble.iloc[0] < entire_ensemble.iloc[99]]
    
    # if we want to select from a certain maximum that is closest to the ensemble ci
    if use_thresh == True:
        entire_ensemble = entire_ensemble.loc[:, entire_ensemble.iloc[99] > thresh]

    # calculate difference at each time step between ensemble member and ci percentile
    diffdf = entire_ensemble.sub(ensemble_ci[pctile], axis='index')

    # calculate sum of squares over years 0-30 (key for reforestation projects)
    ss = (diffdf.iloc[0:30,:]**2).sum()
    # find index that minimizes ss
    min_index = ss.index[ss == ss.min()].values[0]

    # select ensemble member that minimized ss 
    sub_ensemble = entire_ensemble.loc[:,min_index]

    return sub_ensemble