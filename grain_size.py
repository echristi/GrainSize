# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator, UnivariateSpline
import numpy as np
import warnings

# Define Grain Size Class
class Grain_Size:
    ''' 
    Create Grain_Size class
        
    Parameters
    ----------
    data : list of tuples
       List of tuples of (grainsize, percent passing). Grain size should be
       in mm and percent expressed as fration * 100 (e.g. 10% is 10 not 0.1)
       For example, [(5.0, 100), (0.5, 80), (0.2, 60)] would indicate
       100 percent pass the 5 mm size, 80 percent pass 0.5 mm and,
       60 percent pass 0.2 mm.
                 
    fit : str, optional
       interpolation method used to fit grain size curve
           'linear' uses linear interp1d function in scipy (default)
           'spline' uses univariate spline in scipy
           'c_cubic' uses contrained cubic interpolation as defined
           'm_cubic' uses PchipInterpolator in scipy (1-d monotonic cubic)
    
    gravel_max : float, optional
        size in mm to automatically consider as max grain size, where it is
        assumed that 100 percent is smaller than this size
    
    hydro_min : float, optional
       size in mm to automatically consider as min grain size, where it is 
       assumed that 0 percent is smaller than this size. 
       
    assume_missing_sieves: True,False
       If True will add in sieve sizes greater than max given grain size
       and assume 100% passing
       
    extrapolate : True,False
        If True will extrapolate from smallest grain size provided to 
        hydromin using linear interpolation.  Note that in log space it won't
        appear linear.
    
    dup_percent_offset : float
        Value of offset to add to duplicate percentages other than 0 or 100.
        If negative, no offset will be applied and an error will be raised.
       
    Attributes
    ----------
    
    data : data as provided as the data parameter
    
    gravel_max : size provided as gravel_max parameter
    
    hydro_min : size provided as hydro_min parameter
    
    percents : a list of np.linspace(0,100,101).  Values of sizes from these
        percent values
        
    sizes : list of sizes for every 1 percent cut-off from the fitted 
        grain size distribution curve
        
    d10 : d10 grain size.
    
    d20 : d10 grain size. 
     
    d30 : d10 grain size. 
      
    d40 : d10 grain size.
    
    d50 : d50 grain size
    
    d60 : d60 grain size
    
    d70 : d70 grain size
    
    d80 : d80 grain size
    
    d90 : d90 grain size
    
    cu : Uniformity coefficient.  d60/d10
    
    cc : Coefficient of curvature. d30^2/(d10 * d60)
        
    '''
    def __init__(self, data, fit = 'm_cubic', gravel_max = 100, 
                 hydro_min = 0.0001, assume_missing_sieves = True,
                 extrapolate = True, dup_percent_offset = 0.0001):

        # Maintain data as input
        self.data = sorted(data, key=lambda record: (record[1], record[0]))
        self.data = [tuple(float(flt) for flt in sublist) for sublist in self.data]
        # Fit data will be modified for curve
        self.fit_data = sorted(self.data, key=lambda record: (record[1], record[0]))
        self.gravel_max = gravel_max
        self.hydro_min = hydro_min

        def _check_duplicates():
            # !! Need to remove duplicate values of percent as they are used as 'x'
            # in the curve fitting and can cause problems
            # at 0 percent keep largest size
            # at 100 percent keep smallest size
            self.fit_data = sorted(self.fit_data, key=lambda record: (record[1], record[0]))
            fit_data_index = 0
            zero = False
            hundred = False
            for i in self.fit_data:
                #size = i[0]
                percent = i[1]
                if percent == 0.0 and zero == True:
                    self.fit_data.pop(fit_data_index-1)
                if percent == 0.0 and zero == False:
                    zero = True
                if percent == 100.0 and hundred == True:
                    self.fit_data = self.fit_data[:fit_data_index]
                    break
                if percent == 100.0 and hundred == False:
                    hundred = True
                fit_data_index += 1
            # For other duplicates remove if size is the same
            iprevious = None
            for i in self.fit_data:
                if i == iprevious:
                    self.fit_data.remove(i)
                iprevious = i
            # If duplicates still present raise error
            percent_seen = []
            for i in self.fit_data:
                percent = i[1]
                if percent in percent_seen:
                    if dup_percent_offset > 0:
                        while percent in percent_seen:
                            percent += dup_percent_offset
                        percent_seen.append(percent)
                    else:                    
                        raise ValueError ('Duplicate percent (%f) for two size values' % (percent))
                else:
                    percent_seen.append(percent)
            # Keep fit_data sorted
            self.fit_data = sorted(self.fit_data, key=lambda record: (record[1], record[0]))
            
        if extrapolate == True:
            _ex_percents = [0, (self.fit_data[0][1])]
            _ex_sizes = [hydro_min, (self.fit_data[0][0])]
            _extrapolation_func = interp1d(_ex_percents,  _ex_sizes)
            _extrapolation_percents = np.arange(0, self.fit_data[0][1], 0.1)
            _extrapolation_sizes = _extrapolation_func(_extrapolation_percents)
            for index, _extrapolation_size in enumerate(_extrapolation_sizes):
                self.fit_data.append((_extrapolation_size, _extrapolation_percents[index]))
            self.fit_data = sorted(self.fit_data, key=lambda record: (record[1], record[0]))
            #Check for duplicates
            _check_duplicates()
            
        # Add hydro_min and gravel_max to fit_data       
        self.fit_data.append((float(gravel_max), 100.0))
        self.fit_data.append((float(hydro_min), 0.0))
        self.fit_data = sorted(self.fit_data, key=lambda record: (record[1], record[0]))                 
        _check_duplicates()
            
        if assume_missing_sieves == True:        
            # Standard ASTM sieve sizes in mm from ASTM D422-63
            sieve_sizes = [75.0, 50.0, 37.5, 25.0, 19.0, 9.5, 4.75, 2.00,
                           0.850, 0.425, 0.250, 0.106, 0.075]
            # Cycle through and add sizes that are greater than max of fit_data 
            # This is assuming that for sieve sizes larger than the largest presented
            # in the data 100 percent pass.
            for sieve_size in sieve_sizes:
                if sieve_size > self.data[-1][0]:
                    self.fit_data.append((sieve_size, 100.0))
                    # Keep fit_data sorted
                    self.fit_data = sorted(self.fit_data, key=lambda record: (record[1], record[0]))
                    # Check for duplicates
                    _check_duplicates()
                    

            
        def _curve ():
            # Fill percent pass and size lists
            data_percent_pass = []
            data_size = []
            for i in self.fit_data:
                data_size.append(i[0])
                data_percent_pass.append(i[1])
            data_size_log = np.log10(data_size)                
            # Curve fucntion          
            if fit == 'spline':
                func = UnivariateSpline(data_percent_pass, data_size_log, k = 2, s = 0.001)
            if fit == 'linear':
                func = interp1d(data_percent_pass, data_size_log, kind = 'linear')
            if fit == 'm_cubic':
                func = PchipInterpolator(data_percent_pass, data_size_log)

            
            percents = np.linspace(0, 100, 101)
            log_sizes = func(percents)
            tens = np.ones(percents.shape)*10
            sizes = np.power(tens, log_sizes)
            return sizes, percents
        
        # Get initial curve fit
        sizes, percents = _curve()
        
        
        self.percents = percents
        self.sizes = sizes
        self.d10 = sizes[10]
        self.d20 = sizes[20]
        self.d30 = sizes[30]
        self.d40 = sizes[40]
        self.d50 = sizes[50]
        self.d60 = sizes[60]
        self.d70 = sizes[70]
        self.d80 = sizes[80]
        self.d90 = sizes[90]
        self.cu = self.d60/self.d10
        self.cc = self.d30**2/(self.d10 * self.d60)

    def barr_k (self, porosity = 0.3, cs = 1.1, rho = 1.0, g = 980,
                visc = 0.01007, units_out = 'cm/sec'):
        '''
        Get K using Barr method
        
        Parameters
        ----------
        porosity : float, optional
           Default is 0.3
        cs : float, optional
            shapefactor, default = 1.1
        rho : float, optional
            Density of fluid in g/ml, default = 1.0
        g : float, optional
            Gravitational constant in cm/s**2, default = 980
        visc : float, optional
            Dyn viscosity, default = 0.01007
        
        Returns
        --------
        k : float
           Hydraulic conductivity, cm/sec
           
        References
        -----------
        Barr D.W., 2001, Coefficient of Permeability Determined by Measurable
            Parameters. Ground Water v. 39, no. 3 pg. 356-361.
        '''
        So = 0
        for i in self.sizes[1:99]:
            ## Frac retained at each point = 0.01
            So += (3*0.01*cs)/((i/2)*0.1) # Converting size to radius (divide by 2) and mm to cm here  (times 0.1)
        # Do 0 to .5 and 99.5 to 100 components
        # for each frac retained  = 0.005
        So += (3*0.005*cs)/((self.sizes[0]/2)*0.1)
        So += (3*0.005*cs)/((self.sizes[100]/2)*0.1)
        S = So * (1-porosity)
        m = porosity / S
        k = (rho*g*porosity*m**2)/(5*visc) # K in cm/sec
        
        # Convert to units_out
        if units_out == 'm/day':
            k = k * 864
        if units_out == 'ft/day':
            k = k * 2835 
        return k
        
    def kc (self, visc = 1.0e-7, sp_weight = 0.009789, c = 5.0, porosity = 0.3,
            sf = 6.0, units_out = 'cm/sec'):
        '''
        Get K using Kozeny-Carman method
        
        Parameters
        -----------
        visc : float, optional
            Viscosity of fluid in (N*sec/cm^2)
        sp_weight : float, optional
            Specific weight of fluid in (N/cm^3)
        c : float, optional
            emperical coefficient
        porosity : float, optional
           default porosity = 0.3
        sf : float, optional
           Shape factor, defualt = 6.0
           
        Returns
        -------
        k : float
           Hydraulic conductivity, cm/sec
           
        References
        -----------
        Carrier W.D III, 2003. Goodbye, Hazen; Hello, Kozeny-Carman. Journal of 
           Geotechnical and Geoenvironmental Engineering, v. 129 no. 11 
           pg. 1054-1056.
        
           
        '''
        So = 0
        for i in self.sizes[1:99]:
            ## Frac retained at each point = 0.01
            So += (0.01*sf)/((i)*0.1) # Converting  mm to cm here 
        # Do 0 to .5 and 99.5 to 100 components
        # for each frac retained  = 0.005
        So += (0.005*sf)/((self.sizes[0])*0.1)
        So += (0.005*sf)/((self.sizes[100])*0.1)
        e = (porosity/(1-porosity))
        k = (sp_weight/visc)*(1/c)*(1/So**2)*(e**3/(1+e)) # K in cm/sec
        
        # Convert to units_out
        if units_out == 'm/day':
            k = k * 864
        if units_out == 'ft/day':
            k = k * 2835
        
        return k
        
    def hazen (self, units_out = 'cm/sec'):
        ''' Get K value using Hazen method
        
        Parameters
        -----------
        units_out: {'cm/sec'; 'm/day'; 'ft/day'}, optional
            Output units for K, default is 'cm/sec'
            
        Returns
        --------
        K: float
            K estimate in units of units_out
        '''

        # Get K with Hazen in cm/sec
        # When d10 in mm and K in cm/sec, value of coefficient is 1
        K = self.d10**2
        
        # Convert to units_out
        if units_out == 'm/day':
            K = K * 864
        if units_out == 'ft/day':
            K = K * 2835
    
        return K
    
    def prugh (self, density, units_out = 'cm/sec'):
                   
        ''' Get K value using Prugh method
        
        Parameters
        -----------  
        density : {'dense', medium', 'loose'}
        
        units_out : {'cm/sec', 'm/sec', 'm/day', 'ft/day'}
           
           
        Returns
        --------
        k : float
           Hydraulic conductivity in m/sec.  If d50 is out of range k = none and warning
           printed
           
        Reference
        ---------
        Powers, J. P. 1992. Construction dewatering: new methods and 
            applications 2nd ed. John Wiley & Sons. pg 41-44.
           
        '''
        cu = self.cu
        d50 = self.d50
        import barr_utils.grain_size
        fit_points_file = barr_utils.grain_size.__path__[0]+'\\prugh_digitize_pts.npz'
        fit_points = np.load(fit_points_file)
        if cu >= 1.25 and cu < 1.75:
            cu = 1.5
        elif cu >= 1.75:
            cu = float((round(cu)))
        else:
            warnings.warn('Cu out of range.  1.5 < Cu < 6. '
                        'Values as low as 1.25 and as high as 6.49 accepted '
                        'as Cu values are rounded to fit nearest curve of Prugh')
        if density.lower() == 'dense':
            if cu > 6 or cu < 1.25:
                warnings.warn('Cu out of range.  1.5 < Cu < 6. '
                'Values as low as 1.25 and as high as 6.49 accepted '
                'as Cu values are rounded to fit nearest curve of Prugh')            
            if cu == 1.5 or cu < 1.25:
                k_cu = fit_points['cu1_k_dense']
                d50_cu = fit_points['cu1_d50_dense']
            if cu == 2.0:
                k_cu = fit_points['cu2_k_dense']
                d50_cu = fit_points['cu2_d50_dense']
            if cu == 3.0:
                k_cu = fit_points['cu3_k_dense']
                d50_cu = fit_points['cu3_d50_dense']
            if cu == 4.0:
                k_cu = fit_points['cu4_k_dense']
                d50_cu = fit_points['cu4_d50_dense']
            if cu == 5.0:
                k_cu = fit_points['cu5_k_dense']
                d50_cu = fit_points['cu5_d50_dense']
            if cu == 6.0:
                k_cu = fit_points['cu6_k_dense']
                d50_cu = fit_points['cu6_d50_dense']
                
        if density.lower() == 'medium':
            if cu > 10 or cu < 1.25:
                warnings.warn('Cu out of range.  1.5 < Cu < 10. '
                'Values as low as 1.25 and as high as 10.49 accepted '
                'as Cu values are rounded to fit nearest curve of Prugh')  
            if cu == 1.5:
                k_cu = fit_points['cu1_k_med_density']
                d50_cu = fit_points['cu1_d50_med_density']
            if cu == 2.0:
                k_cu = fit_points['cu2_k_med_density']
                d50_cu = fit_points['cu2_d50_med_density']
            if cu == 3.0:
                k_cu = fit_points['cu3_k_med_density']
                d50_cu = fit_points['cu3_d50_med_density']
            if cu == 4.0:
                k_cu = fit_points['cu4_k_med_density']
                d50_cu = fit_points['cu4_d50_med_density']
            if cu == 5.0:
                k_cu = fit_points['cu5_k_med_density']
                d50_cu = fit_points['cu5_d50_med_density']
            if cu == 6.0:
                k_cu = fit_points['cu6_k_med_density']
                d50_cu = fit_points['cu6_d50_med_density']
            if cu == 7.0:
                k_cu = fit_points['cu7_k_med_density']
                d50_cu = fit_points['cu7_d50_med_density']
            if cu == 8.0:
                k_cu = fit_points['cu8_k_med_density']
                d50_cu = fit_points['cu8_d50_med_density']
            if cu == 9.0:
                k_cu = fit_points['cu9_k_med_density']
                d50_cu = fit_points['cu9_d50_med_density']
            if cu == 10.0:
                k_cu = fit_points['cu10_k_med_density']
                d50_cu = fit_points['cu10_d50_med_density']
                
        if density.lower() == 'loose':
            if cu > 6 or cu < 1.25:
                warnings.warn('Cu out of range.  1.5 < Cu < 6. '
                'Values as low as 1.25 and as high as 6.49 accepted '
                'as Cu values are rounded to fit nearest curve of Prugh')            
            if cu == 1.5 or cu < 1.25:
                k_cu = fit_points['cu1_k_loose']
                d50_cu = fit_points['cu1_d50_loose']
            if cu == 2.0:
                k_cu = fit_points['cu2_k_loose']
                d50_cu = fit_points['cu2_d50_loose']
            if cu == 3.0:
                k_cu = fit_points['cu3_k_loose']
                d50_cu = fit_points['cu3_d50_loose']
            if cu == 4.0:
                k_cu = fit_points['cu4_k_loose']
                d50_cu = fit_points['cu4_d50_loose']
            if cu == 5.0:
                k_cu = fit_points['cu5_k_loose']
                d50_cu = fit_points['cu5_d50_loose']
            if cu == 6.0:
                k_cu = fit_points['cu6_k_loose']
                d50_cu = fit_points['cu6_d50_loose']
    
        # Fit curve
        try:
            func = interp1d(d50_cu, k_cu)
        except:
            None # Warning given later
        # Get K Value
        try:
            k = func(d50)
            k = k.item()
            # raw k is in m/sec
            if units_out == 'cm/sec':
                k = k * 100
            if units_out == 'm/sec':
                k = k
            if units_out == 'm/day':
                k = k * 86400
            if units_out == 'ft/day':
                k = k * 283465
        except:
            warnings.warn('Error: d50 is out of range for Cu and density')
            k = None  
        return k

        
    def plot(self):
        '''Plot grain size curve
        '''
        raw_data = sorted(self.data, key=lambda record: record[1])
        data_size = []
        data_percent = []
        for i in raw_data:
            data_size.append(i[0])
            data_percent.append(i[1])
        plt.figure()
        plt.plot(self.sizes, self.percents)
        plt.plot(data_size, data_percent, 'ro')
        plt.locator_params(axis = 'y', nbins = 10)
        plt.xscale('log')
        plt.xlim(0.0001, 100)
        plt.gca().invert_xaxis()
        plt.grid(which='both', axis = 'both')
        plt.ylabel('Percent finer by weight')
        plt.xlabel('Grain Size (mm)')
                
    
