"""
Implement optics algorithms for optical phase tomography using GPU

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
November 22, 2017
"""
import scipy.io as sio
import numpy as np
import arrayfire as af
import skimage.feature
import contexttimer
from opticaltomography import settings
from opticaltomography.opticsmodel import MultiTransmittance,  MultiPhaseContrast, MultiBorn
from opticaltomography.opticsmodel import Defocus, Aberration
from opticaltomography.opticsutil import ImageCrop, calculateNumericalGradient, EmptyClass
from opticaltomography.regularizers import Regularizer
from IPython.core.debugger import set_trace

np_complex_datatype = settings.np_complex_datatype
np_float_datatype   = settings.np_float_datatype
af_float_datatype   = settings.af_float_datatype
af_complex_datatype = settings.af_complex_datatype

class AlgorithmConfigs:
    """
    Class created for all parameters for tomography solver
    """
    def __init__(self):
        self.method              = "FISTA" #or GradientDescent"
        self.stepsize            = 1e-2
        self.max_iter            = 20
        self.error               = []
        self.reg_term            = 0.0      #L2 norm

        #FISTA
        self.fista_global_update = False
        self.restart             = False

        #total variation regularization
        self.total_variation     = False
        self.reg_tv              = 1.0 #lambda
        self.max_iter_tv         = 15
        self.order_tv            = 1
        self.total_variation_gpu = False
        self.total_variation_anisotropic = False

        #lasso
        self.lasso               = False
        self.reg_lasso           = 1.0

        #positivity constraint
        self.positivity_real     = (False, "larger") #or smaller
        self.positivity_imag     = (False, "larger") #or smaller
        self.pure_real           = False
        self.pure_imag           = False

        #purely amplitude/phase
        self.pure_amplitude      = False
        self.pure_phase          = False

        #batch gradient update
        self.batch_size          = 1

        #random order update
        self.random_order        = False
    
        # Reconstruct from field or amplitude measurement
        self.recon_from_field    = False
        self.cost_criterion      = "amplitude"

class PhaseObject3D:
    """
    Class created for 3D objects.
    Depending on the scattering model, one of the following quantities will be used:
    - Refractive index (RI)
    - Transmittance function (Trans)
    - PhaseContrast
    - Scattering potential (V)

    shape:              shape of object to be reconstructed in (x,y,z), tuple
    voxel_size:         size of each voxel in (x,y,z), tuple
    RI_obj:             refractive index of object(Optional)
    RI:                 background refractive index (Optional)
    slice_separation:   For multislice algorithms, how far apart are slices separated, array (Optional)
    """
    def __init__(self, shape, voxel_size, RI_obj = None, RI = 1.0, slice_separation = None):
        assert len(shape) == 3, "shape should be 3 dimensional!"
        self.shape           = shape
        self.RI_obj          = RI * np.ones(shape, dtype = np_complex_datatype) if RI_obj is None else RI_obj.astype(np_complex_datatype)
        print("RI max:", np.max(np.abs(self.RI_obj)))
        self.RI              = RI
        self.pixel_size      = voxel_size[0]
        self.pixel_size_z    = voxel_size[2]

        if slice_separation is not None:
            #for discontinuous slices
            assert len(slice_separation) == shape[2]-1, "number of separations should match with number of layers!"
            self.slice_separation = np.asarray(slice_separation).astype(np_float_datatype)
        else:
            #for continuous slices
            self.slice_separation = self.pixel_size_z * np.ones((shape[2]-1,), dtype = np_float_datatype)

    def convertRItoTrans(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.trans_obj       = np.exp(1.0j*k0*(self.RI_obj - self.RI)*self.pixel_size_z)
        self.RI_obj          = None

    def convertRItoPhaseContrast(self):
        self.contrast_obj    = self.RI_obj - self.RI
        self.RI_obj          = None

    def convertPhaseContrasttoRI(self):
        self.RI_obj          = self.contrast_obj + self.RI

    def convertRItoV(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.V_obj           = k0**2 * (self.RI**2 - self.RI_obj**2)
        self.RI_obj          = None

    def convertVtoRI(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        B                    = -1.0 * (self.RI**2 - self.V_obj.real/k0**2)
        C                    = -1.0 * (-1.0 * self.V_obj.imag/k0**2/2.0)**2
        RI_obj_real          = ((-1.0 * B + (B**2-4.0*C)**0.5)/2.0)**0.5
        RI_obj_imag          = -0.5 * self.V_obj.imag/k0**2/RI_obj_real
        self.RI_obj          = RI_obj_real + 1.0j * RI_obj_imag

class TomographySolver:
    """
    Highest level solver object for tomography problem

    phase_obj_3d:               phase_obj_3d object defined from class PhaseObject3D
    fx_illu_list:               illumination angles in x, default = [0] (on axis)
    fy_illu_list:               illumination angles in y
    propagation_distance_list:  defocus distances for each illumination
    """
    def __init__(self, phase_obj_3d, fx_illu_list = [0], fy_illu_list = [0], \
                 propagation_distance_list = [0], **kwargs):
        
        self.phase_obj_3d    = phase_obj_3d
        self.wavelength      = kwargs["wavelength"]
        self.na              = kwargs["na"]

        #Illumination angles
        assert len(fx_illu_list) == len(fy_illu_list)
        self.fx_illu_list    = fx_illu_list
        self.fy_illu_list    = fy_illu_list
        self.number_illum    = len(self.fx_illu_list)
        
        #Defocus distances and object
        self.prop_distances  = propagation_distance_list
        self._aberration_obj = Aberration(self.phase_obj_3d.shape[:2], self.phase_obj_3d.pixel_size, self.wavelength, self.na)
        self._defocus_obj    = Defocus(self.phase_obj_3d.shape[:2], self.phase_obj_3d.pixel_size, **kwargs)
        self._crop_obj       = ImageCrop(self.phase_obj_3d.shape[:2], **kwargs)
        self.number_defocus  = len(self.prop_distances)

        #Scattering models and algorithms
        self._opticsmodel    = {"MultiTrans":                  MultiTransmittance,
                                "MultiPhaseContrast":          MultiPhaseContrast,
                                "MultiBorn":                   MultiBorn,
                                }
        self._algorithms     = {"GradientDescent":    self._solveFirstOrderGradient,
                                "FISTA":              self._solveFirstOrderGradient,
                               }
        self.scat_model_args = kwargs

    def setScatteringMethod(self, model = "MultiTrans"):
        """
        Define scattering method for tomography

        model: scattering models, it can be one of the followings:
               "MultiTrans", "MultiPhaseContrast", "Rytov", "MultiRytov", "Born", "Born"
        """
        
        if hasattr(self, '_scattering_obj'):
            del self._scattering_obj
        if model == "MultiTrans":
            try:
                self.phase_obj_3d.convertRItoTrans(self.wavelength)
            except:
                self.phase_obj_3d.trans_obj = self._x
            self._x          = self.phase_obj_3d.trans_obj

        elif model == "MultiPhaseContrast":
            if not hasattr(self.phase_obj_3d, 'contrast_obj'):
                try:
                    self.phase_obj_3d.convertRItoPhaseContrast()
                except:
                    self.phase_obj_3d.contrast_obj = self._x
            self._x          = self.phase_obj_3d.contrast_obj

        else:
            if not hasattr(self.phase_obj_3d, 'V_obj'):
                try:
                    self.phase_obj_3d.convertRItoV(self.wavelength)
                except:
                    self.phase_obj_3d.V_obj = self._x
            self._x          = self.phase_obj_3d.V_obj

        self._scattering_obj = self._opticsmodel[model](self.phase_obj_3d, **self.scat_model_args)
        self._initialization(x_init = self._x)
        self.scat_model      = model
        
    def forwardPredict(self, field = False):
        """
        Uses current object in the phase_obj_3d to predict the amplitude of the exit wave
        Before calling, make sure correct object is contained
        """
        obj_gpu = af.to_array(self._x) #self._x created in self.setScatteringMethod()
        with contexttimer.Timer() as timer:
            forward_scattered_predict= []
            for illu_idx in range(self.number_illum):
                fx_illu           = self.fx_illu_list[illu_idx]
                fy_illu           = self.fy_illu_list[illu_idx]               

                fields = self._forwardMeasure(fx_illu, fy_illu, obj = obj_gpu)
                if field:
                    forward_scattered_predict.append(np.array(fields["forward_scattered_field"]))
                else:
                    forward_scattered_predict.append(np.abs(fields["forward_scattered_field"]))
                if self.number_illum > 1:
                    print("illumination {:03d}/{:03d}.".format(illu_idx, self.number_illum), end="\r")                            
        if len(forward_scattered_predict[0][0].shape)==2:
            forward_scattered_predict = np.array(forward_scattered_predict).transpose(2, 3, 1, 0)
        elif len(forward_scattered_predict[0][0].shape)==1:
            forward_scattered_predict = np.array(forward_scattered_predict).transpose(1, 2, 0)            
            return forward_scattered_predict

    def checkGradient(self, delta = 1e-4):
        """
        check if the numerical gradient is similar to the analytical gradient. Only works for 64 bit data type.
        """
        
        #assert af_float_datatype == af.Dtype.f64, "This will only be accurate if 64 bit datatype is used!"
        shape     = self.phase_obj_3d.shape
        point     = (np.random.randint(shape[0]), np.random.randint(shape[1]), np.random.randint(shape[2]))
        illu_idx  = np.random.randint(len(self.fx_illu_list))
        fx_illu   = self.fx_illu_list[illu_idx]
        fy_illu   = self.fy_illu_list[illu_idx]
        x         = np.ones(shape, dtype = np_complex_datatype)
        if self._crop_obj.pad:
            amplitude = af.randu(shape[0]//2, shape[1]//2, dtype = af_float_datatype)
        else:
            amplitude = af.randu(shape[0], shape[1], dtype = af_float_datatype)
        print("Computing the gradient at point: ", point)
        print("fx     : %5.2f, fy     : %5.2f " %(fx_illu, fy_illu))

        def func(x0):
            fields              = self._scattering_obj.forward(x0, fx_illu, fy_illu)
            field_scattered     = self._defocus_obj.forward(field_scattered, self.prop_distances)
            field_measure       = self._crop_obj.forward(field_scattered)
            residual            = af.abs(field_measure) - amplitude
            function_value      = af.sum(residual*af.conjg(residual)).real
            return function_value

        numerical_gradient      = calculateNumericalGradient(func, x, point, delta = delta)

        fields                  = self._scattering_obj.forward(x, fx_illu, fy_illu)
        forward_scattered_field = fields["forward_scattered_field"]
        forward_scattered_field = self._defocus_obj.forward(forward_scattered_field, self.prop_distances)
        field_measure           = self._crop_obj.forward(forward_scattered_field)
        fields["forward_scattered_field"] = field_measure
        analytical_gradient     = self._computeGradient(fields, amplitude)[0][point]

        print("numerical gradient:  %5.5e + %5.5e j" %(numerical_gradient.real, numerical_gradient.imag))
        print("analytical gradient: %5.5e + %5.5e j" %(analytical_gradient.real, analytical_gradient.imag))

    def _forwardMeasure(self, fx_illu, fy_illu, obj = None):

        """
        From an illumination angle, this function computes the exit wave.
        fx_illu, fy_illu:       illumination angle in x and y (scalars)
        obj:                    object to be passed through (Optional, default pick from phase_obj_3d)
        """
        #Forward scattering of light through object
        #_scattering_obj set in setScatteringModel()
        if obj is None:
            fields = self._scattering_obj.forward(self._x, fx_illu, fy_illu)
        else:
            fields = self._scattering_obj.forward(obj, fx_illu, fy_illu)
        field_scattered                   = self._defocus_obj.forward(fields["forward_scattered_field"], self.prop_distances)    
        field_scattered                   = self._aberration_obj.forward(field_scattered)
        field_scattered                   = self._crop_obj.forward(field_scattered)
        fields["forward_scattered_field"] = field_scattered
        return fields

    def _computeGradient(self, fields, measurement):
        """
        Error backpropagation to return a gradient
        fields:  contains all necessary predicted information (predicted field, intermediate cache)
                 to compute gradient
        measurement: amplitude measured
        """
        cache                        = fields["cache"]
        field_predict                = fields["forward_scattered_field"]
        if self.configs.recon_from_field:
            field_bp                 = field_predict - measurement
        else:
            if self.configs.cost_criterion == "intensity":
                field_bp             = 2 * field_predict * (af.abs(field_predict) ** 2 - af.abs(measurement) ** 2)
            elif self.configs.cost_criterion == "amplitude":
                field_bp             = field_predict - measurement*af.exp(1.0j*af.arg(field_predict))
        field_bp                     = self._crop_obj.adjoint(field_bp)
        field_bp                     = self._aberration_obj.adjoint(field_bp)
        field_bp                     = self._defocus_obj.adjoint(field_bp, self.prop_distances)
        gradient                     = self._scattering_obj.adjoint(field_bp, cache)

        return gradient["gradient"]

    def _initialization(self,configs=AlgorithmConfigs(), \
                        x_init = None, obj_support = None,\
                        pupil = None, pupil_support = None, \
                        verbose = True):
        """
        Initialize algorithm
        configs:         configs object from class AlgorithmConfigs
        x_init:          initial guess of object
        """
        self.configs = configs
        if pupil is not None:
            self.configs.pupil = pupil
        if pupil_support is not None:
            self.configs.pupil_support = pupil_support

        if x_init is None:
            if self.scat_model is "MultiTrans":
                self._x[:, :, :] = 1.0
            else:
                self._x[:, :, :] = 0.0
        else:
            self._x[:, :, :] = x_init

        self.obj_support = obj_support

        self._regularizer_obj = Regularizer(self.configs, verbose)

    def _solveFirstOrderGradient(self, measurements, verbose, callback=None):
        """
        MAIN part of the solver, runs the FISTA algorithm
        configs:        configs object from class AlgorithmConfigs
        measurements:     all measurements 
                            self.configs.recon_from_field == True: field
                            self.configs.recon_from_field == False: amplitude measurement
        verbose:        boolean variable to print verbosely
        """
        flag_FISTA    = False
        if self.configs.method == "FISTA":
            flag_FISTA = True

        # update multiple angles at a time
        batch_update = False
        if self.configs.fista_global_update or self.configs.batch_size != 1:
            gradient_batch    = af.constant(0.0, self.phase_obj_3d.shape[0],\
                                                 self.phase_obj_3d.shape[1],\
                                                 self.phase_obj_3d.shape[2], dtype = af_complex_datatype)
            batch_update = True
            if self.configs.fista_global_update:
                self.configs.batch_size = 0

        #TODO: what if num_batch is not an integer
        if self.configs.batch_size == 0:
            num_batch = 1
        else:
            num_batch = self.number_illum // self.configs.batch_size
        stepsize      = self.configs.stepsize
        max_iter      = self.configs.max_iter
        reg_term      = self.configs.reg_term
        self.configs.error = []
        obj_gpu       = af.constant(0.0, self.phase_obj_3d.shape[0],\
                                         self.phase_obj_3d.shape[1],\
                                         self.phase_obj_3d.shape[2], dtype = af_complex_datatype)

        #Initialization for FISTA update
        if flag_FISTA:
            restart       = self.configs.restart
            y_k           = self._x.copy()
            t_k           = 1.0

        #Set Callback flag
        if callback is None:
            run_callback = False
        else:
            run_callback = True

        #Start of iterative algorithm
        with contexttimer.Timer() as timer:
            if verbose:
                print("---- Start of the %5s algorithm ----" %(self.scat_model))
            for iteration in range(max_iter):
                illu_counter          = 0
                cost                  = 0.0
                obj_gpu[:]            = af.to_array(self._x)
                if self.configs.random_order:
                    illu_order = np.random.permutation(range(self.number_illum))
                else:
                    illu_order = range(self.number_illum)

                for batch_idx in range(num_batch):
                    if batch_update:
                        gradient_batch[:,:,:] = 0.0

                    if self.configs.batch_size == 0:
                        illu_indices = illu_order
                    else:
                        illu_indices = illu_order[batch_idx * self.configs.batch_size : (batch_idx+1) * self.configs.batch_size]
                    for illu_idx in illu_indices:                                                    
                        #forward scattering
                        fx_illu                       = self.fx_illu_list[illu_idx]
                        fy_illu                       = self.fy_illu_list[illu_idx]
                        fields                        = self._forwardMeasure(fx_illu, fy_illu, obj = obj_gpu)
                        #calculate error
                        measurement             = af.to_array(measurements[:,:,:,illu_idx].astype(np_complex_datatype))

                        if self.configs.recon_from_field:
                            residual                  = fields["forward_scattered_field"] - measurement
                        else:
                            if self.configs.cost_criterion == "intensity":
                                residual          = af.abs(fields["forward_scattered_field"])**2 - measurement**2
                            elif self.configs.cost_criterion == "amplitude":
                                residual          = af.abs(fields["forward_scattered_field"]) - measurement
                        cost                     += af.sum(residual*af.conjg(residual)).real
                        #calculate gradient
                        if batch_update:
                            gradient_batch[:, :, :]  += self._computeGradient(fields, measurement)[0]
                        else:
                            gradient                  = self._computeGradient(fields, measurement)
                            obj_gpu[:, :, :]         -= stepsize * gradient

                        if verbose:
                            if self.number_illum > 1:
                                print("gradient update of illumination {:03d}/{:03d}.".format(illu_counter, self.number_illum), end="\r")
                                illu_counter += 1
                        fields      = None
                        residual    = None
                        gradient    = None
                        measurement = None
                        pupil       = None
                        af.device_gc()

                    if batch_update:
                        obj_gpu[:, :, :] -= stepsize * gradient_batch

                if np.isnan(obj_gpu).sum() > 0:
                    stepsize     *= 0.1
                    self.configs.time_elapsed = timer.elapsed
                    print("WARNING: Gradient update diverges! Resetting stepsize to %3.2f" %(stepsize))
                    t_k = 1.0
                    continue

                # L2 regularizer
                obj_gpu[:, :, :] -= stepsize * reg_term * obj_gpu

                #record total error
                self.configs.error.append(cost + reg_term * af.sum(obj_gpu*af.conjg(obj_gpu)).real)

                #Prox operators
                af.device_gc()
                obj_gpu = self._regularizer_obj.applyRegularizer(obj_gpu)

                if flag_FISTA:
                    #check convergence
                    if iteration > 0:
                        if self.configs.error[-1] > self.configs.error[-2]:
                            if restart:
                                t_k              = 1.0
                                
                                self._x[:, :, :] = y_k
                                # stepsize        *= 0.8
                                
                                print("WARNING: FISTA Restart! Error: %5.5f" %(np.log10(self.configs.error[-1])))
                                if run_callback:
                                    callback(self._x, self.configs)
                                continue
                            else:
                                print("WARNING: Error increased! Error: %5.5f" %(np.log10(self.configs.error[-1])))

                    #FISTA auxiliary variable
                    y_k1                 = np.array(obj_gpu)
                    if len(y_k1.shape) < 3:
                        y_k1 = y_k1[:,:,np.newaxis]

                    #FISTA update
                    t_k1                 = 0.5*(1.0 + (1.0 + 4.0*t_k**2)**0.5)
                    beta                 = (t_k - 1.0) / t_k1
                    self._x[:, :, :]     = y_k1 + beta * (y_k1 - y_k)
                    t_k                  = t_k1
                    y_k                  = y_k1.copy()
                else:
                    #check convergence
                    temp = np.array(obj_gpu)
                    if len(temp.shape) < 3:
                        temp = temp[:,:,np.newaxis]                    
                    self._x[:, :, :]  = temp
                    if iteration > 0:
                        if self.configs.error[-1] > self.configs.error[-2]:
                            print("WARNING: Error increased! Error: %5.5f" %(np.log10(self.configs.error[-1])))
                            stepsize     *= 0.8
                   
                if verbose:
                    print("iteration: %d/%d, error: %5.5f, elapsed time: %5.2f seconds" %(iteration+1, max_iter, np.log10(self.configs.error[-1]), timer.elapsed))

                if run_callback:
                    callback(self._x, self.configs)
                    
        self.configs.time_elapsed = timer.elapsed
        return self._x

    def solve(self, configs, measurements, \
              x_init = None, obj_support = None,\
              pupil = None, pupil_support = None,\
              verbose = True, callback = None):
        """
        function to solve for the tomography problem

        configs:        configs object from class AlgorithmConfigs
        measurements:   measurements in amplitude not INTENSITY, ordered by (x,y,defocus, illumination) (if configs.recon_from_field is False)
                        measurements of fields (if configs.recon_from_field is False)
        x_init:         initial guess for object
        verbose:        boolean variable to print verbosely
        """
        
        # Ensure all data is of appropriate dimension
        # Namely, measurements are [image_size_1,image_size_2,number_defocus,number_illum] 
        if self.number_defocus < 2:
            measurements               = measurements[:,:, np.newaxis]
        
        if self.number_illum < 2:
            measurements               = measurements[:, :, :, np.newaxis]
            
        self._initialization(configs = configs, x_init = x_init, obj_support = obj_support, pupil = pupil, pupil_support = pupil_support,\
                             verbose = verbose)  
        return self._algorithms[configs.method](measurements, verbose, callback)