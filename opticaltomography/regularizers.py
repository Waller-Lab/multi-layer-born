"""
Regularizer class for that also supports GPU code

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
March 04, 2018
"""

import arrayfire as af
import numpy as np
from opticaltomography import settings

np_complex_datatype = settings.np_complex_datatype
np_float_datatype   = settings.np_float_datatype
af_float_datatype   = settings.af_float_datatype
af_complex_datatype = settings.af_complex_datatype

class Regularizer:
	"""
	Highest-level Regularizer class that is responsible for parsing user arguments to create proximal operators
	All proximal operators operate on complex variables (real & imaginary part separately)
	Pure Amplitude:
        pure_amplitude: boolean, whether or not to enforce object to be purely amplitude
	Pure phase:
        pure_phase: boolean, whether or not to enforce object to be purely phase object        
	Pure Real:
		pure_real: boolean, whether or not to enforce object to be purely real
	
	Pure imaginary:
		pure_imag: boolean, whether or not to enforce object to be purely imaginary

	Positivity:
		positivity_real(positivity_imag): boolean, whether or not to enforce positivity for real(imaginary) part

	Negativity:
		negativity_real(negativity_imag): boolean, whether or not to enforce negativity for real(imaginary) part		
	
	LASSO (L1 regularizer):
		lasso: boolean, whether or not to use LASSO proximal operator
		lasso_parameter: threshold for LASSO

	Total variation (3D only):
		total_variation: boolean, whether or not to use total variation regularization
		total_variation_gpu: boolean, whether or not to use GPU implementation
		total_variation_parameter: scalar, regularization parameter (lambda)
		total_variation_maxitr: integer, number of each iteration for total variation		
	"""
	def __init__(self, configs = None, verbose = True, **kwargs):
		#Given all parameters, construct all proximal operators
		self.prox_list = []
		reg_params = kwargs
		if configs != None:
			reg_params = self._parseConfigs(configs)

		#Purely real
		if reg_params.get("pure_real", False):
			self.prox_list.append(PureReal())

		#Purely imaginary
		if reg_params.get("pure_imag", False):
			self.prox_list.append(Pureimag())

		#Purely amplitude object
		if reg_params.get("pure_amplitude", False):
			self.prox_list.append(PureAmplitude())
        
		#Purely phase object
		if reg_params.get(("pure_phase"), False):
			self.prox_list.append(PurePhase())
            
		#Total Variation
		if reg_params.get("total_variation", False):
			if reg_params.get("total_variation_gpu", False):
				self.prox_list.append(TotalVariationGPU(**reg_params))
			else:
				self.prox_list.append(TotalVariationCPU(**reg_params))

		#L1 Regularizer (LASSO)
		elif reg_params.get("lasso", False):
			self.prox_list.append(Lasso(reg_params.get("lasso_parameter", 1.0)))		
		
		#Others
		else:
			#Positivity
			positivity_real = reg_params.get("positivity_real", False)
			positivity_imag = reg_params.get("positivity_imag", False)
			if positivity_real or positivity_imag:
				self.prox_list.append(Positivity(positivity_real, positivity_imag))

			#Negativity
			negativity_real = reg_params.get("negativity_real", False)
			negativity_imag = reg_params.get("negativity_imag", False)
			if negativity_real or negativity_imag:
				self.prox_list.append(Negativity(negativity_real, negativity_imag))

		if verbose:
			for prox_op in self.prox_list:
				print("Regularizer -", prox_op.proximal_name)

	def _parseConfigs(self, configs):
		params = {}
		params["pure_real"]                 = configs.pure_real
		params["pure_imag"]                 = configs.pure_imag
		#Total variation
		params["total_variation"]           = configs.total_variation
		params["total_variation_gpu"]       = configs.total_variation_gpu
		params["total_variation_maxitr"]    = configs.max_iter_tv
		params["total_variation_order"]     = configs.order_tv
		params["total_variation_parameter"] = configs.reg_tv
		#LASSO
		params["lasso"]                     = configs.lasso
		params["lasso_parameter"]           = configs.reg_lasso
		#Pure amplitude/Pure phase
		params["pure_amplitude"]            = configs.pure_amplitude
		params["pure_phase"]                = configs.pure_phase        
        #Positivity/Negativity
		if configs.positivity_real[0]:
			if configs.positivity_real[1] == "larger":
				params["positivity_real"] = True
			elif configs.positivity_real[1] == "smaller":
				params["negativity_real"] = True
			else:
				raise ValueError("positivity/negativity parameter not recognized")
				
		if configs.positivity_imag[0]:
			if configs.positivity_imag[1] == "larger":
				params["positivity_imag"] = True
			elif configs.positivity_imag[1] == "smaller":
				params["negativity_imag"] = True
			else:
				raise ValueError("positivity/negativity parameter not recognized")
		return params

	def computeCost(self, x):
		cost = 0.0
		for prox_op in self.prox_list:
			cost_temp = prox_op.computeCost(x)
			if cost_temp != None:
				cost += cost_temp
		return cost

	def applyRegularizer(self, x):
		for prox_op in self.prox_list:
			x = prox_op.computeProx(x)
		return x

class ProximalOperator():
	def __init__(self, proximal_name):
		self.proximal_name = proximal_name
	def computeCost(self):
		pass
	def computeProx(self):
		pass	
	def setParameter(self):
		pass
	def _boundRealValue(self, x, value = 0, flag_project = True):
		"""If flag is true, only values that are greater than 'value' are preserved"""
		if flag_project:
			x[x < value] = 0
		return x		

class TotalVariationGPU(ProximalOperator):
	def __init__(self, **kwargs):
		proximal_name       = "Total Variation"
		parameter           = kwargs.get("total_variation_parameter", 1.0)
		maxitr              = kwargs.get("total_variation_maxitr",   15)
		self.order          = kwargs.get("total_variation_order",    1)
		self.pure_real      = kwargs.get("pure_real",                False)
		self.pure_imag      = kwargs.get("pure_imag",                False)
		self.pure_amplitude = kwargs.get("pure_amplitude",           False)
		self.pure_phase     = kwargs.get("pure_phase",               False)        
        
		#real part
		if kwargs.get("positivity_real", False):
			self.realProjector = lambda x: self._boundRealValue(x, 0, True)
			proximal_name      = "%s+%s" % (proximal_name, "positivity_real")
		elif kwargs.get("negativity_real", False):
			self.realProjector = lambda x: -1.0 * self._boundRealValue(-1.0 * x, 0, True)
			proximal_name      = "%s+%s" % (proximal_name, "negativity_real")
		else:
			self.realProjector = lambda x: x

		#imaginary part
		if kwargs.get("positivity_imag", False):
			self.imagProjector = lambda x: self._boundRealValue(x, 0, True)
			proximal_name      = "%s+%s" % (proximal_name, "positivity_imag")
		elif kwargs.get("negativity_imag", False):
			self.imagProjector = lambda x: -1.0 * self._boundRealValue(-1.0 * x, 0, True)
			proximal_name      = "%s+%s" % (proximal_name, "negativity_imag")
		else:
			self.imagProjector = lambda x: x
		self.setParameter(parameter, maxitr)
		super().__init__(proximal_name)
	


	def setParameter(self, parameter, maxitr):
		self.parameter = parameter
		self.maxitr = maxitr

	def computeCost(self, x):
		return None
	
	def computeProx(self, x):		
		if self.pure_real:
			x = self._computeProxReal(af.real(x), self.realProjector) + 1.0j * 0.0
		elif self.pure_imag:
			x = 1.0j *self._computeProxReal(af.imag(x), self.imagProjector)
		elif self.pure_amplitude:
			x = self._computeProxReal(af.abs(x), self.realProjector) * af.exp(1.0j * 0 * x)
		elif self.pure_phase:
			x = af.exp(1.0j * self._computeProxReal(af.arg(x), self.realProjector))
		else:
			# x = self._computeProxReal(af.real(x), self.realProjector) + \
			# 	1.0j * self._computeProxReal(af.imag(x), self.imagProjector)
			temp = self._computeProxReal(af.real(x), self.realProjector) + 1.0j * 0.0
			self.setParameter(self.parameter / 1.0, self.maxitr)
			x    = temp + 1.0j * self._computeProxReal(af.imag(x), self.imagProjector)
			self.setParameter(self.parameter * 1.0, self.maxitr)
		return x
	
	def _indexLastAxis(self, x, index = 0):
		if len(x.shape) == 3:
			return x[:,:,index]
		elif len(x.shape) == 4:
			return x[:,:,:,index]
		return 

	def _computeTVNorm(self, x):
			x_norm             = x**2
			x_norm             = af.sum(x_norm, dim = 3 if len(x.shape) == 4 else 2)**0.5
			x_norm[x_norm<1.0] = 1.0
			return x_norm		

	def _filterD(self, x, axis):
		assert axis<3, "This function only supports matrix up to 3 dimension!"
		if len(x.shape) == 2:
			if self.order == 1:
				if axis == 0:
					Dx     = x - af.shift(x,  1, 0)
				elif axis == 1:
					Dx     = x - af.shift(x,  0, 1)
			elif self.order == 2:
				if axis == 0:
					Dx     = x - 2*af.shift(x,  1, 0) + af.shift(x, 2, 0)
				elif axis == 1:
					Dx     = x - 2*af.shift(x,  0, 1) + af.shift(x, 0, 2)
			elif self.order == 3:
				if axis == 0:
					Dx     = x - 3*af.shift(x,  1, 0) + 3*af.shift(x, 2, 0) - af.shift(x, 3, 0)
				elif axis == 1:
					Dx     = x - 3*af.shift(x,  0, 1) + 3*af.shift(x, 0, 2) - af.shift(x, 0, 3)
			else:
				raise NotImplementedError("filter orders larger than 3 are not implemented!")			
		elif len(x.shape) == 3:
			if self.order == 1:
				if axis == 0:
					Dx     = x - af.shift(x,  1, 0, 0)
				elif axis == 1:
					Dx     = x - af.shift(x,  0, 1, 0)
				else:
					Dx     = x - af.shift(x,  0, 0, 1)
			elif self.order == 2:
				if axis == 0:
					Dx     = x - 2*af.shift(x,  1, 0, 0) + af.shift(x, 2, 0, 0)
				elif axis == 1:
					Dx     = x - 2*af.shift(x,  0, 1, 0) + af.shift(x, 0, 2, 0)
				else:
					Dx     = x - 2*af.shift(x,  0, 0, 1) + af.shift(x, 0, 0, 2)
			elif self.order == 3:
				if axis == 0:
					Dx     = x - 3*af.shift(x,  1, 0, 0) + 3*af.shift(x, 2, 0, 0) - af.shift(x, 3, 0, 0)
				elif axis == 1:
					Dx     = x - 3*af.shift(x,  0, 1, 0) + 3*af.shift(x, 0, 2, 0) - af.shift(x, 0, 3, 0)
				else:
					Dx     = x - 3*af.shift(x,  0, 0, 1) + 3*af.shift(x, 0, 0, 2) - af.shift(x, 0, 0, 3)
			else:
				raise NotImplementedError("filter orders larger than 3 are not implemented!")
		return Dx

	def _filterDT(self, x):
		if self.order == 1:
			if len(x.shape) == 3:
				DTx    = self._indexLastAxis(x, 0) - af.shift(self._indexLastAxis(x, 0), -1, 0) + \
			             self._indexLastAxis(x, 1) - af.shift(self._indexLastAxis(x, 1), 0, -1)
			elif len(x.shape) == 4:
				DTx    = self._indexLastAxis(x, 0) - af.shift(self._indexLastAxis(x, 0), -1, 0, 0) + \
			             self._indexLastAxis(x, 1) - af.shift(self._indexLastAxis(x, 1), 0, -1, 0) + \
			             self._indexLastAxis(x, 2) - af.shift(self._indexLastAxis(x, 2), 0, 0, -1)
		elif self.order == 2:
			if len(x.shape) == 3:
				DTx    = self._indexLastAxis(x, 0) - 2*af.shift(self._indexLastAxis(x, 0), -1, 0) + af.shift(self._indexLastAxis(x, 0), -2, 0) + \
						 self._indexLastAxis(x, 1) - 2*af.shift(self._indexLastAxis(x, 1), 0, -1) + af.shift(self._indexLastAxis(x, 1), 0, -2)
			elif len(x.shape) == 4:			
				DTx    = self._indexLastAxis(x, 0) - 2*af.shift(self._indexLastAxis(x, 0), -1, 0, 0) + af.shift(self._indexLastAxis(x, 0), -2, 0, 0) + \
						 self._indexLastAxis(x, 1) - 2*af.shift(self._indexLastAxis(x, 1), 0, -1, 0) + af.shift(self._indexLastAxis(x, 1), 0, -2, 0) + \
						 self._indexLastAxis(x, 2) - 2*af.shift(self._indexLastAxis(x, 2), 0, 0, -1) + af.shift(self._indexLastAxis(x, 2), 0, 0, -2)
		elif self.order == 3:
			if len(x.shape) == 3:
				DTx    = self._indexLastAxis(x, 0) - 3*af.shift(self._indexLastAxis(x, 0), -1, 0) + 3*af.shift(self._indexLastAxis(x, 0), -2, 0) - af.shift(self._indexLastAxis(x, 0), -3, 0) + \
						 self._indexLastAxis(x, 1) - 3*af.shift(self._indexLastAxis(x, 1), 0, -1) + 3*af.shift(self._indexLastAxis(x, 1), 0, -2) - af.shift(self._indexLastAxis(x, 1), 0, -3)
			elif len(x.shape) == 4:					
				DTx    = self._indexLastAxis(x, 0) - 3*af.shift(self._indexLastAxis(x, 0), -1, 0, 0) + 3*af.shift(self._indexLastAxis(x, 0), -2, 0, 0) - af.shift(self._indexLastAxis(x, 0), -3, 0, 0) + \
						 self._indexLastAxis(x, 1) - 3*af.shift(self._indexLastAxis(x, 1), 0, -1, 0) + 3*af.shift(self._indexLastAxis(x, 1), 0, -2, 0) - af.shift(self._indexLastAxis(x, 1), 0, -3, 0) + \
						 self._indexLastAxis(x, 2) - 3*af.shift(self._indexLastAxis(x, 2), 0, 0, -1) + 3*af.shift(self._indexLastAxis(x, 2), 0, 0, -2) - af.shift(self._indexLastAxis(x, 2), 0, 0, -3)
		else:
			raise NotImplementedError("filter orders larger than 3 are not implemented!")
		return DTx

	def _computeProxReal(self, x, projector):
		t_k        = 1.0
		
		def _gradUpdate():
		    grad_u_hat = x - self.parameter * self._filterDT(u_k1)
		    return grad_u_hat

		if len(x.shape) == 2:
			u_k        = af.constant(0.0, x.shape[0], x.shape[1], 2, dtype = af_float_datatype)
			u_k1       = af.constant(0.0, x.shape[0], x.shape[1], 2, dtype = af_float_datatype)
			# grad_u_hat = af.constant(0.0, x.shape[0], x.shape[1], dtype = af_float_datatype)
		elif len(x.shape) == 3:
			u_k        = af.constant(0.0, x.shape[0], x.shape[1], x.shape[2], 3, dtype = af_float_datatype)
			u_k1       = af.constant(0.0, x.shape[0], x.shape[1], x.shape[2], 3, dtype = af_float_datatype)
			# grad_u_hat = af.constant(0.0, x.shape[0], x.shape[1], x.shape[2], dtype = af_float_datatype)

		for iteration in range(self.maxitr):
			if iteration > 0:
				grad_u_hat  = _gradUpdate()
			else:
				grad_u_hat  = x.copy()

			grad_u_hat         = projector(grad_u_hat)
			if len(x.shape) == 2: #2D case
				u_k1[:,:, 0]   = self._indexLastAxis(u_k1, 0) + (1.0/(8.0)**self.order/self.parameter) * self._filterD(grad_u_hat, axis=0)
				u_k1[:,:, 1]   = self._indexLastAxis(u_k1, 1) + (1.0/(8.0)**self.order/self.parameter) * self._filterD(grad_u_hat, axis=1)
			elif len(x.shape) == 3: #3D case
				u_k1[:,:,:, 0] = self._indexLastAxis(u_k1, 0) + (1.0/(12.0)**self.order/self.parameter) * self._filterD(grad_u_hat, axis=0)
				u_k1[:,:,:, 1] = self._indexLastAxis(u_k1, 1) + (1.0/(12.0)**self.order/self.parameter) * self._filterD(grad_u_hat, axis=1)			
				u_k1[:,:,:, 2] = self._indexLastAxis(u_k1, 2) + (1.0/(12.0)**self.order/self.parameter) * self._filterD(grad_u_hat, axis=2)
			grad_u_hat = None
			u_k1_norm          = self._computeTVNorm(u_k1)
			if len(x.shape) == 2: #2D case			
				u_k1[:,:, 0]  /= u_k1_norm
				u_k1[:,:, 1]  /= u_k1_norm
			if len(x.shape) == 3: #3D case
				u_k1[:,:,:, 0]/= u_k1_norm
				u_k1[:,:,:, 1]/= u_k1_norm			
				u_k1[:,:,:, 2]/= u_k1_norm
			u_k1_norm 		   = None
			t_k1               = 0.5 * (1.0 + (1.0 + 4.0*t_k**2)**0.5)
			beta               = (t_k - 1.0)/t_k1
			if len(x.shape) == 2: #2D case
				temp = u_k[:,:,0].copy()
				if iteration < self.maxitr - 1:
					u_k[:,:,0] = u_k1[:,:,0]
				u_k1[:,:,0] =  (1.0 + beta)*u_k1[:,:,0] - beta*temp #now u_hat
				temp = u_k[:,:,1].copy()
				if iteration < self.maxitr - 1:
					u_k[:,:,1] = u_k1[:,:,1]
				u_k1[:,:,1] =  (1.0 + beta)*u_k1[:,:,1] - beta*temp
			elif len(x.shape) == 3: #3D case
				temp = u_k[:,:,:,0].copy()
				if iteration < self.maxitr - 1:
					u_k[:,:,:,0] = u_k1[:,:,:,0]
				u_k1[:,:,:,0] =  (1.0 + beta)*u_k1[:,:,:,0] - beta*temp #now u_hat
				temp = u_k[:,:,:,1].copy()
				if iteration < self.maxitr - 1:
					u_k[:,:,:,1] = u_k1[:,:,:,1]
				u_k1[:,:,:,1] =  (1.0 + beta)*u_k1[:,:,:,1] - beta*temp
				temp = u_k[:,:,:,2].copy()
				if iteration < self.maxitr - 1:
					u_k[:,:,:,2] = u_k1[:,:,:,2]
				u_k1[:,:,:,2] =  (1.0 + beta)*u_k1[:,:,:,2] - beta*temp
			temp = None

		grad_u_hat = projector(_gradUpdate())
		u_k 	   = None
		u_k1 	   = None		
		return grad_u_hat

class TotalVariationCPU(TotalVariationGPU):
	def _computeTVNorm(self, x):
		u_k1_norm           	 = af.to_array(x)
		u_k1_norm[:]       	    *= u_k1_norm
		u_k1_norm                = af.sum(u_k1_norm, dim = 3 if len(x.shape) == 4 else 2)**0.5
		u_k1_norm[u_k1_norm<1.0] = 1.0
		return np.array(u_k1_norm)
	
	def computeProx(self, x):		
		if self.pure_real:
			x = self._computeProxReal(np.real(x), self.realProjector) + 1.0j * 0.0
		elif self.pure_imag:
			x = 1.0j *self._computeProxReal(np.imag(x), self.imagProjector)
		else:
			x = self._computeProxReal(np.real(x), self.realProjector) \
			    + 1.0j * self._computeProxReal(np.imag(x), self.imagProjector)
		return af.to_array(x)

	def _computeProxReal(self, x, projector):
		t_k        = 1.0
		u_k        = np.zeros(x.shape + (3 if len(x.shape) == 3 else 2,), dtype = np_float_datatype);
		u_k1       = u_k.copy()
		u_hat      = u_k.copy()

		def _gradUpdate():
			u_hat_af   = af.to_array(u_hat)
			if len(x.shape) == 2:
				DTu_hat    = self._indexLastAxis(u_hat_af, 0) - af.shift(self._indexLastAxis(u_hat_af, 0), -1, 0) + \
				             self._indexLastAxis(u_hat_af, 1) - af.shift(self._indexLastAxis(u_hat_af, 1), 0, -1)
			elif len(x.shape) == 3:
				DTu_hat    = self._indexLastAxis(u_hat_af, 0) - af.shift(self._indexLastAxis(u_hat_af, 0), -1, 0, 0) + \
				             self._indexLastAxis(u_hat_af, 1) - af.shift(self._indexLastAxis(u_hat_af, 1), 0, -1, 0) + \
				             self._indexLastAxis(u_hat_af, 2) - af.shift(self._indexLastAxis(u_hat_af, 2), 0, 0, -1)
			grad_u_hat = x - np.array(self.parameter * DTu_hat)
			return grad_u_hat

		for iteration in range(self.maxitr):
			if iteration > 0:
				grad_u_hat  = _gradUpdate()
			else:
				grad_u_hat  = x.copy()

			grad_u_hat         = projector(grad_u_hat)
			u_k1[..., 0]  = u_hat[..., 0] + (1.0/12.0/self.parameter) * (grad_u_hat-np.roll(grad_u_hat, 1, axis = 0))
			u_k1[..., 1]  = u_hat[..., 1] + (1.0/12.0/self.parameter) * (grad_u_hat-np.roll(grad_u_hat, 1, axis = 1))
			if len(x.shape) == 3:
				u_k1[..., 2]  = u_hat[..., 2] + (1.0/12.0/self.parameter) * (grad_u_hat-np.roll(grad_u_hat, 1, axis = 2))
			u_k1_norm     = self._computeTVNorm(u_k1)
			u_k1[:]      /= u_k1_norm[..., np.newaxis]
			t_k1          = 0.5 * (1.0 + (1.0 + 4.0*t_k**2)**0.5)
			beta          = (t_k - 1.0)/t_k1
			u_hat         = (1.0 + beta)*u_k1 - beta*u_k
			if iteration < self.maxitr - 1:
				u_k            = u_k1.copy()
		return projector(_gradUpdate())



class Positivity(ProximalOperator):
	"""Enforce positivity constraint on a complex variable's real & imaginary part."""
	def __init__(self, positivity_real, positivity_imag, proximal_name = "Positivity"):
		super().__init__(proximal_name)
		self.real = positivity_real
		self.imag = positivity_imag

	def computeCost(self, x):
		return None

	def computeProx(self, x):
		if type(x).__module__ == "arrayfire.array":
			x = self._boundRealValue(af.real(x), 0, self.real) +\
                      1.0j * self._boundRealValue(af.imag(x), 0, self.imag)
		else:
			x = self._boundRealValue(x.real, 0, self.real) +\
                      1.0j * self._boundRealValue(x.imag, 0, self.imag)
		return x

class Negativity(Positivity):
	"""Enforce positivity constraint on a complex variable's real & imaginary part."""
	def __init__(self, negativity_real, negativity_imag):
		super().__init__(negativity_real, negativity_imag, "Negativity")

	def computeProx(self, x):
		return (-1.) * super().computeProx((-1.) * x)

class PureReal(ProximalOperator):
	"""Enforce real constraint on a complex, imaginary part will be cleared"""
	def __init__(self):
		super().__init__("Pure real")

	def computeCost(self, x):
		return None

	def computeProx(self, x):	
		if type(x).__module__ == "arrayfire.array":
		    x = af.real(x) + 1j*0.0
		else:
		    x = x.real + 1j*0.0
		return x

class Pureimag(ProximalOperator):
	"""Enforce imaginary constraint on a complex, real part will be cleared"""	
	def __init__(self):
		super().__init__("Pure imaginary")

	def computeCost(self, x):
		return None

	def computeProx(self, x):
		if type(x).__module__ == "arrayfire.array":
		    x = 1j*af.imag(x)
		else:
		    x = 1j*x.imag
		return x

class Lasso(ProximalOperator):
	"""||x||_1 regularizer, soft thresholding with certain parameter"""
	def __init__(self, parameter):	
		super().__init__("LASSO")
		self.setParameter(parameter)

	def _softThreshold(self, x):
		if type(x).__module__ == "arrayfire.array":
			#POTENTIAL BUG: af.sign implementation does not agree with documentation
			x = (af.sign(x)-0.5)*(-2.0) * (af.abs(x) - self.parameter) * (af.abs(x) > self.parameter)
		else:
			x = np.sign(x) * (np.abs(x) - self.parameter) * (np.abs(x) > self.parameter)
		return x

	def setParameter(self, parameter):		
		self.parameter = parameter

	def computeCost(self, x):
		return af.norm(af.moddims(x, np.prod(x.shape)), norm_type = af.NORM.VECTOR_1)

	def computeProx(self, x):	
		if type(x).__module__ == "arrayfire.array":
			x = self._softThreshold(af.real(x)) + 1.0j * self._softThreshold(af.imag(x))
		else:
		    x = self._softThreshold(x.real) + 1.0j * self._softThreshold(x.imag)
		return x		

#TODO: implement Tikhonov
class Tikhonov(ProximalOperator):
	def __init__(self):	
		pass
	def setParameter(self, parameter):				
		self.parameter = parameter
	def computeCost(self, x):
		pass
	def computeProx(self, x):	
		return x

class PureAmplitude(ProximalOperator):
	def __init__(self):
		super().__init__("Purely Amplitude")    
	def computeCost(self, x):
		return None
	def computeProx(self, x):	
		return af.abs(x) * af.exp(1.0j * 0 * x)

class PurePhase(ProximalOperator):
	def __init__(self):
		super().__init__("Purely Phase")    
	def computeCost(self, x):
		return None
	def computeProx(self, x):	
		return af.exp(1.0j*af.arg(x))

