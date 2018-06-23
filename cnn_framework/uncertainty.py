import keras.backend as K
import numpy as np
from keras.engine.topology import Layer 
from keras.models import Model
import time

class EnsembleModel(Model):
    def __init__(self, seeds=[None], **kwargs):
        super(EnsembleModel, self).__init__(**kwargs)
        self.grad_model = None
        self.seeds = seeds
        self.weight_generators = None
        self.mask_id = 0
        
    def gen_mask(self, seed):
        rng = np.random.RandomState()
        if seed is not None:
            rng.seed(seed)
        for layer in self.layers:
            if 'gen_mask' in dir(layer):
                layer.gen_mask(rng)
        
    def reset_mask_id(self):
        self.mask_id = 0        

    def train_on_batch(self, x, y, **kwargs):   
        seed = np.random.choice(self.seeds)      
        self.gen_mask(seed)
        loss = super(EnsembleModel,self).train_on_batch(x, y, **kwargs)
        return loss

    
    def test_on_batch(self, x, y, **kwargs):   
        seed = np.random.choice(self.seeds)
        self.gen_mask(seed)
        loss = super(EnsembleModel,self).test_on_batch(x, y, **kwargs)
        return loss
    
    def test_model(self, x, y, **kwargs):   
        Y = []
        for j in range(len(self.seeds)):
            print 'mask {}'.format(j)
            self.gen_mask(self.seeds[j])
            Y += [super(EnsembleModel,self).predict(x, **kwargs)] 
        Y_avg = np.mean(Y,axis=0)
        Y_var = np.var(Y,axis=0)
        f = open('test_output.txt','w')
        for i, Y_true in enumerate(y):
            f.write('{} {} {}\n'.format(y[i], Y_avg[i][0], Y_var[i][0]))

    def predict(self, x, sigma=False, **kwargs):
        Y = []
        for j in range(len(self.seeds)):
            self.gen_mask(self.seeds[j])
            Y += [super(EnsembleModel,self).predict(x, **kwargs)] 
        Y_avg = np.mean(Y,axis=0)
        if sigma:
            Y_sigma = np.std(Y,axis=0)
            return Y_avg, Y_sigma
        return Y_avg   
    
    def get_config(self):
        config = super(EnsembleModel, self).get_config()
        config['seeds'] = self.seeds
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super(EnsembleModel, cls).from_config(config, custom_objects=custom_objects)
        model.seeds = config.get('seeds')
        return model
        
class RandomMask(Layer):
    """Applies Mask to the input.
    """
    
    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate       
        super(RandomMask, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        size = K.int_shape(x)[1:]
        self.mask = K.variable(np.ones(shape=size,dtype=np.float32))
        x *= self.mask
        return x
    
    def gen_mask(self, rng):
        retain_prob = 1.0 - self.dropout_rate
        size = K.int_shape(self.mask)
        K.set_value(self.mask,rng.binomial(n=1,p=retain_prob,size=size).astype(np.float32))
#        K.set_value(self.mask,np.random.binomial(n=1,p=retain_prob,size=size).astype(np.float32))
    
    def get_config(self):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(RandomMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
