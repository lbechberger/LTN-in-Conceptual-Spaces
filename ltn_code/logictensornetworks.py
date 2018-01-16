__author__ = 'luciano'
import tensorflow as tf
import numpy as np

default_layers = 5                      # number of receptive fields per predicate
default_smooth_factor = 1e-7            # factor to which large weights are penalized
default_tnorm = "product"               # appropriate t-conorm is used to compute disjunction of literals within clauses; 'product', 'yager2', 'luk', 'goedel'
default_aggregator = "min"              # aggregation across data points when computing validity of a clause; 'product', 'mean', 'gmean', 'hmean', 'min'
default_clauses_aggregator = "min"      # aggregate over clauses to define overall satisfiability of KB; 'min', 'mean', 'hmean', 'wmean'
default_optimizer = "gd"                # optimizing algorithm to use; 'ftrl', 'gd', 'ada', 'rmsprop'
default_positive_fact_penality = 1e-6   # penalty for predicates that are true everywhere
default_norm_of_u = 5.0                 # initialization of the u vector (determining how close to 0 and 1 the membership values can get)
default_type = "original"               # default type of membership function to use; 'original', 'rbfDistribution', 'rbfDistance', 'linear', 'cuboid'
default_epsilon = 1e-4                  # smoothing parameter for covariance matrix of 'rbf' type

def train_op(loss, optimization_algorithm):
    if optimization_algorithm == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.01,learning_rate_power=-0.5)
    if optimization_algorithm == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    if optimization_algorithm == "ada":
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    if optimization_algorithm == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01,decay=0.9)
    return optimizer.minimize(loss)

def PR(tensor):
    np.set_printoptions(threshold=np.nan)
    result = tf.Print(tensor,[tf.shape(tensor),tensor.name,tensor],summarize=20)
    return result

def apply_t_norm(literals, label="no_label"):
    list_of_literal_tensors = [lit.tensor for lit in literals]
    literals_tensor = tf.concat(list_of_literal_tensors,1)
    if default_tnorm == "product":
        result = 1.0-tf.reduce_prod(1.0-literals_tensor,1,keep_dims=True)
    if default_tnorm=="yager2":
        result = tf.minimum(1.0,tf.sqrt(tf.reduce_sum(tf.square(literals_tensor),1, keep_dims=True)))
    if default_tnorm=="luk":
        result = tf.minimum(1.0,tf.reduce_sum(literals_tensor,1, keep_dims=True))
    if default_tnorm == "goedel":
        result = tf.reduce_max(literals_tensor,1,keep_dims=True,name=label)
    return result

def disjunction_of_literals(literals,label="no_label"):
    result = apply_t_norm(literals, label)
    if default_aggregator == "product":
        return tf.reduce_prod(result,keep_dims=True,name=label)
    if default_aggregator == "mean":
        return tf.reduce_mean(result,keep_dims=True,name=label)
    if default_aggregator == "gmean":
        return tf.exp(tf.multiply(tf.reduce_sum(tf.log(result), keep_dims=True),
                             tf.reciprocal(tf.to_float(tf.size(result)))),name=label)
    if default_aggregator == "hmean":
        return tf.div(tf.to_float(tf.size(result)),tf.reduce_sum(tf.reciprocal(result),keep_dims=True),name=label)
    if default_aggregator == "min":
        return tf.reduce_min(result, keep_dims=True,name=label)

def smooth(parameters):
    
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(
                     [tf.expand_dims(tf.reduce_sum(tf.square(par)),0) for par in parameters],0),1))
    return tf.multiply(default_smooth_factor,norm_of_omega)

class Domain:
    def __init__(self,columns, dom_type="float",label=None):
        self.dom_type = dom_type
        self.columns = columns
        self.label = label
        self.parameters = []
        self.tensor = tf.placeholder(self.dom_type,
                                        shape=[None, self.columns],
                                        name=self.label)

class Domain_concat(Domain):

    def __init__(self, domains):
        self.columns = sum([dom.columns for dom in domains])
        self.label = "concatenation_of_" + "_".join([dom.label for dom in domains])
        self.parameters = [par for dom in domains for par in dom.parameters]
        self.domains = domains
        self.tensor = tf.concat([dom.tensor for dom in self.domains],1)

class Domain_union(Domain):

    def __init__(self, domains):
        self.columns = domains[0].columns
        self.label = "union_of_" + "_".join([dom.label for dom in domains])
        self.parameters = [par for dom in domains for par in dom.parameters]
        self.domains = domains
        self.tensor = tf.concat([dom.tensor for dom in self.domains],0)

class Domain_slice(Domain):

    def __init__(self, domain, begin_column, end_column):
        self.columns = end_column - begin_column
        self.label = "projection_of_" + domain.label + "_from_column_"+str(begin_column) + "_to_column_" + str(end_column)
        self.parameters = domain.parameters
        self.domain = domain
        self.begin_column = begin_column
        self.end_column = end_column
        self.tensor = domain.tensor[:,begin_column:end_column]

class Constant(Domain):
    def __init__(self, label, value=None, domain=None, init_with=None):
        self.label = label
        if value != None:
            self.tensor = tf.constant([value],dtype=tf.float32)
            self.parameters = []
            self.columns = len(value)
        else:
            self.columns = domain.columns
            if init_with is not None:
                self.tensor = tf.Variable(tf.expand_dims(np.array(init_with, dtype="float32"),0), name=label)
            else:
                self.tensor = tf.Variable(tf.random_normal([1, domain.columns], mean=1),name=label)
            self.parameters = []




class Function:
    def __init__(self, label, domain, range, value=None,mean=0.0,stddev=0.5):
        self.label = label
        self.in_columns = domain.columns
        self.columns = range.columns
        self.domain = domain
        if value is None:
            self.M = tf.Variable(tf.random_normal([self.in_columns+1,
                                                   self.in_columns+1],mean=mean,stddev=stddev),
                                 name = "M_"+self.label)
            self.N = tf.Variable(tf.random_normal([self.in_columns+1,
                                                   self.columns],mean=mean,stddev=stddev),
                                 name = "N_"+self.label)
            self.parameters = [self.M,self.N]
            self.is_defined = False
        else:
            self.parameters = []
            self.is_defined = True
            self.value = value

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        if self.is_defined:
            return apply(self.value,[domain])
        else:
            extended_domain = tf.concat([tf.ones((tf.shape(domain.tensor)[0],1)),
                                           domain.tensor],1)

            h = tf.nn.sigmoid(tf.matmul(extended_domain,self.M))
            return tf.matmul(h,self.N)

class Term(Domain):
    def __init__(self,function,domain):
        self.label = function.label+"_of_"+domain.label
        self.parameters = function.parameters + domain.parameters
        self.domain = domain
        self.function = function
        self.columns = function.columns
        self.tensor = self.function.tensor(self.domain)

class CompositePredicate:
    def __init__(self, label, domain, layers = None, ltn_type = None):
        self.label = label
        self.domain = domain
        if layers == None:
            self.number_of_layers = default_layers
        else:
            self.number_of_layers = layers
        
        if ltn_type == None:
            self.ltn_type = default_type
        else:
            self.ltn_type = ltn_type
            
        self.predicates = []
        self.parameters = []
        for i in range(self.number_of_layers):
            pred = Predicate("{0}_{1}".format(label, i), self.domain, layers = 1, ltn_type = self.ltn_type)
            self.predicates.append(pred)
            self.parameters += pred.parameters
        print self.predicates
        
    def tensor(self, domain = None):
        if domain is None:
            domain = self.domain
        return self.predicates[0].tensor(domain) 
        #return tf.reduce_max(tf.concat([p.tensor() for p in self.predicates], 0), keep_dims = True)
        #return Clause([Literal(True, pred, domain) for pred in self.predicates], label="clause"+self.label)
        list_of_pred_tensors = [p.tensor(self.domain) for p in self.predicates]
        literals_tensor = tf.concat(list_of_pred_tensors,1)
        if default_tnorm == "product":
            result = 1.0-tf.reduce_prod(1.0-literals_tensor,1,keep_dims=True)
        if default_tnorm=="yager2":
            result = tf.minimum(1.0,tf.sqrt(tf.reduce_sum(tf.square(literals_tensor),1, keep_dims=True)))
        if default_tnorm=="luk":
            result = tf.minimum(1.0,tf.reduce_sum(literals_tensor,1, keep_dims=True))
        if default_tnorm == "goedel":
            result = tf.reduce_max(literals_tensor,1,keep_dims=True,name=self.label)   
        return result
        
    def constraints(self):
        point = Constant(label = self.label + "_prototype", domain = self.domain)
        clauses = []
        for pred in self.predicates:
            clauses.append(Clause([Literal(True, pred, point)]))
        return clauses

class Predicate:
    def __init__(self, label, domain, layers = None, max_min = 0.0, ltn_type = None, data_points = None):
        self.label = label
        self.max_min = max_min
        self.domain = domain
        # modification by lbechberger: if we use default_layers as default value in method header, overwriting it from external
        # files doesn't change anything, so we need to check for it during runtime
        if layers == None:
            self.number_of_layers = default_layers
        else:
            self.number_of_layers = layers

        # modification by lbechberger: take care of different ltn types (i.e., different membership functions)            
        if ltn_type == None:
            self.ltn_type = default_type
        else:
            self.ltn_type = ltn_type
        
        if self.ltn_type == "original":
            self.W = tf.matrix_band_part(tf.Variable(tf.random_normal([self.number_of_layers,
                                                  self.domain.columns+1,
                                                  self.domain.columns+1],stddev=0.5)),0,-1 ,
                                 name = "W"+label)  # upper triangualr matrix
            # modification by lbechberger: instead of using tf.ones, use tf.constant() to make sure that norm of u is ~5
            # (which in turn ensures that the membership function can reach values close to 0 and 1)
            self.u = tf.Variable(tf.constant(default_norm_of_u/self.number_of_layers, shape=[self.number_of_layers,1]), name = "u"+label)
            self.parameters = [self.W]
            
        elif self.ltn_type == "rbfDistribution":
            # mu is the center of the receptive field, W is the basis for the covariance matrix
            if data_points == None:
                self.W = tf.Variable(tf.eye(self.domain.columns, batch_shape=[self.number_of_layers]), name = "W"+label)
                self.mu = tf.Variable(tf.random_normal([self.number_of_layers, self.domain.columns]), name = "mu"+label)
            else:
                # make initial guess based on some data points
                self.mu = tf.Variable(tf.expand_dims(tf.reduce_mean(data_points, 0), 1))
                self.W = tf.Variable(tf.eye(self.domain.columns, batch_shape=[self.number_of_layers]), name = "W"+label)
            self.u = tf.Variable(tf.ones(shape=[self.number_of_layers,1]), name = "u"+label)
            self.parameters = [self.W]
            
        elif self.ltn_type == "rbfDistance":
            # W contains a weight for each dimension
            if data_points == None:
                self.W = tf.Variable(tf.constant(1.0/self.domain.columns, shape=[self.number_of_layers, self.domain.columns]), name = "W"+label)
                self.mu = tf.Variable(tf.random_normal([self.number_of_layers, self.domain.columns]), name = "mu"+label)
            else:
                # make initial guess based on some data points
                self.mu = tf.Variable(tf.expand_dims(tf.reduce_mean(data_points, 0), 1))
                self.W = tf.Variable(tf.constant(1.0/self.domain.columns, shape=[self.number_of_layers, self.domain.columns]), name = "W"+label)
            self.u = tf.Variable(tf.ones(shape=[self.number_of_layers,1]), name = "u"+label)
            self.parameters = [self.W]
            
        elif self.ltn_type == "linear":
            # use only a linear model
            self.V = tf.Variable(tf.random_normal([self.number_of_layers, self.domain.columns]), name = "V"+label)
            self.b = tf.Variable(tf.random_normal([1,self.number_of_layers]), name = "b"+label)
            self.u = tf.Variable(tf.constant(default_norm_of_u/self.number_of_layers, shape=[self.number_of_layers,1]), name = "u"+label)
            self.parameters = [self.V, self.b]   
            
        elif self.ltn_type == "cuboid":
            # use a single cuboid
#            if data_points == None:
#                self.p_min = tf.Variable(tf.random_normal([1,self.domain.columns]), name = "p_min_" + label)
#            else:
#                self.p_min = tf.Variable(tf.expand_dims(tf.reduce_mean(data_points, axis=0), 1))
#            self.point_1 = tf.Variable(tf.add(tf.reduce_min(data_points, axis=0, keep_dims=True), tf.random_normal(shape=[1,self.domain.columns], stddev=0.05)), name = self.label + "p_1")#tf.random_normal([1,self.domain.columns]))            
#            self.point_2 = tf.Variable(tf.add(tf.reduce_max(data_points, axis=0, keep_dims=True), tf.random_normal(shape=[1,self.domain.columns], stddev=0.05)), name = self.label + "p_2")#tf.Variable(tf.random_normal([1,self.domain.columns]))
            self.point_1 = tf.Variable(tf.add(tf.reduce_mean(data_points, axis=0, keep_dims=True), tf.random_normal(shape=[1,self.domain.columns], stddev=0.15)))#tf.random_normal([1,self.domain.columns]))            
            self.point_2 = tf.Variable(tf.add(tf.reduce_mean(data_points, axis=0, keep_dims=True), tf.random_normal(shape=[1,self.domain.columns], stddev=0.15)))#tf.Variable(tf.random_normal([1,self.domain.columns]))
            self.p_min = tf.minimum(self.point_1, self.point_2)
            self.p_max = tf.maximum(self.point_1, self.point_2)
            self.c = tf.abs(tf.Variable(tf.constant(10.0, shape=[1])))
            self.weights = tf.abs(tf.Variable(tf.ones(shape=[self.domain.columns]), name = "W"+label))
            self.parameters = [tf.abs(tf.subtract(self.point_1, self.point_2))]
            
        else:
            raise Exception("Unknown LTN type - cannot construct predicate")

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        
        # modification by lbechberger: take care of different ltn types (i.e., different membership functions)            
        if self.ltn_type == "original":
            X = tf.concat([tf.ones((tf.shape(domain.tensor)[0],1)), domain.tensor],1)
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.number_of_layers, 1, 1]), self.W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])),squeeze_dims=[1])
            gX = tf.matmul(tf.tanh(XWX),self.u)
            result = tf.sigmoid(gX,name=self.label+"_at_"+domain.label)
            
        elif self.ltn_type == "rbfDistribution":
            # compute covariance as WW' + eps*I (try to ensure positive definiteness)
            scaled_eye = tf.multiply(tf.eye(self.domain.columns), tf.constant(default_epsilon, shape=self.W.shape))
            covariance = tf.add(tf.matmul(self.W, self.W, transpose_b = True), scaled_eye)
            # multivariate normal distribution with mu and covariance; normalize pdf by prob(mu)            
            dist = tf.contrib.distributions.MultivariateNormalFullCovariance(self.mu, covariance)
            height = dist.prob(self.mu)
            X = tf.expand_dims(domain.tensor, 1)
            rbf = tf.multiply(dist.prob(X), tf.reciprocal(height))
            # take a linear combination of the different receptive fields based on u
            result = tf.matmul(rbf, tf.multiply(self.u, tf.reciprocal(tf.reduce_sum(self.u, 1, keep_dims=True))))
            
        elif self.ltn_type == "rbfDistance":
            # compute weighted distance to prototype
            X = tf.expand_dims(domain.tensor)
            difference = tf.abs(tf.subtract(self.mu, X))
            normalized_weights = tf.multiply(self.W, tf.reciprocal(tf.reduce_sum(self.W, 1, keep_dims=True)))
            weighted_difference = tf.multiply(normalized_weights, difference)
            distance = tf.reduce_sum(weighted_difference, 1)
            membership = tf.exp(-distance)
            # aggregate as max over all receptive fields (todo: also use weighted sum?)
            #result = tf.reduce_max(membership, keep_dims = True)
            result = tf.matmul(membership, tf.multiply(self.u, tf.reciprocal(tf.reduce_sum(self.u, 1, keep_dims=True))))
            
        elif self.ltn_type == "linear":
            # use a simple linear NN
            X = domain.tensor
            XV = tf.matmul(X, tf.transpose(self.V))
            gX = tf.matmul(tf.tanh(XV + self.b),self.u)
            result = tf.sigmoid(gX)
            print result.shape
            
        elif self.ltn_type == "cuboid":
            # use a single cuboid
            X = domain.tensor
            y = tf.maximum(self.p_min, tf.minimum(X, self.p_max))            
            normalized_weights = tf.divide(self.weights, tf.reduce_sum(self.weights))
            # need to add a small epsilon before doing sqrt such that gradient is always defined
            eucl_dist = tf.sqrt(tf.reduce_sum(tf.multiply(normalized_weights, tf.square(tf.subtract(X,y))),axis=1, keep_dims = True) + 1e-15)
            exp = tf.exp(-self.c * eucl_dist)
            result = exp
            
        else:
            raise Exception("Unknown LTN type - cannot compute membership")
        
        if self.max_min == 0.0:
            return result
        if self.max_min > 0.0:
            return tf.add(result,
                          tf.multiply(1-result,
                                 self.max_min))
        else:
            return tf.multiply(tf.add(1.0,self.max_min),result)

class In_range(Predicate):
    def __init__(self,domain,lower,upper,label="inrange",sharpness=10.0):
        self.label = label
        self.domain = domain
        self.parameters = []
        self.lower = tf.constant(lower,dtype=tf.float32)
        self.upper = tf.constant(upper,dtype=tf.float32)
        self.normalize = tf.square(tf.divide(tf.subtract(self.upper,
                                                         self.lower),2.0))
        self.sharpness = sharpness

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        return tf.reduce_min(
                 tf.nn.sigmoid(tf.multiply(self.sharpness,
                                      tf.divide(
                                          tf.multiply(tf.subtract(domain.tensor,self.lower),
                                                      tf.subtract(self.upper,domain.tensor)),
                                          self.normalize))),keep_dims=True)

class Less_than(Predicate):
    def __init__(self,domain1,domain2,label,sharpness=10.0):
        self.label = label
        self.domain = Domain_concat([domain1,domain2])
        self.parameters = []
        self.sharpness = sharpness

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        return tf.reduce_min(
            tf.nn.sigmoid(tf.multiply(self.sharpness,
                                     tf.subtract(domain.tensor[:,domain.tensor.shape[1]/2:],
                                                 domain.tensor[:,:domain.tensor.shape[1]/2]))),keep_dims=True)


class Literal:
    def __init__(self,polarity,predicate,domain=None):
        self.predicate = predicate
        self.polarity = polarity
        if domain is None:
            self.domain = predicate.domain
            self.parameters = predicate.parameters
        else:
            self.domain = domain
            self.parameters = predicate.parameters + domain.parameters
        if polarity:
            self.tensor = predicate.tensor(domain)
        else:
            self.tensor = 1-predicate.tensor(domain)
#        self.y = predicate.y
#        self.diff = predicate.diff
#        self.square = predicate.square
#        self.sum = predicate.sum
#        self.eucl_dist = predicate.eucl_dist
#        self.exp = predicate.exp

class Clause:
    def __init__(self,literals,label=None, weight=1.0):
        self.weight=weight
        self.label=label
        self.literals = literals
        self.tensor = disjunction_of_literals(self.literals,label=label)
        self.predicates = set([lit.predicate for lit in self.literals])
        self.parameters = [par for lit in literals for par in lit.parameters]

class KnowledgeBase:

    def __init__(self,label,clauses,save_path=""):
        print("defining the knowledge base",label)
        self.label = label
        self.clauses = clauses
        self.parameters = [par for cl in self.clauses for par in cl.parameters]
        if not self.clauses:
            self.tensor = tf.constant(1.0)
        else:
            clauses_value_tensor = tf.concat([cl.tensor for cl in clauses],0)
            if default_clauses_aggregator == "min":
                self.tensor = tf.reduce_min(clauses_value_tensor)
            if default_clauses_aggregator == "mean":
                self.tensor = tf.reduce_mean(clauses_value_tensor,name=label)
            if default_clauses_aggregator == "hmean":
                self.tensor = tf.squeeze(tf.div(tf.to_float(tf.size(clauses_value_tensor)),
                                        tf.reduce_sum(tf.reciprocal(clauses_value_tensor), keep_dims=True),name=label))
            if default_clauses_aggregator == "wmean":
                weights_tensor = tf.constant([cl.weight for cl in clauses])
                self.tensor = tf.div(tf.reduce_sum(tf.multiply(weights_tensor, clauses_value_tensor)),tf.reduce_sum(weights_tensor),name=label)
        if default_positive_fact_penality != 0:
            self.loss = tf.add(smooth(self.parameters),
                               (tf.multiply(default_positive_fact_penality,
                                      self.penalize_positive_facts()) -
                                self.tensor) ,name="Loss")
        else:
            self.loss = tf.subtract(smooth(self.parameters),self.tensor,name="Loss")
        self.save_path = save_path
        self.train_op = train_op(self.loss,default_optimizer)
        self.saver = tf.train.Saver()

    def penalize_positive_facts(self):
        tensor_for_positive_facts = [tf.reduce_sum(Literal(True,lit.predicate,lit.domain).tensor,keep_dims=True) for cl in self.clauses for lit in cl.literals]
        return tf.reduce_sum(tf.concat(tensor_for_positive_facts,0))

    def save(self,sess, version = ""):
        save_path = self.saver.save(sess,self.save_path+self.label+version+".ckpt")

    def restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self,sess,feed_dict={}):
        return sess.run(self.train_op,feed_dict)

    def is_nan(self,sess,feed_dict={}):
        return sess.run(tf.is_nan(self.tensor),feed_dict)

class Equal(Predicate):
    def __init__(self, label, domain,diameter=1.0):
        self.label = label
        self.domain = domain
        self.parameters = []
        self.diameter = diameter

    def tensor(self,dom=None):
        if dom is None:
            dom = self.domain
        dom1 = Domain_slice(dom,0,dom.columns/2)
        dom2 = Domain_slice(dom,dom.columns/2,dom.columns)
        delta = tf.sqrt(tf.reduce_sum(tf.square(dom1.tensor - dom2.tensor)  ,1,keep_dims=True))
        return 1.0 - tf.divide(delta,self.diameter)



