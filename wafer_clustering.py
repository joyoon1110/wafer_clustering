'''Wafer VAE  + K_means Clustering '''

##### Package ######
import time
from datetime import datetime 
import ast
import natsort
import json
import shutil
import argparse
import glob
import os
import sys 
import tensorflow as tf
from imutils import paths
import cv2
import random
import numpy as np
from tensorflow import set_random_seed
from keras import backend as K
tf.logging.set_verbosity(tf.logging.INFO)




# image_path, cluster_number, max_number, hash, old_hash, auto, seed
parser = argparse.ArgumentParser()
parser.add_argument("-image_path", "--image_path", type=str, default=None,
                help='Image Path ')
parser.add_argument("-hash", "--hash", type=str, default='None',
                help='Chart Draw hash')
parser.add_argument("-cluster_number", "--cluster_number", type=int, default=-1,
                help='cluster_number ')
parser.add_argument("-max_number", "--max_number", type=int, default=-1,
                help='max_number ')
parser.add_argument("-auto", "--auto", type=str, default='true',
                help='Auto Finding for Cluster Number ')
parser.add_argument("-seed", "--seed", type=int, default=-1,
                help='seed')
args = parser.parse_args()




################################# Tensorflow Kmeans Cluster Function #################################

from tensorflow.contrib.factorization.python.ops import clustering_ops
from tensorflow.python.training import training_util
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModelFnOps
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.control_flow_ops import with_dependencies
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.util.deprecation import deprecated

_USE_TF_CONTRIB_FACTORIZATION = (
    'Please use tf.contrib.factorization.KMeansClustering instead of'
    ' tf.contrib.learn.KMeansClustering. It has a similar interface, but uses'
    ' the tf.estimator.Estimator API instead of tf.contrib.learn.Estimator.')


class _LossRelativeChangeHook(session_run_hook.SessionRunHook):
  """Stops when the change in loss goes below a tolerance."""

  def __init__(self, tolerance):
    """Initializes _LossRelativeChangeHook.

    Args:
      tolerance: A relative tolerance of change between iterations.
    """
    self._tolerance = tolerance
    self._prev_loss = None
    self._counting = 0

  def begin(self):
    self._loss_tensor = ops.get_default_graph().get_tensor_by_name(
        KMeansClustering.LOSS_OP_NAME + ':0')
    assert self._loss_tensor is not None

  def before_run(self, run_context):
    del run_context
    return SessionRunArgs(
        fetches={KMeansClustering.LOSS_OP_NAME: self._loss_tensor})

  def after_run(self, run_context, run_values):
    loss = run_values.results[KMeansClustering.LOSS_OP_NAME]
    assert loss is not None
    # if self._prev_loss is not None:
    #   relative_change = (abs(loss - self._prev_loss) /
    #                      (1 + abs(self._prev_loss)))
    #   if relative_change < self._tolerance:
    #     run_context.request_stop()
    if self._prev_loss:
      if self._prev_loss <= loss :
        self._counting= self._counting+1
      else:
        self._counting=0
      if self._counting == self._tolerance:
        run_context.request_stop()
    self._prev_loss = loss


class _InitializeClustersHook(session_run_hook.SessionRunHook):
  """Initializes clusters or waits for cluster initialization."""

  def __init__(self, init_op, is_initialized_op, is_chief):
    self._init_op = init_op
    self._is_chief = is_chief
    self._is_initialized_op = is_initialized_op

  def after_create_session(self, session, _):
    assert self._init_op.graph == ops.get_default_graph()
    assert self._is_initialized_op.graph == self._init_op.graph
    while True:
      try:
        if session.run(self._is_initialized_op):
          break
        elif self._is_chief:
          session.run(self._init_op)
        else:
          time.sleep(1)
      except RuntimeError as e:
        logging.info(e)


def _parse_tensor_or_dict(features):
  """Helper function to parse features."""
  if isinstance(features, dict):
    keys = sorted(features.keys())
    with ops.colocate_with(features[keys[0]]):
      features = array_ops.concat([features[k] for k in keys], 1)
  return features


def _kmeans_clustering_model_fn(features, labels, mode, params, config):
  """Model function for KMeansClustering estimator."""
  assert labels is None, labels
  (all_scores, model_predictions, losses,
   is_initialized, init_op, training_op) = clustering_ops.KMeans(
       _parse_tensor_or_dict(features),
       params.get('num_clusters'),
       initial_clusters=params.get('training_initial_clusters'),
       distance_metric=params.get('distance_metric'),
       use_mini_batch=params.get('use_mini_batch'),
       mini_batch_steps_per_iteration=params.get(
           'mini_batch_steps_per_iteration'),
       random_seed=params.get('random_seed'),
       kmeans_plus_plus_num_retries=params.get(
           'kmeans_plus_plus_num_retries')).training_graph()
  incr_step = state_ops.assign_add(training_util.get_global_step(), 1)
  loss = math_ops.reduce_sum(losses, name=KMeansClustering.LOSS_OP_NAME)
  summary.scalar('loss/raw', loss)
  training_op = with_dependencies([training_op, incr_step], loss)
  predictions = {
      KMeansClustering.ALL_SCORES: all_scores[0],
      KMeansClustering.CLUSTER_IDX: model_predictions[0],
  }
  eval_metric_ops = {KMeansClustering.SCORES: loss}
  training_hooks = [_InitializeClustersHook(
      init_op, is_initialized, config.is_chief)]
  relative_tolerance = params.get('relative_tolerance')
  if relative_tolerance is not None:
    training_hooks.append(_LossRelativeChangeHook(relative_tolerance))
  return ModelFnOps(
      mode=mode,
      predictions=predictions,
      eval_metric_ops=eval_metric_ops,
      loss=loss,
      train_op=training_op,
      training_hooks=training_hooks)


# TODO(agarwal,ands): support sharded input.
class KMeansClustering(estimator.Estimator):
  """An Estimator for K-Means clustering.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """
  SQUARED_EUCLIDEAN_DISTANCE = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
  COSINE_DISTANCE = clustering_ops.COSINE_DISTANCE
  RANDOM_INIT = clustering_ops.RANDOM_INIT
  KMEANS_PLUS_PLUS_INIT = clustering_ops.KMEANS_PLUS_PLUS_INIT
  SCORES = 'scores'
  CLUSTER_IDX = 'cluster_idx'
  CLUSTERS = 'clusters'
  ALL_SCORES = 'all_scores'
  LOSS_OP_NAME = 'kmeans_loss'

  @deprecated(None, _USE_TF_CONTRIB_FACTORIZATION)
  def __init__(self,
               num_clusters,
               model_dir=None,
               initial_clusters=RANDOM_INIT,
               distance_metric=SQUARED_EUCLIDEAN_DISTANCE,
               random_seed=0,
               use_mini_batch=True,
               mini_batch_steps_per_iteration=1,
               kmeans_plus_plus_num_retries=2,
               relative_tolerance=None,
               config=None):
    """Creates a model for running KMeans training and inference.

    Args:
      num_clusters: number of clusters to train.
      model_dir: the directory to save the model results and log files.
      initial_clusters: specifies how to initialize the clusters for training.
        See clustering_ops.kmeans for the possible values.
      distance_metric: the distance metric used for clustering.
        See clustering_ops.kmeans for the possible values.
      random_seed: Python integer. Seed for PRNG used to initialize centers.
      use_mini_batch: If true, use the mini-batch k-means algorithm. Else assume
        full batch.
      mini_batch_steps_per_iteration: number of steps after which the updated
        cluster centers are synced back to a master copy. See clustering_ops.py
        for more details.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample O(log(num_to_sample)) additional points.
      relative_tolerance: A relative tolerance of change in the loss between
        iterations.  Stops learning if the loss changes less than this amount.
        Note that this may not work correctly if use_mini_batch=True.
      config: See Estimator
    """
    params = {}
    params['num_clusters'] = num_clusters
    params['training_initial_clusters'] = initial_clusters
    params['distance_metric'] = distance_metric
    params['random_seed'] = random_seed
    params['use_mini_batch'] = use_mini_batch
    params['mini_batch_steps_per_iteration'] = mini_batch_steps_per_iteration
    params['kmeans_plus_plus_num_retries'] = kmeans_plus_plus_num_retries
    params['relative_tolerance'] = relative_tolerance
    super(KMeansClustering, self).__init__(
        model_fn=_kmeans_clustering_model_fn,
        params=params,
        model_dir=model_dir,
        config=config)

  @deprecated(None, _USE_TF_CONTRIB_FACTORIZATION)
  def predict_cluster_idx(self, input_fn=None):
    """Yields predicted cluster indices."""
    key = KMeansClustering.CLUSTER_IDX
    results = super(KMeansClustering, self).predict(
        input_fn=input_fn, outputs=[key])
    for result in results:
      yield result[key]

  @deprecated(None, _USE_TF_CONTRIB_FACTORIZATION)
  def score(self, input_fn=None, steps=None):
    """Predict total sum of distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative of the sum of distances.

    Args:
      input_fn: see predict.
      steps: see predict.

    Returns:
      Total sum of distances to nearest clusters.
    """
    return np.sum(
        self.evaluate(
            input_fn=input_fn, steps=steps)[KMeansClustering.SCORES])

  @deprecated(None, _USE_TF_CONTRIB_FACTORIZATION)
  def transform(self, input_fn=None, as_iterable=False):
    """Transforms each element to distances to cluster centers.

    Note that this function is different from the corresponding one in sklearn.
    For SQUARED_EUCLIDEAN distance metric, sklearn transform returns the
    EUCLIDEAN distance, while this function returns the SQUARED_EUCLIDEAN
    distance.

    Args:
      input_fn: see predict.
      as_iterable: see predict

    Returns:
      Array with same number of rows as x, and num_clusters columns, containing
      distances to the cluster centers.
    """
    key = KMeansClustering.ALL_SCORES
    results = super(KMeansClustering, self).predict(
        input_fn=input_fn,
        outputs=[key],
        as_iterable=as_iterable)
    if not as_iterable:
      return results[key]
    else:
      return results

  @deprecated(None, _USE_TF_CONTRIB_FACTORIZATION)
  def clusters(self):
    """Returns cluster centers."""
    return super(KMeansClustering, self).get_variable_value(self.CLUSTERS)

################################# Tensorflow Kmeans Cluster Function #################################




##### Seed Function #####
def seed_set(num):
    random.seed(num)
    np.random.seed(num)
    set_random_seed(num)
    
##### Model Function definition #####
def Vae_model():
    original_dim = 120*120
    latent_dim=2
    epsilon_std=1.0
    # input
    x = tf.placeholder(tf.float32, [None, original_dim])
    # encoder
    h = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h, units=64, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
    z_mean = tf.layers.dense(inputs=h3, units=latent_dim)
    z_log_var= tf.layers.dense(inputs=h3, units=latent_dim)
    epsilon = tf.random_normal([tf.shape(z_mean)[0],2], mean=0,stddev=1, dtype=tf.float32)
    z = tf.add(z_mean, tf.exp(z_log_var/2) *epsilon, name='z_add')
    # decoder 
    h_decoded = tf.layers.dense(inputs=z, units=32, activation=tf.nn.relu)
    h_decoded2 = tf.layers.dense(inputs=h_decoded, units=64, activation=tf.nn.relu)
    h_decoded3 = tf.layers.dense(inputs=h_decoded2, units=128, activation=tf.nn.relu)
    decoder_mean = tf.layers.dense(inputs=h_decoded3 , units=original_dim, activation=tf.nn.sigmoid)
    xent_loss = original_dim * tf.keras.metrics.binary_crossentropy(x,decoder_mean)
    kl_loss = -0.5* K.sum(1+z_log_var - tf.square(z_mean)- tf.exp(z_log_var),axis=-1)
    cost = K.mean(xent_loss+kl_loss)
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
    return x,cost, optimizer, z


##### Image Function Definition #####
def image_load(read=True, path=None):
    bw_path = os.path.join(path, 'bw*')
    # image Load and Sort
    image_list = natsort.natsorted(glob.glob(bw_path), reverse=False)
    # image information to tuple 
    image_tuple = list(enumerate(image_list))
    # image to shuffle for good training 
    random.shuffle(image_tuple)
    # image index, image_path 
    image_index, image_list = zip(*image_tuple) 
    image_array = []
    # convert image to array in read=True 
    if read:
      for img in image_list:
          image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
          image_array.append(np.array(image).flatten())  
      image_array = np.array(image_array, dtype="float") / 128. - 1
      return image_array,image_index
    # only need image_index in read=False
    else:
      return image_index


def image_batch(data, start,end):
    '''start : batch start point
       end : batch end point 
       start to final point where end=-1'''
    if end==-1:
        return data[start:]
    else:
        return data[start:end]
    

##### VAE Training #####    
def Vae_training(img_data):
    ''' Variational autoencoder training  '''
    # image count 
    image_counting = len(img_data)
    # rep_num : The quotient of the image divided by 100
    rep_num = image_counting//100
    # VAE Model Load 
    x,cost, optimizer, z = Vae_model()
    # start Session 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # total 10 epoch 
        for epoch in range(2):
            cost_mean = 0
            for i in range(rep_num):
                # final batch : start to final point 
                if i ==rep_num-1:
                    img_batch = image_batch(img_data,i*100,-1)
                    part_cost,_ = sess.run([cost,optimizer], feed_dict={x: img_batch})
                else:
                    img_batch = image_batch(img_data, i*100,(i+1)*100)
                    part_cost,_ = sess.run([cost,optimizer], feed_dict={x: img_batch})
                cost_mean +=part_cost
            #print(cost_mean/image_counting)
        # Image Extract Feature 
        compressed = sess.run(z,
                       feed_dict={x: img_data})
    return compressed
    
##### Kmeans Train And Score #####    
def kmeans_score(n=None,hash=None, se_mi=None):
    '''kmeans model definition and kmeans training '''
    # kmeans model Save directory ( hash directory )
    save_dir = os.path.join('./cluster_model',str(hash),  str(n)+'_'+se_mi)
    cluster_model = KMeansClustering(
                                        n, 
                                        distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE, 
                                        initial_clusters=tf.contrib.learn.KMeansClustering.KMEANS_PLUS_PLUS_INIT, 
                                        use_mini_batch=False, 
                                        relative_tolerance=10, 
                                        model_dir=save_dir)
    # kmeans start training                                     
    cluster_model.fit(input_fn=train_input_fn)
    # kmeans score evaluate 
    score = cluster_model.score(input_fn=train_input_fn,steps=1)
    return cluster_model, score   

##### Kmeans Delete #####
def kmeans_model_delete(n=None, hash=None,se_mi=None):
    ''' kmeans model delete because model is big size and useless '''
    save_dir = os.path.join('./cluster_model', str(hash), str(n)+'_'+se_mi)
    # kmeans delete 
    shutil.rmtree(save_dir,True)


##### Kmeans Elbow and Cosin Calculation#####
def ElbowMethod(max_number,hash,se_mi):
    ''' Kmeans Elbow and Cosin Calculation on max number '''
    wcss = np.zeros(max_number)
    
    for i in range(max_number):
        n = i+1
        _, score = kmeans_score(n,hash,se_mi)
        kmeans_model_delete(n,hash,se_mi)
        wcss[i] = score
    cosines = -1 * np.ones(max_number)
    
    for i in range(0,max_number-1):
    # check if the point is below a segment midpoint connecting its neighbors
        if (wcss[i] < (wcss[i+1]+wcss[i-1])/2 ):
            cosines[i]= (-1+(wcss[i-1]-wcss[i])*(wcss[i+1]-wcss[i]))/ \
            ((1+(wcss[i-1]-wcss[i])**2)*(1+ (wcss[i+1]-wcss[i])**2))**.5
    return wcss, (np.flip(np.argsort(cosines))+1)






##### -----------------Execute------------------------ #####
# Options Load
# parameter  = args.parameter
# parameter = ast.literal_eval(parameter)

img_path= args.image_path

cluster_number = args.cluster_number
max_number = args.max_number
auto = args.auto
auto = json.loads(auto.lower())
seed = args.seed
hash = args.hash

# Seed Set 
seed_set(seed)
npy_path = os.path.join ('./cluster_model',str(hash),'vae_feature.npy')

# To avoid duplication of kmeans models
now = datetime.now()
se_mi = str(now.minute)+str(now.second)+str(now.microsecond) 

if os.path.isdir(os.path.join ('./cluster_model',str(hash))):
  pass
else:
  os.makedirs(os.path.join ('./cluster_model',str(hash)))
 # npy file is not exists   ---> First Step Training  
if not os.path.isfile(npy_path):
    img_data,image_index= image_load(read=True, path=img_path)
    compressed = Vae_training(img_data)
    del img_data
    def train_input_fn():
        data = tf.constant(compressed, tf.float32)
        return (data, None)
    if auto:
        #'''"First training And Auto"'''
        if len(image_index) < max_number:
          max_number = len(image_index)
        wc, cos = ElbowMethod(max_number,hash, se_mi)
        select_number = int(cos[0])
        cluster_model, k_train = kmeans_score(select_number,hash,se_mi)
        class_idx = list(cluster_model.predict_cluster_idx(input_fn=train_input_fn))
        class_distance = np.amin(cluster_model.transform(input_fn=train_input_fn),axis=1)
        kmeans_model_delete(select_number,hash,se_mi)
        np.save(npy_path, [compressed,[select_number],[max_number]])        
    else:
        #'''First training And Not Auto,  Max_number=[-1]'''
        if len(image_index) < cluster_number:
          select_number = len(image_index)
        else:
          select_number = cluster_number
        cluster_model, k_train = kmeans_score(select_number,hash,se_mi)
        class_idx = list(cluster_model.predict_cluster_idx(input_fn=train_input_fn))
        class_distance = np.amin(cluster_model.transform(input_fn=train_input_fn),axis=1)    
        kmeans_model_delete(select_number,hash,se_mi)
        np.save(npy_path, [compressed,[select_number],[-1]])

# npy file is exists   ---> First Step Elbows or Kmeans  
else:
    image_index = image_load(read=False, path=img_path)
    compressed,exist_select_number,exist_max_number = np.load(npy_path)
    def train_input_fn():
        data = tf.constant(compressed, tf.float32)
        return (data, None)    
    # Auto True 
    if auto:
        #''' Same Max Number is that Start Kmeasn '''
        if max_number in exist_max_number:
            select_number = exist_select_number[exist_max_number.index(max_number)]
            cluster_model, k_train = kmeans_score(select_number,hash,se_mi)
            class_idx = list(cluster_model.predict_cluster_idx(input_fn=train_input_fn))
            class_distance = np.amin(cluster_model.transform(input_fn=train_input_fn),axis=1)
            kmeans_model_delete(select_number,hash,se_mi)
        else:
        #''' Not Same Max Number is that Start Elbows + Kmeans  '''
            if len(image_index) < max_number:
              max_number = len(image_index)
            wc, cos = ElbowMethod(max_number,hash,se_mi)
            select_number = int(cos[0])
            cluster_model, k_train = kmeans_score(select_number,hash,se_mi)
            class_idx = list(cluster_model.predict_cluster_idx(input_fn=train_input_fn))
            class_distance = np.amin(cluster_model.transform(input_fn=train_input_fn),axis=1)
            kmeans_model_delete(select_number,hash,se_mi)
            exist_max_number.append(max_number)
            exist_select_number.append(select_number)
            np.save(npy_path, [compressed,exist_select_number,exist_max_number])
    # Auto False 
    else:
        #''' Start is Kmeans not Training '''  
        if len(image_index) < cluster_number:
          select_number = len(image_index)
        else:
          select_number= cluster_number
        cluster_model, k_train = kmeans_score(select_number, hash,se_mi)
        class_idx = list(cluster_model.predict_cluster_idx(input_fn=train_input_fn))
        class_distance = np.amin(cluster_model.transform(input_fn=train_input_fn),axis=1)    
        kmeans_model_delete(select_number,hash,se_mi)

result_output = {"select_number":select_number, "chart_index":list(image_index),"class_index":class_idx,"distance":list(class_distance)}

def default(o):
    if isinstance(o, np.int64): return int(o)  
    if isinstance(o, np.float32): return float(o)  
    raise TypeError
result_json = json.dumps(result_output,default=default)
sys.stdout.write(result_json)     
