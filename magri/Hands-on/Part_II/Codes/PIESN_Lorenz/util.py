"""
@author: Nguyen Anh Khoa Doan
"""
# Util module

import tensorflow as tf

# function to initialize uninitialized variables
# useful when progressively building new parts of the graph and introducing new optimizer
def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
"""
optimizer = tf.contrib.opt.ScipyOptimizerInterface(LOSS_TF,
        method = 'L-BFGS-B',var_list=[Wout_tf],options = {'maxfun': 100000, 'maxiter':50000,
                                        'maxcor':50,'maxls':50,
                                        'eps': 1.0 * np.finfo(float).eps,
                                        'ftol' : 1.0 * np.finfo(float).eps})
optimizer.minimize(sess,feed_dict ={Ytarget_tf:rnn_target[begin:end,:],stateF:st,outputs:stored_outputs})
"""
