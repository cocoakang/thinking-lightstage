import tensorflow as tf

def fully_connect(input_vector,input_len,output_len,scope_name,reuse = tf.AUTO_REUSE,with_bias = True,activate_fun=tf.nn.leaky_relu):
    '''
    input_vector [batch_size,input_len]
    return [batch_size,output_len]
    input_len scalar
    output_len scalar
    '''
    with tf.variable_scope(scope_name,reuse=reuse) as vs:
        W = tf.get_variable('W', [input_len, output_len])
        b = tf.get_variable('b', [output_len], initializer=tf.constant_initializer(0.0))
        if activate_fun is not None:
            x_n = activate_fun(tf.matmul(input_vector, W) + b)
        else:
            x_n = tf.matmul(input_vector, W) + b
    return x_n