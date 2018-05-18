import tensorflow as tf

def m_activation(x):
  #return tf.sign(x)*tf.minimum(tf.abs(x),4)
  return tf.tanh(x)*4

def conv1d(x,f,k,s,d,act):
    out = tf.layers.conv1d(x,filters=f,kernel_size=k,strides=s,
       padding='same',activation=act,dilation_rate = d,
       kernel_initializer=tf.contrib.layers.xavier_initializer())
    return out#tf.layers.batch_normalization(out)

def conv2d_transpose(x,f,k,s,act):
    out = tf.layers.conv2d_transpose(x,filters=f,
    kernel_size=[k,1],strides=[s,1],activation=act,padding='SAME',
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return out#tf.layers.batch_normalization(out)

def dilation_block(encoding,current_layer):
    out_dila = tf.constant(2,dtype=tf.int64)
    for n in range(40):
        if encoding:
            current_layer1 = conv1d(current_layer,32,2,1,2**(n%10),tf.tanh)
            current_layer2 = conv1d(current_layer,32,2,1,2**(n%5),tf.tanh)
            #current_layer3 = conv1d(current_layer,32,2,1,2**(n%6),tf.tanh)
            #current_layer4 = conv1d(current_layer,32,2,1,2**(n%4),tf.tanh)
        else:
            current_layer1 = conv1d(current_layer,32,2,1,2**((49-n)%10),tf.tanh)
            current_layer2 = conv1d(current_layer,32,2,1,2**((49-n)%5),tf.tanh)
            #current_layer3 = conv1d(current_layer,32,2,1,2**((49-n)%6),tf.tanh)
            #current_layer4 = conv1d(current_layer,32,2,1,2**((49-n)%4),tf.tanh)
        out=current_layer1*current_layer2#*current_layer3*current_layer4
        #out = tf.divide(out,4)
        transformed = out

        current_layer = current_layer+transformed #residual

        if (out_dila.dtype==tf.int64):
             out_dila = out
        else:
             out_dila = out_dila + out #skipped

    current_layer = tf.divide(out_dila,40)
    return current_layer

def generate(inputx,sample_length,init_conv=None,init_deconv=None,n_hidden=2,tr=True):
    inputx = tf.reshape(inputx,(-1,sample_length,1))
    activation = m_activation
    current_layer = inputx
    current_layer = conv1d(current_layer,32,2,1,1,tf.tanh)

    current_layer = dilation_block(True,current_layer)
    current_layer = conv1d(current_layer,1,1,1,1,None)

    for f in [128,256,256,512]:
          current_layer = conv1d(current_layer,f,9,2,1,tf.tanh)
    current_layer = conv1d(current_layer,512,5,2,1,tf.tanh)

    ###########################
    current_layer = conv1d(current_layer,1,1,1,1,None)
    current_layer = tf.layers.flatten(current_layer)

    mn = tf.layers.dense(current_layer, n_hidden, activation = None)

    sd = 0.5 * tf.layers.dense(current_layer, n_hidden, activation = None)
    epsilon = tf.random_normal([tf.shape(current_layer)[0], n_hidden], 0.0, 1.0)

    z  = mn + tf.multiply(epsilon, tf.exp(sd))

    current_layer = z
    current_layer = tf.layers.dense(current_layer, sample_length/32/2)
    current_layer = tf.layers.dense(current_layer, sample_length/32)
    current_layer=tf.reshape(current_layer,(-1,sample_length/32,1,1))
    ##################


    current_layer = conv2d_transpose(current_layer,512,5,2,tf.tanh)
    for f in [512,256,256,128]:
        current_layer = conv2d_transpose(current_layer,f,9,2,tf.tanh)
    current_layer = conv2d_transpose(current_layer,1,1,1,tf.tanh)
    current_layer=tf.reshape(current_layer,(-1,sample_length,1))
    current_layer = conv1d(current_layer,32,1,1,1,None)

    current_layer = dilation_block(False,current_layer)

    #output_layer = conv1d(current_layer,32,2,1,1,tf.tanh)
    #print(current_layer)
    current_layer = tf.reshape(current_layer,(-1,sample_length,1,32))
    current_layer = conv2d_transpose(current_layer,32,2,1,None)
    current_layer = conv2d_transpose(current_layer,1,1,1,None)
    current_layer = tf.reshape(current_layer,(-1,sample_length,1))

    current_layer= conv1d(current_layer,1,1,1,1,None)
    current_layer = tf.contrib.layers.flatten(current_layer)

    current_layer = tf.layers.dense(current_layer, sample_length)

    output_layer=tf.reshape(current_layer,(-1,sample_length,1))
    return output_layer, z ,inputx, mn , sd
