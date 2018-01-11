#test script for initialization
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters(hparams,conf)
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W"].eval().shape))
    print("b = " + str(parameters["b"].eval().shape))

#placeholders
X, Y = create_placeholder(hparams,conf)
print ("X = " + str(X))
print ("Y = " + str(Y))

#forward prop
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholder(hparams,conf)
    parameters = initialize_parameters(hparams,conf)
    output_sequence_logits = forward_propagation(X, parameters,hparams,conf)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(output_sequence_logits, {X: np.random.randn(2,25,51), Y: np.random.randn(2,25,51)})
    print("output_sequence_logits = " + str(len(a)))

#cost
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholder(hparams,conf)
    parameters = initialize_parameters(hparams,conf)
    output_sequence_logits = forward_propagation(X, parameters,hparams,conf)
    cost = compute_cost(output_sequence_logits,Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(2,25,51), Y: np.random.randn(2,25,51)})
    print("output_sequence_logits = " + str(a))

#char input generator
def input_generator(filename,seq_len):
    x=[]
    y=[]
    with open(filename, 'r') as f:
        char_list = f.read()
        clist=[]
        for i in range(0,len(char_list),2):
            clist.append(char_list[i])
        p=0
        while p+seq_len+1<=len(clist):
            input_ = clist[p:p+seq_len]
            output_ = clist[p+1:p+1+seq_len]
            p = p+seq_len
            x.append(v_.getOneHot(input_))
            y.append(v_.getOneHot(output_))
    x=np.array(x)
    y=np.array(y)
    return x,y

#tes
def get_mini_batches(X,Y,batch_size):
    num_of_input = X.shape[0]
    num_of_batches = int(num_of_input/batch_size)
    mini_batches = []
    for i in range(num_of_batches):
        start_ind = i*batch_size
        batch_x = X[start_ind:start_ind+batch_size]
        batch_y = Y[start_ind:start_ind+batch_size]
        mini_batches.append((batch_x,batch_y))
    if num_of_input%batch_size != 0:
        start_ind = num_of_batches*batch_size
        batch_x = X[start_ind:]
        batch_y = Y[start_ind:]
        mini_batches.append((batch_x,batch_y))
    return mini_batches    

#shape change 
#a=np.array(list(range(1,31))).reshape(3,2,5)
a=[np.array(list(range(1,7))).reshape(2,3),np.array(list(range(13,19))).reshape(2,3),np.array(list(range(7,13))).reshape(2,3)]

A = tf.concat(values=a,axis=1)
A = tf.reshape(A, shape=(-1,3,3))
with tf.Session() as sess:
    a_=sess.run(A)
    print(a_)