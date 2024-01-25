import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(nsamples, range_uniform=1, sigma=0.):
    
    """
    Generate a dataset with the specified number of samples
    
    :param nsamples: number of sample points
    :type nsamples: int
    :param sigma: standard deviation of the noise
    :type sigma: float
    
    :return: Generated dataset {(x_n, y_n), n = 1, ..., nsamples}
    :rtype: tuple of numpy arrays
    """

    x = np.zeros((nsamples, 2))
    x[:, 0] = np.random.uniform(-range_uniform, range_uniform, nsamples)
    x[:, 1] = np.random.uniform(-range_uniform, range_uniform, nsamples)

    eps = np.random.normal(loc=0, scale=sigma, size=nsamples)
    
    y = 4*x[:, 0] + 3*x[:, 1] + eps
    return x, y


def unison_shuffled_copies(a, b):
    '''
    Returns a similar permutation between arrays a,b (used for shuffle)
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_train_test_datasets(A,Y,TRAINING_RATIO):
    '''
    From the whole dataset, separate it in two and return train and test separately
    '''
    N = len(Y) #numbers of rows
    N_TRAIN = int(TRAINING_RATIO*N) # number of rows in the training set

    A_train = A[:N_TRAIN,:]
    Y_train = Y[:N_TRAIN]

    A_test = A[N_TRAIN:,:]
    Y_test = Y[N_TRAIN:]

    #we will normalize our data as well to better performance (less anisotropy, so the gradient descent should be faster)
    mean = A_train.mean(axis=0)
    std = A_train.std(axis=0)
    A_train = (A_train-mean)/std

    return A_train, Y_train, A_test, Y_test


def gradient_descent(A_train, Y_train, nb_of_iterations, step_size, ridge_parameter, add_bias):
    '''
    Perform GD on the whole dataset and return the plot of the training loss with respect to the number of iterations
    '''

    N_TRAIN = A_train.shape[0]
    D = A_train.shape[1]

    if add_bias == True :
         #We initialize x (parameters of the model)
        x = np.random.rand(D+1) #because x takes in account a bias

        one_vector_train = np.ones(N_TRAIN)
        A_train_b = np.column_stack((one_vector_train, A_train))

        loss_function_array= []
        iteration = []

        for i in range(nb_of_iterations): 
            iteration.append(i)

            loss_function_train = np.linalg.norm(np.dot(A_train_b,x)-Y_train)**2 + ridge_parameter*x.transpose().dot(x)
            loss_function_array.append(loss_function_train)

            grad = 2*A_train_b.T.dot(np.dot(A_train_b,x)-Y_train) + 2*ridge_parameter*x
            x = x - step_size*grad

    else:
         #We initialize x (parameters of the model)
        x = np.random.rand(D) #because x takes in account no bias

        loss_function_array= []
        iteration = []

        for i in range(nb_of_iterations): 
            iteration.append(i)

            loss_function_train = np.linalg.norm(np.dot(A_train,x)-Y_train)**2 + ridge_parameter*x.transpose().dot(x)
            loss_function_array.append(loss_function_train)

            grad = 2*A_train.T.dot(np.dot(A_train,x)-Y_train)+ 2*ridge_parameter*x
            x = x - step_size*grad
        
    
    return iteration, loss_function_array

def plot_convergence_rate(A_train,Y_train, niter, ridge_parameter):
    '''
    Code from Gabriel Peyré
    '''
    flist = np.zeros((niter,1))

    D = A_train.shape[1]

    #various step_size to test 
    comparison = [0.01,0.1,1,3,3.1,3.2]
    step_size_mult = [ x*1e-5 for x in comparison ]

    xopt = np.linalg.solve(A_train.transpose().dot(A_train) + ridge_parameter*np.eye(D), A_train.transpose().dot(Y_train) )
    plt.clf

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout()

    for istep in np.arange(0,len(step_size_mult)):
        step_size = step_size_mult[istep]
        x = np.random.rand(D) #because x takes in account no bias
        
        for i in np.arange(0,niter):
            flist[i] = np.linalg.norm(np.dot(A_train,x)-Y_train)**2 + ridge_parameter*x.transpose().dot(x)
            grad = 2*A_train.T.dot(np.dot(A_train,x)-Y_train)+2*ridge_parameter*x
            x = x - step_size*grad

        ax1.plot(flist)
        ax1.axis('tight')
        plt.title('f(x_k)')

        e = np.log10(flist - np.linalg.norm(np.dot(A_train,xopt)-Y_train)**2 - ridge_parameter*xopt.transpose().dot(xopt) +1e-20)
        ax2.plot(e-e[0], label=str(round(step_size_mult[istep],8)))
        ax2.axis('tight')
        leg = ax2.legend()
        plt.title('$log(f(x_k)-min GD)$')

def plot_evolution_on_test_error(A_train, Y_train, A_test, Y_test, start, stop, num):
    '''
    Code from Gabriel Peyré
    '''
    p = A_train.shape[1]
    lmax = np.linalg.norm(A_train,2)**2
    lambda_list = lmax*np.linspace(start,stop,num)
    X = np.zeros( (p,num) )
    E = np.zeros( (num,1) )

    for i in np.arange(0,num):
        Lambda = lambda_list[i]
        x = np.linalg.solve( A_train.transpose().dot(A_train) + Lambda*np.eye(p), A_train.transpose().dot(Y_train) )
        X[:,i] = x.flatten() # bookkeeping
        E[i] = np.linalg.norm(A_test.dot(x)-Y_test) / np.linalg.norm(Y_test)

    # find optimal lambda
    i = E.argmin()
    lambda0 = lambda_list[i]
    xRidge = X[:,i]
    print( 'Ridge: ' + str(E.min()) )
    print( 'Optimal X is :')
    print(xRidge)
    # Display error evolution.
    plt.clf
    plt.plot(lambda_list/lmax, E)
    plt.plot( [lambda0/lmax,lambda0/lmax], [E.min(), E.max()], 'r--')
    plt.axis('tight')
    plt.xlabel('$\lambda/|Atrain|^2$')
    plt.ylabel('$E$')

def plot_evolution_on_test_error_with_bias(A_train, Y_train, A_test, Y_test, start, stop, num):
    '''
    Code from Gabriel Peyré
    '''
    p = A_train.shape[1]
    one_vector_train = np.ones(A_train.shape[0])
    one_vector_test = np.ones(A_test.shape[0])

    A_train = np.column_stack((one_vector_train, A_train))
    A_test = np.column_stack((one_vector_test, A_test))
    lmax = np.linalg.norm(A_train,2)**2
    lambda_list = lmax*np.linspace(start,stop,num)
    X = np.zeros( (p+1,num) )
    E = np.zeros( (num,1) )

    for i in np.arange(0,num):
        Lambda = lambda_list[i]
        x = np.linalg.solve( A_train.transpose().dot(A_train) + Lambda*np.eye(p+1), A_train.transpose().dot(Y_train) )
        X[:,i] = x.flatten() # bookkeeping
        E[i] = np.linalg.norm(A_test.dot(x)-Y_test) / np.linalg.norm(Y_test)

    # find optimal lambda
    i = E.argmin()
    lambda0 = lambda_list[i]
    xRidge = X[:,i]
    print( 'Ridge: ' + str(E.min()) )
    print( 'Optimal X is :')
    print(xRidge)
    # Display error evolution.
    plt.clf
    plt.plot(lambda_list/lmax, E)
    plt.plot( [lambda0/lmax,lambda0/lmax], [E.min(), E.max()], 'r--')
    plt.axis('tight')
    plt.xlabel('$\lambda/|Atrain|^2$')
    plt.ylabel('$E$')

def gradient_descent_with_momentum(A_train,Y_train,nb_of_iterations,step_size,gamma):
    # Gradient descent with momentum for mu-strongly convex and L-smooth functions
    
    D = A_train.shape[1]

    x = np.random.rand(D) #because x takes in account no bias
    m = np.zeros(D) #init of the momentum parameter

    loss_function_array= []
    iteration = []

    for i in range(nb_of_iterations): 
        iteration.append(i)

        loss_function_train = np.linalg.norm(np.dot(A_train,x)-Y_train)**2
        loss_function_array.append(loss_function_train)

        grad = 2*A_train.T.dot(np.dot(A_train,x)-Y_train)
        m = (1-gamma) * grad + gamma * m 
        x = x - step_size*grad
    
    return iteration, loss_function_array

def plot_convergence_rate_with_momentum(A_train,Y_train, niter, step_size):
    '''
    Code inspired Gabriel Peyré
    '''
    flist = np.zeros((niter,1))

    D = A_train.shape[1]

    #various step_size to test 
    comparison = [0., 0.7, 0.8, 0.9,]
    gamma_mult = [ x for x in comparison ]

    xopt = np.linalg.solve(A_train.transpose().dot(A_train), A_train.transpose().dot(Y_train) )
    plt.clf

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout()

    for igamma in np.arange(0,len(gamma_mult)):
        gamma = gamma_mult[igamma]
        x = np.random.rand(D) #because x takes in account no bias
        m = np.zeros(D)

        for i in np.arange(0,niter):
            flist[i] = np.linalg.norm(np.dot(A_train,x)-Y_train)**2 
            grad = 2*A_train.T.dot(np.dot(A_train,x)-Y_train) 
            m = (1-gamma) * grad + gamma * m
            x = x - step_size*m

        ax1.plot(flist)
        ax1.axis('tight')
        plt.title('f(x_k)')

        e = np.log10(flist - np.linalg.norm(np.dot(A_train,xopt)-Y_train)**2 + 1e-20)
        ax2.plot(e-e[0], label=str(gamma_mult[igamma]))
        ax2.axis('tight')
        leg = ax2.legend()
        plt.title('$log(f(x_k)-minGD)$')

def plot_iterates_l2_2D(iterates, proj_iterates, R, with_set):
    '''
    Code inspired from Irène Waldspurger
    '''
    # Display a list of points in R^2
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(iterates[0,:], iterates[1,:], 'bx-', label='Iterates GD')
    ax.plot(proj_iterates[0,:], proj_iterates[1,:], 'gx-',label='Iterates GD Projected')
    ax.legend()

    if with_set:
        circle = plt.Circle((0,0),R, color='k', fill=False)
        ax.add_patch(circle)

    # Axes limits
    xmin = min(min(iterates[0,:]),min(proj_iterates[0,:]))
    xmax = max(max(iterates[0,:]),max(proj_iterates[0,:]))
    (xmin,xmax) = (1.2*xmin-0.2*xmax,1.2*xmax-0.2*xmin)
    ymin = min(min(iterates[1,:]),min(proj_iterates[1,:]))
    ymax = max(max(iterates[1,:]),max(proj_iterates[1,:]))
    (ymin,ymax) = (1.2*ymin-0.2*ymax,1.2*ymax-0.2*ymin)
    ax.set_xlim(xmin=xmin,xmax=xmax)
    ax.set_ylim(ymin=ymin,ymax=ymax)
    
    # Grid
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

def plot_iterates_l1_2D(iterates, proj_iterates, R, with_set):
    '''
    Code inspired from Irène Waldspurger
    '''
    # Display a list of points in R^2
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(iterates[0,:], iterates[1,:], 'bx-', label='Iterates GD')
    ax.plot(proj_iterates[0,:], proj_iterates[1,:], 'gx-',label='Iterates GD Projected')
    ax.legend()

    if with_set:
        square = plt.Rectangle((0,-R),R*np.sqrt(2),R*np.sqrt(2), angle = 45, color='k', fill=False)
        ax.add_patch(square)

    # Axes limits
    xmin = min(min(iterates[0,:]),min(proj_iterates[0,:]))
    xmax = max(max(iterates[0,:]),max(proj_iterates[0,:]))
    (xmin,xmax) = (1.2*xmin-0.2*xmax,1.2*xmax-0.2*xmin)
    ymin = min(min(iterates[1,:]),min(proj_iterates[1,:]))
    ymax = max(max(iterates[1,:]),max(proj_iterates[1,:]))
    (ymin,ymax) = (1.2*ymin-0.2*ymax,1.2*ymax-0.2*ymin)
    ax.set_xlim(xmin=xmin,xmax=xmax)
    ax.set_ylim(ymin=ymin,ymax=ymax)
    
    # Grid
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')