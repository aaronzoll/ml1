import numpy as np 
import numpy.linalg as la 
import matplotlib.pyplot as plt 
import scipy.special as sm
import time as time

class MLModel:
    def __init__(self,epochs = 250,lr = 0.05):
        ###In this constructor we set number of epochs for training and learning rate
        ## The only method you will need to modify in this class is fit
        self.epochs = epochs
        self.lr = lr 

    def gen_data(self,):
        raise NotImplementedError

    def loss(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError

    def grad(self,):
        raise NotImplementedError 

    def update(self):
        raise NotImplementedError  

    def metrics(self,x,y):
        raise NotImplementedError     

    def fit(self,x_data,y_data,x_eval,y_eval, printing = False):
        ### This method implements our "1. forward 2. backward 3. update" paradigm
        ## it should call forward(), grad(), and update(), in that order. 
        # you should also call metrics so that you may see progress during training
        if True:
            self.x_eval = x_eval 
            self.y_eval = y_eval
        
        for epoch in range(self.epochs):
            ## TODO (implement train step ) 
            y_pred = self.forward(x_data)
            grad, loss = self.grad(x_data, y_pred, y_data)
            self.update(grad)
            m=self.metrics(x_eval,y_eval)
            if printing: 
                if epoch % 100 == 0:
                    m = self.metrics(x_eval,y_eval)
                    print(f"epoch {epoch} and train loss {loss.mean():.2f}, test metrics {m:.2f}, grad {np.round(grad,3)}, grad norm {la.norm(grad):.2f}")

    
    def e2e(self,n_train = 100, n_test = 10000,printing = False,data = None):
        #end to end method generates data + trains model 
        if data is None:
            x_train, y_train = self.gen_data(n_train)
            x_test, y_test = self.gen_data(n_test)
            data = (x_train,y_train,x_test,y_test)
        else:
            x_train,y_train,x_test,y_test = data
        self.fit(x_train,y_train,x_test,y_test,printing)
        m = self.metrics(x_test,y_test)
        return m , data


class LinReg(MLModel):
    def __init__(self,fun = np.square, deg =1,n_data = 10000,lr = .05,x_sig = 1,epoch = None):
        if epoch is not None:
            super().__init__(epochs=epoch,lr=lr)
        else:
            super().__init__(lr=lr)
        self.x_sig = x_sig
        self.lr = lr
        self.n_data = n_data
        self.fun = fun
        self.degree = deg
        self.sf = np.ones(deg+1) #??? 
        self.set_hilberts()
        self.model_coeff = .1*(np.random.random(self.hp_coeff.shape[0])-.5)

    def set_hilberts(self):
        x,y = self.gen_data()
        hp_coeff = self.solve_hp(x,y,self.degree)
        self.hp_coeff = hp_coeff 

    def gen_data(self, n = None, noise_var = .1):
        if n is None:
            n = self.n_data 
        x_dat = np.random.normal(0,self.x_sig,n)
        x_dat.sort()
        y_dat = self.fun(x_dat) + np.random.normal(0,noise_var,n)
        return x_dat, y_dat 

    def forward(self,x):
        coeff = self.model_coeff/sm.factorial(np.arange(self.degree+1))
        y_pre = self.poly(x,coeff)
        return y_pre

    def loss(self,y_approx,y_true):
        L = y_approx-y_true
        L = L**2
        loss = np.mean(L)
        return loss

    def metrics(self,x,y):
        # TODO: Implement some metric for evaluating reg performance
        #return np.sum(np.abs(y-self.poly(x,self.model_coeff)))
        coeff = self.model_coeff/sm.factorial(np.arange(self.degree+1))
        return self.loss(self.poly(x,coeff),y)

    def grad(self,x, y_pre, y_data):
        m = len(x)
        test=1
        if test == 1:
            grad = np.zeros(self.degree+1)
            vec_x=np.zeros((self.degree+1,m))
            for i in range(self.degree+1):
                vec_x[i] = x**i
            c = 2/m*(y_pre-y_data)
            grad = np.matmul(vec_x,c)
            grad /= sm.factorial(np.arange(grad.shape[0]))
        if test == 0:
            grad = np.zeros(self.degree+1)
            for i in range(m):
                vec_x=np.zeros(self.degree+1)
                for j in range(self.degree+1):
                    vec_x[j]=x[i]**j
                grad += 2/m*(y_pre[i]-y_data[i])*vec_x
            grad /= sm.factorial(np.arange(grad.shape[0]))
        loss = self.loss(y_data,y_pre)
        return grad, loss

    def update(self,grad):
        k=self.degree+1
       # coeff=[]
       # for i in range(k):
       #     coeff.append(self.model_coeff[i]*sm.factorial(i))
        self.model_coeff = self.model_coeff - (self.lr * grad)
       # for i in range(k):
       #     self.model_coeff[i]=self.model_coeff[i]/sm.factorial(i)
        


    def poly(self,t,coeff=None,):
            deg = coeff.shape[0]
            if not isinstance(t,np.ndarray):
                t = np.array([t])
            nts = t.shape
            if len(nts) == 1:
                t = t.reshape([nts[0],1])
            tp = t ** np.arange(deg) 
            result = tp @ coeff 
            return result 


    def solve_hp(self,x_data,y_data,deg,a=None,b=None):
        if x_data is not None:
            a,b = self.setup_hp(x_data,y_data,deg)
        coeff = self.matrix_inv(a,b)
        return coeff 

    def setup_hp(self,x_data,y_data,n):
        covx_rows = np.zeros(2*n+1)
        covxy = np.zeros(n+1)
        covx = np.zeros([n+1,n+1])
        for j in range(2*n+1):
            tj = x_data**j
            if j < n+1:
                covxy[j] = ((tj) * y_data).mean()
            covx_rows[j] = tj.mean() 
        for j in range(n+1):
            covx[j,:] = covx_rows[j:j+n+1]
        return covx,covxy  

    def matrix_inv(self,a,b):
        ainv = la.inv(a)
        x = ainv @ b 
        return x 

class LogReg(MLModel):
    def __init__(self,p1 = .5,loss = 'nll',x_sig = 1,epochs=1000,m1=-2,m2=2):
        super().__init__(epochs = epochs)
        self.model_coeff =  np.random.random(2)
        self.p1 = p1 
        self.p0 = 1-p1 
        self.m1=m1
        self.m2=m2
        self.x_sig = x_sig
        if loss == 'nll':
            self.grad = self.log_grad 
            self.loss = self.nll_loss 
        else:
            self.grad = self.sq_grad
            self.loss = self.mse_loss 

    def gen_data(self,n = 10000):
        n1 = int(self.p1 * n)
        n0 = int(self.p0 * n)
        x0 = np.random.normal(self.m1,self.x_sig,n0)
        x1 = np.random.normal(self.m2,self.x_sig,n1)
        x0.sort()
        x1.sort()
        y0 = np.zeros(n0)
        y1 = np.ones(n1)
        x = np.concatenate([x0,x1])
        y = np.concatenate([y0,y1]).astype(int)
        return x, y

    def forward(self,x,alpha = None,):
        a=self.model_coeff[1]
        b=self.model_coeff[0]
        ytild = 1/(1+np.exp(-(a*x+b)))
        return ytild 


    def mse_loss(self,y_approx,y_true):
        L=(y_approx-y_true)**2
        return np.mean(L)

    def nll_loss(self,y_approx,y_true):
        L=-y_true*np.log(y_approx+0.0000)-(1-y_true)*np.log(1-y_approx+0.0000)
        return np.mean(L)

    def metrics(self,x,y):
        y1=np.ones(y.shape[0])
        y1[self.forward(x)<0.5]=0
        acc = 1-np.mean(np.abs(y1-y))
        return acc


    def sq_grad(self,x,y_pre, y_data,alpha = None,):
        m = len(x)
        vec_x0=np.ones(m)
        vec_x1=x
        vec_x=np.vstack((vec_x0,vec_x1))
        c = 2/m*(y_pre-y_data)*y_pre*(1-y_pre)
        grad = np.matmul(vec_x,c)
        loss = self.mse_loss(y_pre,y_data)
        return grad, loss

    def log_grad(self,x,y_pre,y_data, alpha = None,eps = 0.001):
        m = len(x)
        vec_x0=np.ones(m)
        vec_x1=x
        vec_x=np.vstack((vec_x0,vec_x1))
        c = 1/m*((1-y_data)/(1-y_pre)-y_data/y_pre)*y_pre*(1-y_pre)
        grad = np.matmul(vec_x,c)
        loss = self.nll_loss(y_pre,y_data)
        return grad, loss 

    def update(self,grad):
        # TODO: update model params
        self.model_coeff = self.model_coeff - (self.lr * grad)


    


def testinglinreg(regr,n_tr,n_eval):
    m = regr.e2e(n_train = n_tr,n_test=n_eval,printing = True)
    #print(m)
    plt.plot(regr.x_eval,regr.y_eval,'-.',label = 'data')
    t = np.linspace(-10,10,1000)
    L=[]
    k=0
    coeff = reg.model_coeff/sm.factorial(np.arange(reg.degree+1))
    for data in coeff:
       L.append('{:8.9f}'.format(data)+'x^'+str(k))
       k+=1 
    print(" + ".join(L))
    plt.plot(t,regr.poly(t,regr.hp_coeff),label = 'hp')
    plt.plot(t,regr.forward(t),label = 'linreg')
    plt.plot(t,reg.fun(t),'-.',label = 'truth')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-6,6])
    plt.ylim([-2,2])
    plt.show()

def testinglogreg(logregr1,logregr2,n_train,n_test,thresh=0.5):
    m1,d = logregr1.e2e(n_train,n_test,printing = True,)
    m2,_ = logregr2.e2e(printing = True,data = d)
    x_dat = np.linspace(logregr1.x_eval.min(),logregr1.x_eval.max(),500)
    ytild1 = logregr1.forward(x_dat)
    ytild2 = logregr2.forward(x_dat)
    x_eval = d[2]
    y_eval = d[3]
    x0 = x_eval[y_eval == 0]
    x1 = x_eval[y_eval == 1]
    #m1 = logregr1.m 
    #m2= logregr2.m
    thresh1 = x_dat[np.argmin(np.abs(ytild1-thresh))]
    thresh2 = x_dat[np.argmin(np.abs(ytild2-thresh))]
    y1 = np.ones(x_dat.shape[0])
    y2 = np.ones(x_dat.shape[0])
    y1[x_dat < thresh1] = 0 
    y2[x_dat < thresh2] = 0 

    plt.hist(x0,density = True,bins = 50,alpha = .5,label='y=0')
    plt.hist(x1,density = True,bins = 50,alpha = .5,label ='y=1')

    test=1
    if test == 1:
        plt.plot(x_dat,ytild1,'-.',label = f"y_score w nll loss (accuracy {m1:.7f})")
        plt.plot(x_dat,ytild2,'-.',label = f"y_score w mse loss (accuracy {m2:.7f})")
        plt.plot(x_dat,y1,'-.',label = 'y_pred w nll loss')
        plt.plot(x_dat,y2,'-.',label = 'y_pred w mse loss')
    plt.legend()
    plt.show()
    return logregr1,logregr2


if __name__ == "__main__": 
    d=19
    reg = LinReg(n_data = 1500,deg = d,fun = np.sin,x_sig = 2.5,epoch=10000,lr=0.0003)
    testinglinreg(reg,500,1500)
    #logreg1 = LogReg(loss = 'nll',p1 =.05,epochs = 5000,x_sig=2,m1=-2)
    #logreg2 = LogReg(loss = 'not nll',p1=.05,epochs = 5000,x_sig=2,m1=-2)
    #r1,r2 =  testinglogreg(logreg1,logreg2,10000,250000,thresh=0.5)




    
