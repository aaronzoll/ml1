import time 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm as gauss

class GaussGlivenkoHoeffding:
    def __init__(self,A = [0,1]):
        #np.random.seed(0)
        self.A = A #defines event $A\subset \R$
        self.task_dict = {"event":self.event_probability,"cdf":self.glivenko_cdf}

    def hoeffding_bound(self,m,epsilon):
        #step 0: return the right hand side of (1)
        p_bound = 2*np.exp(-2*m*epsilon**2)
        return p_bound

    def gen_data(self,n,sort):
        #step 1: generate mean 0, stdev 1, guassian data
        data = np.random.normal(0,1,n)
        if sort:
            data.sort()
        return data

    def event_probability(self,data,A = None):
        #step 2: return empirical probability that data \in A 
        #as well as the true probability P(A)
        if A is None:
            A = self.A
        p_approx = (abs(data-0.5) < 0.5).sum()/len(data)
        p_true = 0.3413447461
        return p_approx, p_true 

    def glivenko_cdf(self,data):
        #step 9: return *arrays* of approximate (empirical) cdf
        #+ true cdf. This should almost mimic measured event in (3)
        #I recommend that you evaluate the sup_t in run_hoeffding_pre
        #(if you do it right, this modification should not break it working 
        # task == 'event')
        n_data = data.shape[0]
        ecdf = gauss.cdf(data)
        tcdf = self.ecdf(n_data)
        return ecdf, tcdf

    def run_task(self,n,task,sort):
        #step 3: generate data with n samples + run task 
        p_approx, p_true = self.task_dict[task](self.gen_data(n,sort))
        return p_approx, p_true
    
    def run_hoeffding_pre(self,m_data,epsilon,task):
        #step 4: evaluates (2)        
       
        if task == "cdf":
            p_approx, p_true = self.run_task(m_data,task,True)
            t=np.argmax(abs(p_approx-p_true))
            p_approx=p_approx[t]
            p_true=p_true[t]
        else:
            p_approx, p_true = self.run_task(m_data,task,False)
        indic_fail=0
        if abs(p_approx-p_true) > epsilon:
            indic_fail = 1
        return indic_fail

    def run_hoeffding(self,m_data,n_runs,epsilon,task,printing = False):
        #step 5: samples (2) numerous times
        #return empirical fm, hoeffding bound (rhs of 1), hb - fm
        p_bound = self.hoeffding_bound(m_data,epsilon) #(the Hoeffding bound for m_data,epsilon)
        exp_array = np.zeros(n_runs)
        # for j in range(n_runs):
        #     indic_exp = self.run_hoeffding_pre(m_data,epsilon,task) 
        #     exp_array[j] = indic_exp
        exp_array = [self.run_hoeffding_pre(m_data,epsilon,task) for j in range(n_runs)]
        p_fail = np.mean(exp_array) 
        p_diff = p_bound - p_fail 
        return p_fail, p_bound, p_diff
    
    def find_nruns(self,epsilon,delta):
        #step 6a
        #solves for m in Hoeffding
        n_runs = np.ceil((np.log(delta/2))/(-2*epsilon**2))
        return int(n_runs)
        
    def get_prec(self,delta,m_data):
        #solves for epsilon in Hoeffding
        eps = np.sqrt(np.log(delta/2)/(-2*m_data))
        return eps
    
        
    def exp_hoeffding(self,m_data,epsilon,task,delta):
        #step 6b
        window_candidate = self.hoeffding_bound(m_data,epsilon)/2
        n_runs = self.find_nruns(np.exp(-2*m_data*epsilon**2),delta)
        print(f"number of runs for eps {epsilon:.4f}, m_data {m_data} is {n_runs}")
        p_fail, p_bound, p_diff= self.run_hoeffding(m_data,n_runs,epsilon,task)
        e_prime = abs(2*np.exp(-2*m_data*epsilon**2)-p_fail)
        print(e_prime)
        d_emp = self.hoeffding_bound(n_runs,e_prime)
        if d_emp > delta:
            pass
            n_runs_new = self.find_nruns(e_prime,delta)
            #p_fail, p_bound, p_diff= self.run_hoeffding(m_data,n_runs,epsilon,task)
            print(n_runs_new)
            pass #may need to run more if you don't satisfy confidence bound
        conf_window = self.get_prec(delta,n_runs)
        return p_fail, p_diff, p_bound,m_data, conf_window
    
    def sweep_hoeffding(self,epsilon,exp_range,task,delta = 0.025):
        #step 7 
        self.task = task
        t1 = time.time()
        results_data = np.zeros([exp_range.shape[0],5])
        for j,exp in enumerate(exp_range):
            m_data = int(exp)
            results_data[j,:] = self.exp_hoeffding(m_data,epsilon,task,delta)
        t2 = time.time()
        t_diff = t2 - t1
        return results_data, t_diff

    def ecdf(self,n):
        #step 8 
        data = self.gen_data(n,sort=True)
        empirical_cdf = np.linspace(0,1,n)
        return empirical_cdf
    
    def plot_results(self,results,delta=0.01):
        p_approx = np.log(results[:,0]+0)
        p_uncert = np.log(results[:,0]+results[:,4])
        p_bound = np.log(results[:,2])
        m_range = results[:,3]
        plt.figure(4)
        plt.plot(m_range,p_approx,label = 'p_f(m) (empirical approximation)')
        plt.plot(m_range,p_bound, label = 'concentration fm bound')
        plt.plot(m_range,p_uncert,'-.',label = 'p_f(m) uncertainty envelope')
        plt.title(f'confidence: {1-delta:.4f}')
        plt.legend()
        plt.xlabel('m_data')
        plt.ylabel('log(fm)')
        plt.show()

 
gch = GaussGlivenkoHoeffding()
###Part I: Sampling Data
#step 0: define hoeffding_bound, which takes epsilon and m as input
# and returns a bound on p as in the right hand side of eq (1) 
# Then plot log hoeffding bounds for various epsilons
t = np.arange(1,10**5)
eps_range = np.array([.01,.025,.05,.1])
for eps in eps_range:
    hb = gch.hoeffding_bound(t,eps)
    val_idx = hb<1
    t_filt = t[val_idx]
    hb_filt = hb[val_idx]
    plt.plot(t_filt,np.log(hb_filt),label = f"epsilon = {eps:.3f}")
    plt.plot(t_filt[0],np.log(hb_filt[0]),'*')
plt.xlabel('m')
plt.ylabel('log(hoeffding bound)')
plt.legend() 
# plt.show()

    
#step 1: generate data and look at it
N = 15000
dat = gch.gen_data(N,False)
plt.figure(2)
plt.hist(dat,bins = 50,density = True)
plt.show()

#step 2: check approximation of probability for event A (x\in [0,1])
#unless you choose a different interval, this number should be ~0.34
p_approx, p_true = gch.event_probability(dat)
p_err = np.abs(p_approx - p_true)
print(f"probability that x in [0,1] is roughly: {p_approx:.4f}, error = {p_err:.6f} ")

#step 3: define run_task which takes num data + task (see dict in constructor) and 
#returns approximate probability and true probability, error should be smaller 
p_approx, p_true = gch.run_task(10**6,"event",False)
p_err = np.abs(p_approx - p_true)
print(f"probability that x in [0,1] is roughly: {p_approx:.4f}, error = {p_err:.6f} ")


#step 4: define test_hoeffding_pre which evaluates the indicator of event in (2)
#takes m_data + epsilon + task (see dict in constructor)
indic = gch.run_hoeffding_pre(1000,.0025,"event")
print(f'output of run_hoeffding_pre is 0/1: {indic}')

#step 5: define run_hoeffding
p1,p2,p3 = gch.run_hoeffding(10000,1000,.0025,"event")
print(f"probabilities fm: {p1:.5f}, hoeffding bound: {p2:.5f}, diff: {p3:.5f}")


#step 6: define exp_hoeffding
#should return the same as run_hoeffding, but you need to define the number of runs
#(i.e. number of times to sample (2)) so that you may be confident, up to a confidence 
#encoded by 1-delta, that the true p for (2) is *below* the decreasing exponential in (1)
p1,p2,p3,m,delta = gch.exp_hoeffding(15000,.0025,'event',.005)

#step 7: Done!, run sweep and plot 
task = "event"
epsilon = 0.01
exp_range = np.linspace(4000,28000,7)
delt = 0.025
e_data, _ = gch.sweep_hoeffding(epsilon,exp_range,task,delta = delt)
gch.plot_results(e_data,delt) 

#step 8: define ecdf method
    #a. as a linspace; make sure you understand 
    # which x-values this linspace as empirical cdf correspond to 
    #b. evaluate the true cdf 
    #c check things work as expected
data = gch.gen_data(N,sort=True)
print(data)
t_cdf = gauss.cdf(data)
print(t_cdf)
e_cdf = gch.ecdf(data.shape[0])
if t_cdf.shape[0] != e_cdf.shape[0]:
    print('size mismatch on empirical cdf!')
plt.plot(data,t_cdf,'-.',label = 'true cdf')
plt.plot(data,e_cdf,'--',label = 'empirical cdf')
plt.xlabel('t')
plt.ylabel('P(x<= t)')
plt.legend() 
plt.title('cdf plots')
plt.show()

#Step 9:
#define glivenko_cdf, which should look similar to event_probability
#it will return an approximate cdf and a true cdf
e_cdf, t_cdf = gch.glivenko_cdf(data)
plt.plot(data,t_cdf-e_cdf,label='cdf error')
plt.legend() 
plt.title('cdf error')
plt.show()


#step 11: Done!, run sweep and plot 
task = "cdf"
#exp_range = np.arange(7,15)
e_data2, _ = gch.sweep_hoeffding(epsilon,exp_range,task,delta = delt)
gch.plot_results(e_data2,delt)

