# -*- coding: utf-8 -*-
"""
@author: ChernHong Lim
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

def CL_FQRC_Train(X_train, y_train, binNum, visualize):
    yUnique = np.unique(y_train)    
    fqmf = [];
    fqmf_mu = [];       
    for i in yUnique:
        X_filt = X_train[y_train == i] 
        mf,mu = build4tuplesMF(X_filt , binNum)
        fqmf.append(mf)
        fqmf_mu.append(mu)

    if(visualize):        
        mfVisualize(fqmf)
        histVisualize(fqmf, fqmf_mu)
    return fqmf

def CL_FQRC_Predict(X_test, fqmf, visualize):
    output,feaDegreeMF = inference(X_test, fqmf)   
    
    if(visualize):
        infVisualize(X_test, fqmf, feaDegreeMF)
    return output


def build4tuplesMF(feaMat , binNum):
    
    # Parameters initialization
    B = binNum #num of bin
    feaMat = feaMat #feature matrix
    J = feaMat.shape[1] #num of features (also indicate that num of 4-tuples membership function will be builded at the end)
    mf = np.zeros((J,4))
    mu = [];
    
    for j in range(0,J): #start building MF
        # calculate the bin width, v
        v = (float(np.amax(feaMat[:,j])) - float(np.amin(feaMat[:,j]))) / float(B)            
                  
        # count the ocurrence of the data in the bin and represent in histogram
        h = np.histogram(feaMat[:,j],B);
        N = h[0]
        xout = h[1]        

        # calculate how many bins which have distributed data > 0 (denoted as b)
        b = 0
        for n in range(0,len(N)):
            if (N[n] > 0):
                b = b + 1;
        
        # Calculate mean value (mu) for the histogram 
        histMean = float(sum(N)) / b;
                      
    
        """
        % Find 4-tuple trapezoid position from histogram
        %     _________
        %    /|       |\
        %   / |       | \
        %  /  |       |  \
        % c   a       b   d
        %
        % a-c : alpha
        % d-b : beta
        % 4-tuple = [a,b,alpha,beta]
        """
        
        # Scan from left to right to obtain a value 
        for n in range(0,len(N)):
           if(N[n] >= histMean):
               a = xout[n] #include the offset to get the lower boundary of that bar
               break
           
        # Scan from right to left to obtain b value
        for n in range(len(N)-1,-1,-1):
            if(N[n] >= histMean):
                b = xout[n+1] #include the offset to get the upper boundary of that bar 
                break
            
        # obtain a value 
        c = xout[0] - v;    
    
        # obtain b value
        d = xout[len(xout)-1] + v;
                
        # compute alpha
        alpha = a - c;
        
        # compute beta
        beta = d - b;
        
        # Output
        mf[j,:] = [a,b,alpha,beta]
        mu.append([histMean, N, xout])  
          
#        #Plot hist for visualization
#        width = 0.9 * (xout[1] - xout[0])        
#        center = (xout[:-1] + xout[1:]) / 2
#        plt.bar(center, N, align='center', width=width)
#        plt.plot([a,b],[histMean,histMean],'ro')
#        datax = [c,a,b,d]
#        datay = np.array([0,histMean,histMean,0])
#        plt.plot(datax,datay,'r--')
#        plt.show()

    return mf,mu

  


def inference( feaVec, fqmf):
    J = len(fqmf[0]) #num of feature
    K = len(fqmf) #num of class
    
    # Obtain degree of membership for each feature value
    feaDegreeMF = np.zeros((K,J))

    for k in range(0,K):
        for j in range(0,J):
            degreeMF = membershipVal(feaVec[j], fqmf[k][j]);       
            feaDegreeMF[k][j] = degreeMF;
   
    temp = copy.copy(feaDegreeMF)
    temp[feaDegreeMF > 0] = 1
    hitCount = np.sum(temp,axis=1)
    sumOfdegreeMF = np.sum(feaDegreeMF,axis=1)
    ratio = np.divide(sumOfdegreeMF, np.amax(hitCount))
    sumRatio = sum(ratio)
    normOfdegreeMF = np.divide(ratio,sumRatio) #Normalization
    output = normOfdegreeMF
    
    if(np.isnan(sum(output))==True or np.isinf(sum(output))==True):
        output = np.zeros((1,K))    
    return output, feaDegreeMF



def membershipVal(fvalue, mf):
    # mf -> 4-tuples number retrieve from FQRC (mf = [a b alpha beta])
    a = mf[0]
    b = mf[1]
    alpha = mf[2]
    beta = mf[3]
    
    if (fvalue >= a and fvalue <= b):      # f_value within [a,b]
        degreeMF = 1    
    elif (fvalue >= a-alpha and fvalue < a):    # x within [a-alpha,a]
        degreeMF = (fvalue - a + alpha) / alpha    
    elif (fvalue > b and fvalue <= b+beta):   # x within [b,b+beta]
        degreeMF = (b + beta - fvalue) / beta    
    else:
        degreeMF = 0    

    return degreeMF
    

def histVisualize(fqmf, fqmf_mu):
    if(type(fqmf) is list):
        J = len(fqmf_mu[0]) #num of feature
        K = len(fqmf_mu) #num of class        
        fig, axes = plt.subplots(J, K, figsize=(8,8))
        fig.tight_layout()
        for k in range(0,K):
            mf = fqmf[k]
            mu = fqmf_mu[k]
            for j in range(0,J):
                #convert back to [c a b d] fuzzy tuple
                c = mf[j,0] - mf[j,2]
                a = mf[j,0]
                b = mf[j,1]
                d = mf[j,1] + mf[j,3] 
                                 
                xout = mu[j][2]
                N = mu[j][1]
                histMean = mu[j][0]
                #Plot hist for visualization
                width = 0.9 * (xout[1] - xout[0])        
                center = (xout[:-1] + xout[1:]) / 2                
                datax = [c,a,b,d]
                datay = np.array([0,histMean,histMean,0])
                axes[j,k].bar(center, N, align='center', width=width)
                axes[j,k].plot([a,b],[histMean,histMean],'ro')
                axes[j,k].plot(datax,datay,'r--')
#                plt.show()
# 
    else:
        mf = copy.copy(fqmf)
        mu = copy.copy(fqmf_mu)
        J = len(mf)
        fig, axes = plt.subplots(J)
        fig.tight_layout()
        for j in range(0,J):      
            #convert back to [c a b d] fuzzy tuple
            c = mf[j,0] - mf[j,2]
            a = mf[j,0]
            b = mf[j,1]
            d = mf[j,1] + mf[j,3] 
                             
            xout = mu[j][2]
            N = mu[j][1]
            histMean = mu[j][0]
            
            #Plot hist for visualization
            width = 0.9 * (xout[1] - xout[0])        
            center = (xout[:-1] + xout[1:]) / 2                
            datax = [c,a,b,d]
            datay = np.array([0,histMean,histMean,0])
            axes[j].bar(center, N, align='center', width=width)
            axes[j].plot([a,b],[histMean,histMean],'ro')
            axes[j].plot(datax,datay,'r--')
#            plt.show()
            
    plt.show() 
    
    
def mfVisualize(fqmf):    
    if(type(fqmf) is list): # to solve issue if only one class in the mf
        J = len(fqmf[0]) #num of feature
        K = len(fqmf) #num of class        
        fig, axes = plt.subplots(J, K, figsize=(8,8))
        fig.tight_layout()
        for k in range(0,K):
            mf = fqmf[k]
            xmin = np.amin(mf[:,0:2])-np.amax(mf[:,2:4])
            xmax = np.amax(mf[:,0:2])+np.amax(mf[:,2:4])
            for j in range(0,J):
                #convert back to [c a b d] fuzzy tuple
                c = mf[j,0] - mf[j,2]
                a = mf[j,0]
                b = mf[j,1]
                d = mf[j,1] + mf[j,3] 
                                 
                datax = [c,a,b,d]
                datay = np.array([0,1,1,0])
                axes[j,k].plot(datax,datay)
                axes[j,k].set_xlim(xmin,xmax)
                axes[j,k].set_ylim(0,1.1)
#                plt.ylabel('Degree of Membership')
#                plt.axis([-1, 10, 0, 1.1])
#                plt.show() 
    else:
        mf = copy.copy(fqmf)
        J = len(mf)
        fig, axes = plt.subplots(J)
        fig.tight_layout()
        xmin = np.amin(mf[:,0:2])-np.amax(mf[:,2:4])
        xmax = np.amax(mf[:,0:2])+np.amax(mf[:,2:4])
        for j in range(0,J):      
            #convert back to [c a b d] fuzzy tuple
            c = mf[j,0] - mf[j,2]
            a = mf[j,0]
            b = mf[j,1]
            d = mf[j,1] + mf[j,3] 
                             
            datax = [c,a,b,d]
            datay = np.array([0,1,1,0])
            axes[j].plot(datax,datay)
#            axes[j].ylabel('Degree of Membership')
            axes[j].set_xlim(xmin,xmax)
            axes[j].set_ylim(0,1.1)
            
    plt.show() 
            

def infVisualize(feaVec, fqmf, feaDegreeMF):    
    if(type(fqmf) is list): # to solve issue if only one class in the mf
        J = len(fqmf[0]) #num of feature
        K = len(fqmf) #num of class        
        fig, axes = plt.subplots(J, figsize=(8,8))
        fig.tight_layout()
        xminFinal = 0
        xmaxFinal = 0
        
        for j in range(0,J):            
            for k in range(0,K):
                mf = fqmf[k]
                xmin = np.amin(mf[:,0:2])-np.amax(mf[:,2:4])
                xmax = np.amax(mf[:,0:2])+np.amax(mf[:,2:4])
                
                if(xmin < xminFinal):
                    xminFinal = xmin
                
                if(xmax > xmaxFinal):
                    xmaxFinal = xmax
                    
                c = mf[j,0] - mf[j,2]
                a = mf[j,0]
                b = mf[j,1]
                d = mf[j,1] + mf[j,3] 
                                 
                datax = [c,a,b,d]
                datay = np.array([0,1,1,0])
                axes[j].plot(datax,datay)   
                
            axes[j].legend((feaDegreeMF[:,j]))                
            axes[j].set_title("Degree of Membership")

        for j in range(0,J):
            axes[j].plot([feaVec[j],feaVec[j]],[0,1.1],'r--')
        
        for (n), subplot in np.ndenumerate(axes):
            subplot.set_xlim(xminFinal,xmaxFinal)
            subplot.set_ylim(0,1.1)        
         
    else:
        mf = copy.copy(fqmf)
        J = len(mf)
        fig, axes = plt.subplots(J)
        xmin = np.amin(mf[:,0:2])-np.amax(mf[:,2:4])
        xmax = np.amax(mf[:,0:2])+np.amax(mf[:,2:4])
        for j in range(0,J):      
            #convert back to [c a b d] fuzzy tuple
            c = mf[j,0] - mf[j,2]
            a = mf[j,0]
            b = mf[j,1]
            d = mf[j,1] + mf[j,3] 
                             
            datax = [c,a,b,d]
            datay = np.array([0,1,1,0])
            axes[j].plot(datax,datay)
#            axes[j].ylabel('Degree of Membership')
            axes[j].set_xlim(xmin,xmax)
            axes[j].set_ylim(0,1.1)
            axes[j].plot([feaVec[j],feaVec[j]],[0,1.1],'r-')
            axes[j].legend((feaDegreeMF[:,j]))                
            axes[j].set_title("Degree of Membership")
            
    plt.show() 




    
"""""""""""""""""""""
Test build mf
""""""""""""""""""""" 
#a = np.array([(1,4,6),(2,5,7),(2,5,8),(3,4,8),(3,5,7),(2,6,9)])
#mf,mu = build4tuplesMF(a,3)
#mfVisualize(mf)
#histVisualize(mf,mu)



"""""""""""""""""""""
Test build mf
""""""""""""""""""""" 
#mf = np.array([4, 5, 1, 1])
#membership_degree = membershipVal(3.7, mf)


"""""""""""""""""""""
Test inference
""""""""""""""""""""" 
#a = np.array([(1,4,6),(2,5,7),(2,5,8),(3,4,8),(3,5,7),(2,6,9)]) # Class 1
#mf_a,mu_a = build4tuplesMF(a,3)
##mfVisualize(mf_a)
##histVisualize(mf_a,mu_a)
#
#b = np.array([(4,6,9),(4,7,10),(3,7,11),(5,7,10),(4,8,10),(5,8,11)]) # Class 2
#mf_b,mu_b = build4tuplesMF(b,3)
##mfVisualize(mf_b)
##histVisualize(mf_b,mu_b)
#
#mf_all = [] # To append membership functions for all classes
#mf_all.append(mf_a)
#mf_all.append(mf_b)
#
#feaVec = np.array([2.5,4.5,9.5]) # New input with feature values
#output,feaDegreeMF = inference(feaVec, mf_all)
#infVisualize(feaVec, mf_all, feaDegreeMF)



"""""""""""""""""""""
Test CL_FQRC_Train and CL_FQRC_Predict
"""""""""""""""""""""
#a = np.array([(1,4,6),(2,5,7),(2,5,8),(3,4,8),(3,5,7),(2,6,9),(4,6,9),(4,7,10),(3,7,11),(5,7,10),(4,8,10),(5,8,11)])
#a_groundTruth = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
#fqmf = CL_FQRC_Train(a, a_groundTruth, 3, True)
#feaVec = np.array([2.5,4.5,9.5]) %--> new testing input
#output = CL_FQRC_Predict(feaVec, fqmf, True)
#print 'output:' + str(output)







