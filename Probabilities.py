#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


# In[2]:


#construiesc reteaua
#probabilitati pentru variabile neconditionale
substances = DiscreteDistribution({'T': 0.09, 'F': 0.91})
tobacco = DiscreteDistribution({'T': 0.31, 'F': 0.69})
airPollution = DiscreteDistribution({'T': 0.9, 'F': 0.1})
viruses = DiscreteDistribution({'T': 0.18, 'F': 0.82})


# In[3]:


#probabilitati conditionale
#simptome
cough = ConditionalProbabilityTable(
        [[ 'T', 'T', 'T', 0.89 ],
         [ 'T', 'T', 'F', 0.11 ],
         [ 'T', 'F', 'T', 0.77 ],
         [ 'T', 'F', 'F', 0.23 ],
         [ 'F', 'T', 'T', 0.75 ],
         [ 'F', 'T', 'F', 0.25 ],
         [ 'F', 'F', 'T', 0.69 ],
         [ 'F', 'F', 'F', 0.31 ]], [airPollution,tobacco])  
painOrPressure = ConditionalProbabilityTable(
        [[ 'T', 'T','T', 0.72 ],
         [ 'T', 'T','F', 0.28 ],
         [ 'T', 'F','T', 0.56 ],
         [ 'T', 'F','F', 0.44 ],
         [ 'F', 'T','T', 0.31 ],
         [ 'F', 'T','F', 0.69 ],
         [ 'F', 'F','T', 0.11 ],     
         [ 'F', 'F','F', 0.89 ]], [substances,tobacco]) 
fever = ConditionalProbabilityTable(
        [[ 'T','T', 0.39 ],
         [ 'T','F', 0.61 ],
         [ 'F','T', 0.08 ],     
         [ 'F','F', 0.92 ]], [viruses]) 

#boli 
lungCancer = ConditionalProbabilityTable(
        [[ 'T', 'T','T', 0.38 ],
         [ 'T', 'T','F', 0.62 ],
         [ 'T', 'F','T', 0.13 ],
         [ 'T', 'F','F', 0.87 ],
         [ 'F', 'T','T', 0.15 ],
         [ 'F', 'T','F', 0.85 ],
         [ 'F', 'F','T', 0.005 ],     
         [ 'F', 'F','F', 0.995 ]], [cough,painOrPressure]) 

pneumonia = ConditionalProbabilityTable(
        [[ 'T', 'T','T', 0.28 ],
         [ 'T', 'T','F', 0.72 ],
         [ 'T', 'F','T', 0.21 ],
         [ 'T', 'F','F', 0.79 ],
         [ 'F', 'T','T', 0.17 ],
         [ 'F', 'T','F', 0.83 ],
         [ 'F', 'F','T', 0.09 ],       
         [ 'F', 'F','F', 0.91 ]], [fever,painOrPressure])


#tratament
chemotherapy = ConditionalProbabilityTable(
        [[ 'T','T', 0.87 ],
         [ 'T','F', 0.13 ],
         [ 'F','T', 0.003],
         [ 'F','F', 0.997]], [lungCancer]) 
surgery = ConditionalProbabilityTable(
        [[ 'T','T', 0.23 ],
         [ 'T','F', 0.77 ],
         [ 'F','T', 0.03 ],    
         [ 'F','F', 0.97 ]], [lungCancer]) 
medication = ConditionalProbabilityTable(
        [[ 'T','T', 0.99 ],
         [ 'T','F', 0.01 ],
         [ 'F','T', 0.25 ],
         [ 'F','F', 0.75 ]], [pneumonia])


# In[4]:


m1 = State(substances, name="substances")
m2 = State(tobacco, name="tobacco")
m3 = State(airPollution, name="airPollution")
m4 = State(viruses, name="viruses")
s1 = State(cough, name="cough")
s2 = State(painOrPressure, name="painOrPressure")
s3 = State(fever, name="fever")
b1 = State(lungCancer, name="lungCancer")
b2 = State(pneumonia, name="pneumonia")
t1 = State(medication, name="medication")
t2 = State(surgery, name="surgery")
t3 = State(chemotherapy, name="chemotherapy")


# In[5]:


# Create the Bayesian network object
net = BayesianNetwork("Lung diseases network")

# Add the states to the network 
net.add_states(m1,m2,m3,m4,s1,s2,s3,b1,b2,t1,t2,t3)

# Connect the states
#primul este parintele al doilea e fiul
net.add_edge(m1, s2)
net.add_edge(m2, s1)
net.add_edge(m2, s2)
net.add_edge(m3, s1)
net.add_edge(m4, s3)
net.add_edge(s1, b1)
net.add_edge(s2, b1)
net.add_edge(s2, b2)
net.add_edge(s3, b2)
net.add_edge(b2, t1)
net.add_edge(b1, t2)
net.add_edge(b1, t3)


# In[6]:


# Integrate everything
net.bake()

#Draw the network
net.plot()


# In[7]:


#1.probabiliatatea de a avea un virus stiind ca avem febra
observations = {'fever' : 'T'}
beliefs = map( str, net.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( net.states, beliefs ) ))


# In[8]:


#2.Probabilitatea de a avea dureri stiind ca avem pneumonie
observations = {'pneumonia' : 'T'}
beliefs = map( str, net.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( net.states, beliefs ) ))


# In[9]:


#3.Probabilitatea de a tusi stiind ca avem cancer
observations = {'lungCancer' : 'T'}
beliefs = map( str, net.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( net.states, beliefs ) ))


# In[10]:


#4.Probabilitatea de a fi fost expus poluarii stiind ca nu avem cancer
observations = {'lungCancer' : 'F'}
beliefs = map( str, net.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( net.states, beliefs ) ))


# In[11]:


#5.Probabilitatea de a avea cancer stiind ca am facut operatie si chimioterapie
observations = {'surgery' : 'T','chemotherapy':'T'}
beliefs = map( str, net.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( net.states, beliefs ) ))


# In[12]:


net.probability(numpy.array([None,None,None,None,None,'T',None,None,None,None,None,None], ndmin=2))


# In[11]:


# probabilitatile marginale a fiecarui nod din graf
net.marginal()


# In[12]:


# the logarithm of the probability de a nu fi expus la airPollution si de a avea painOrPressure si de a fi avut surgery
net.log_probability(numpy.array([None,None,'F',None,None,'T',None,None,None,None,'T',None], ndmin=2))


# In[18]:


#joint pe tabela de cough
cough.joint()


# In[ ]:




