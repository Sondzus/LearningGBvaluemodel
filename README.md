# Learning Groebner value model 

This folder contains code for recreating the analyses from ``[Learning a performance metric of Buchberger’s algorithm](https://arxiv.org/abs/2106.03676)" by Jelena Mojsilović, Dylan Peifer, Sonja Petrović. The goal is to predict the number of polynomial additions when computing Groebner bases using Buchberger's algorithm.


The python code for both regression and network training is [here](https://github.com/Sondzus/LearningGBvaluemodel). 

The code used to generate the data uses the RMI package and new functions [here](https://github.com/RandCommAlg/RMI/tree/randomToricIdeals), and has been merged to [Peifer's M2 folder](https://github.com/dylanpeifer/deepgroebner/tree/master/m2). 
