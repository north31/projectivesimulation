# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried
"""

import __future__
import numpy as np
import matplotlib.pyplot as plt
import os# for current directory
import sys # include paths to subfolders for agents and environments
sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')
#create results folder if it doesn't exist already
if not os.path.exists('results'): 
	os.makedirs('results')
#import functions for initialising agents and environments, controlling their interaction etc
from rl_framework import *

"""This is the master file, which calls other functions to create agents and environments and
run the interaction between them. It may range over different values of parameters,
recording learning data and ultimately saveing it in the results folder."""

"""Choose which agent and which environment you would like to use, from EnvList() and AgentList().
Note that the parameters specified below must match the required input formats of the chosen types.
(These requirements are specified in the docstrings of the respective classes.) """
env_name = 'Neverending_Color'     #环境选择
agent_name = 'PS-generalization'     #代理选择
#The option multiple_agents creates a situation where multiple agents inhabit a single environment.
# Only specific classes of environment support this. 
multiple_agents = False


# performance evaluation
num_agents = 100               #代理数量
#When multiple_agents==False, 通过几个独立的代理重复学习过程以收集统计数据the learning process is repeated with several independent agents in order to gather statistics.
# If multiple_agents==True, 在单个环境中进行交互的代理数量this is the number of agents interacting in a single environment
max_num_trials = 100   #Each agents get several attempts at completing a task, e.g. finding the goal in a maze or reaching the top in the mountain car problem
max_steps_per_trial = 1000  #This parameter serves mostly to prevent agents getting stuck, since it terminates an attempt and resets the environment.
								#此参数主要用于防止代理程序卡住，因为它会终止尝试并重置环境
cishu=0
if not multiple_agents:
	print("not multiple")
	n_param_scan = 11  #Loop over different values of a certain parameter to test which one works best    循环遍历某个参数的不同值以测试哪个参数效果最好
	average_param_performance = np.zeros(n_param_scan)       #生成11位值为0的一维矩阵
	for i_param_scan in range(n_param_scan):
		ps_eta = i_param_scan * 0.1 #set the ps_eta parameter to different values  pa_eta=[0.1,0.2~~1.1]     η值，与奖励有关
		average_learning_curve = np.zeros(max_num_trials)  #this will record the rewards earned at each trial, averaged over all agents    用来记录每次尝试获得的奖励
		for i_agent in range(num_agents):	#train one agent at a time, and iterate over several agents	    一次训练一个代理，遍历多个代理
			env_config = 2, 1, max_num_trials  #need to pass the number of agents for a multi-agent environment    对于多代理环境，需要传递代理数量。 数组[2,1,100]
			env = CreateEnvironment(env_name, env_config)     #构建好的env
			if agent_name in ('PS-basic', 'PS-sparse'):
				num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections = env.num_actions, env.num_percepts_list, 0, ps_eta, 'softmax', 1, 0
				agent_config = [num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections]
			elif agent_name == 'PS-flexible':
				num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax = env.num_actions, 0, ps_eta, 'softmax', 1
				agent_config = [num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax]
			elif agent_name == 'PS-generalization':
				print("in elis",cishu)
				num_actions, num_percepts_list, generalization, gamma_damping, eta_glow_damping, policy_type, beta_softmax, hash_hash_bool, majority_vote, majority_time, number_votes = env.num_actions, env.num_percepts_list, True, 0, 1, 'standard', 1, False, True, 10, 10
				agent_config = [num_actions, num_percepts_list, generalization, gamma_damping, eta_glow_damping, policy_type, beta_softmax, hash_hash_bool, majority_vote, majority_time, number_votes]
			agent = CreateAgent(agent_name, agent_config)	
			interaction = Interaction(agent, env)
			learning_curve = interaction.single_learning_life(max_num_trials, max_steps_per_trial) #This function executes a 'learning life' between the agent and the environment
			average_learning_curve += learning_curve/num_agents
			cishu += 1
			print(cishu)
		average_param_performance[i_param_scan] = average_learning_curve[-1]  #The performance for a given value of the parameter is taken to be the average reward at the last trial.
	print("for over2")
elif multiple_agents:  #Here we do not iterate over a number of independent agents, but rather have them all in a single instance of the environment.
	print("multiple")
	n_param_scan = 5
	average_param_performance = np.zeros(n_param_scan)
	for i_param_scan in range(n_param_scan):
		ps_gamma = 10**(-i_param_scan)
		env_config = num_agents, 40, 4  #need to pass the number of agents for a multi-agent environment
		env = CreateEnvironment(env_name, env_config) 
		if agent_name in ('PS-basic', 'PS-sparse'):
			num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections = env.num_actions, env.num_percepts_list, ps_gamma, 1, 'standard', 1, 0
			agent_config = [num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections]
		elif agent_name == 'PS-flexible':
			num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax = env.num_actions, 0, ps_gamma, 'softmax', 1
			agent_config = [num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax]
		elif agent_name == 'PS-generalization':
			num_actions, num_percepts_list, generalization, gamma_damping, eta_glow_damping, policy_type, beta_softmax, hash_hash_bool, majority_vote, majority_time, number_votes = env.num_actions, env.num_percepts_list, True, 0, 1, 'standard', 1, False, True, 10, 10
			agent_config = [num_actions, num_percepts_list, generalization, gamma_damping, eta_glow_damping, policy_type, beta_softmax, hash_hash_bool, majority_vote, majority_time, number_votes]
			
		agent_list = []
		for i_agent in range(num_agents):	
			agent_list.append(CreateAgent(agent_name, agent_config=agent_config))
         
		interaction = Interaction_Multiple(agent_list, env)
		learning_curve = interaction.single_learning_life(max_num_trials, max_steps_per_trial)
		average_learning_curve = learning_curve.sum(axis=1)/num_agents
		average_param_performance[i_param_scan] = average_learning_curve[-1]
 


# Saving files
current_file_directory = os.path.dirname(os.path.abspath(__file__))
if num_agents == 1:
	if agent_name in ('PS-generalization', 'PS-sparse'):
		np.savetxt(current_file_directory+'/results'+'/h_matrix', agent.h_matrix.toarray(), fmt='%.2f', delimiter=',')
		np.savetxt(current_file_directory+'/results'+'/g_matrix', agent.h_matrix.toarray(), fmt='%.3f', delimiter=',')
	else:
		np.savetxt(current_file_directory+'/results'+'/h_matrix', agent.h_matrix, fmt='%.2f', delimiter=',')
		np.savetxt(current_file_directory+'/results'+'/g_matrix', agent.h_matrix, fmt='%.3f', delimiter=',')
else:
	np.savetxt(current_file_directory+'/results'+'/param_performance', average_param_performance, fmt='%.3f', delimiter=',')
np.savetxt(current_file_directory+'/results'+'/learning_curve', average_learning_curve, fmt='%.3f', delimiter=',')

plt.figure()
plt.plot(average_learning_curve)
plt.show()