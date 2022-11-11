# ------ File description ------
"""
This code takes behavioral training data (outside scanner) from the experiment holding folder, analyzes training performance (i.e. proportion correct same/different judgments), and saves plots showing the results.
For this to work, 'holding' needs to be populated for the subjects defined in 'subjects'.
"""

# ------ Imports ------
from IPython import embed as shell
import os
import glob
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as pl
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import tools.general_tools as general_tools
import tools.BIDS_tools as BIDS_tools

# ------ Set paths to data ------
base_of_BIDS_tree='/fmri2/PI/jbrascamp/fmri_data/templeton_2a_pilot3_FS22/'
main_script_folder_name='analysis_code_Templeton_2a'
path_to_holding_training = 'holding/training/'

# ------ Set global variables ------
readme_text="This derivatives folder contains plots of the participants' behavioral performance, i.e. proportion correct memory performance, etc."
derivatives_folder=BIDS_tools.prepare_derivatives_folder(base_of_BIDS_tree, main_script_folder_name, readme_text)
subjects=['108','109']
task_label='training'

# ------ Define methods ------
def plot_overall_training_performance(analysis_dictionaries, output_folder):
	
	"""
	Plot the training performance computed by computeAccuracy.
	Arguments:
		analysis_dictionaries: (dict) a dictionary of the results per cue type and per mem to cue duration (output of computeAccuracy)
	Returns:
		Nothing. But creates a plot with current subject's performance in that subject's folder in the current analysis' derivatives_folder. 
	"""

	colors_retro_post=['r','b']

	my_figure, my_ax = pl.subplots(1,1, figsize = (8,4))
	my_ax.set_xlim(xmin = 0.5, xmax = 3.8)
	my_ax.set_ylim(ymin = .3, ymax = 1.0)

	for idx,dict in enumerate(analysis_dictionaries):	#0 for retro, 1 for post
		
		pl.scatter(dict.keys(), dict.values(), color=colors_retro_post[idx],label=['Retro cue','Post cue'][idx])
		pl.plot(dict.keys(),dict.values(), color=colors_retro_post[idx])
		
	pl.legend()

	pl.xlabel('Interval memory array to cue (s)')
	pl.ylabel('Prop correct same/diff responses')
	
	my_figure.suptitle('Training performance Templeton 2a Pilot 3, subject: ' + subject)
	my_figure.tight_layout()
	
	entities = {
		'sub': subject,
		'task': task_label,
		'desc': 'overallAccuracy',
		'extension': '.png'
	}
	
	file_name=BIDS_tools.my_build_bids_filename(entities)
	
	pl.savefig(os.path.join(output_folder,file_name))
	pl.close()

def calculateAccuracy():
	"""
	Calculates the training performance per cue type and per mem to cue duration.
	Returns:
		analysis_dictionaries: (dict) a dictionary of the results per cue type and per mem to cue duration
	"""

	list_of_mem_display_to_cue_duration_s = [[0.75, 1.6],[0.75, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]]
	analysis_dictionaries = [{},{}]
	for retro_post_index in [0,1]:	#0 for retro, 1 for post
		accuracy_dict = {}
		for mem_display_to_cue_duration_s in list_of_mem_display_to_cue_duration_s[retro_post_index]:
			this_cue_and_interval_indices=training_data.index[(training_data['retro_0_post_1_training']==retro_post_index) & (round(training_data['mem_display_to_cue_duration_s'], 2) == mem_display_to_cue_duration_s)].tolist()
			this_cue_and_interval_actual = []
			for trial in this_cue_and_interval_indices:
				this_cue_and_interval_actual.append(training_data.at[trial, 'change_or_no_training'])

			this_cue_and_interval_response = []
			for trial in this_cue_and_interval_indices:
				this_cue_and_interval_response.append(training_data.at[trial, 'sd_training_resp.keys'])
			
			for idx,resp in enumerate(this_cue_and_interval_response):
				this_cue_and_interval_response[idx] = change_or_no_key_dict[resp]

			this_cue_and_interval_accuracy = []
			for idx in range(len(this_cue_and_interval_response)):
				this_cue_and_interval_accuracy.append(this_cue_and_interval_response[idx] == this_cue_and_interval_actual[idx])

			accuracy_dict[mem_display_to_cue_duration_s] = np.sum(this_cue_and_interval_accuracy)/len(this_cue_and_interval_accuracy)
		analysis_dictionaries[retro_post_index] = accuracy_dict
	return analysis_dictionaries

def calculateAndPlotTrueTraining(output_folder):
	row_indices_long_intervals_training=training_data.index[training_data['training_trial_index_long_this_block'].notnull()].tolist()

	all_retention_intervals=training_data['mem_display_to_cue_duration_s'][row_indices_long_intervals_training].tolist()
	all_cue_types=training_data['retro_0_post_1_training'][row_indices_long_intervals_training].tolist()
	all_change_or_no_responses=[change_or_no_key_dict[this_response] for this_response in training_data['sd_training_resp.keys'][row_indices_long_intervals_training].tolist()]
	all_change_or_no_actuals=training_data['change_or_no_training'][row_indices_long_intervals_training].tolist()

	all_correct_or_nos=[all_change_or_no_responses[index]==all_change_or_no_actuals[index] for index in range(len(all_change_or_no_actuals))]

	prop_correct_retro=np.average([this_correct_or_no for this_index,this_correct_or_no in enumerate(all_correct_or_nos) if all_cue_types[this_index]==0])
	overall_prop_correct_post=np.average([this_correct_or_no for this_index,this_correct_or_no in enumerate(all_correct_or_nos) if all_cue_types[this_index]==1])
	retention_interval_correct_pairs_post=[[all_retention_intervals[this_index],this_correct_or_no] for this_index,this_correct_or_no in enumerate(all_correct_or_nos) if all_cue_types[this_index]==1]

	all_retention_intervals_post=list(set([element[0] for element in retention_interval_correct_pairs_post]))
	all_retention_intervals_post.sort()
	prop_correct_per_interval_post=[np.average([element[1] for element in retention_interval_correct_pairs_post if element[0]==this_interval]) for this_interval in all_retention_intervals_post]
	num_obs_per_interval_post=[len([element[1] for element in retention_interval_correct_pairs_post if element[0]==this_interval]) for this_interval in all_retention_intervals_post]

	f = pl.figure(figsize = (10,10))
	s = f.add_subplot(1,1,1)
	s.scatter(all_retention_intervals_post,prop_correct_per_interval_post)
	s.plot([all_retention_intervals_post[0],all_retention_intervals_post[-1]],[overall_prop_correct_post,overall_prop_correct_post])
	s.plot([all_retention_intervals_post[0],all_retention_intervals_post[-1]],[prop_correct_retro,prop_correct_retro])
	[pl.text(all_retention_intervals_post[this_index]+.02, prop_correct_per_interval_post[this_index]+.02, str(num_obs_per_interval_post[this_index]), fontsize=10) for this_index in range(len(all_retention_intervals_post))]

	pl.xlabel('Retention interval (s)')
	pl.ylabel('Proportion correct')
	s.set_ylim(ymin = -0.05, ymax = 1.05)
			
	f.suptitle('Participant '+subject+'\nOveral props correct post (blue), retro (orange)='+str(overall_prop_correct_post)+', '+str(prop_correct_retro))

	entities = {
		'sub': subject,
		'task': task_label,
		'desc': 'trueTrainingAccuracy',
		'extension': '.png'
	}
	
	file_name=BIDS_tools.my_build_bids_filename(entities)
	pl.savefig(os.path.join(output_folder,file_name))

def printAccuracy():
	"""
	Print overall retro and post cue accuracies for current subject to the output
	screen.
	Returns:
		Nothing.
	"""
	retro_trial_indices = training_data.index[training_data['retro_0_post_1_training'] == 0].tolist()

	post_trial_indices = training_data.index[training_data['retro_0_post_1_training'] == 1].tolist()

	retro_trial_actual = []
	for trial in retro_trial_indices:
		retro_trial_actual.append(training_data.at[trial, 'change_or_no_training'])

	post_trial_actual = []
	for trial in post_trial_indices:
		post_trial_actual.append(training_data.at[trial, 'change_or_no_training'])

	retro_trial_response = []
	for trial in retro_trial_indices:
		retro_trial_response.append(training_data.at[trial, 'sd_training_resp.keys'])

	post_trial_response = []
	for trial in post_trial_indices:
		post_trial_response.append(training_data.at[trial, 'sd_training_resp.keys'])

	retro_trial_recoded = np.zeros([len(retro_trial_response), 1])
	for idx,resp in enumerate(retro_trial_response):
		retro_trial_recoded[idx] = change_or_no_key_dict[resp]

	post_trial_recoded = np.zeros([len(post_trial_response), 1])
	for idx,resp in enumerate(post_trial_response):
		post_trial_recoded[idx] = change_or_no_key_dict[resp]

	retro = []
	for idx in range(len(retro_trial_response)):
		if retro_trial_recoded[idx] == retro_trial_actual[idx]:
			retro.append(1)
		else:
			retro.append(0)
	retro_accuracy = np.sum(retro)/len(retro_trial_response)

	post = []
	for idx in range(len(post_trial_response)):
		if post_trial_recoded[idx] == post_trial_actual[idx]:
			post.append(1)
		else:
			post.append(0)
	post_accuracy = np.sum(post)/len(post_trial_response)

	print('\nsubject: ' + subject)
	print('retrocue accuracy: ' + str(round(retro_accuracy*100,2)))
	print('postcue accuracy: ' + str(round(post_accuracy*100,2)) + '\n')

# ------ Main body of script ------
for subject in subjects:
	training_file_path = glob.glob(os.path.join(base_of_BIDS_tree,path_to_holding_training,'sub-'+subject + '*.csv'))[0]
	training_data = pd.read_csv(training_file_path)

	participant_response_mapping=training_data['mapping'].dropna().tolist()[0]
	if participant_response_mapping==23:
		change_or_no_key_dict={'down':1.0,'left':0.0}
	elif participant_response_mapping==32:
		change_or_no_key_dict={'left':1.0,'down':0.0}
	else:
		print('cannot figure out which key mapping was used')

	output_folder=os.path.join(derivatives_folder,'sub-'+subject,'figures')
	os.makedirs(output_folder, exist_ok=True)

	plot_overall_training_performance(calculateAccuracy(), output_folder)
	calculateAndPlotTrueTraining(output_folder)
	printAccuracy()