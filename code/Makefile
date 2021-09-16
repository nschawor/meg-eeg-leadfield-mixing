figures_spatial_patterns:
	python3 fig2a_influence_of_reference.py
	python3 fig2b_non_local_origin_alpha_mu.py
	python3 fig2c_eye_movements_instead_of_theta.py
	
figures_simulation:
	python3 fig3a_sim_plot_head_topographies.py	
	python3 fig3b_sim_plot_alpha_map.py
	python3 fig4_sim_changing_dipole_orientation.py

preprocessing_data:
    python3 proc_0_convert_data_to_mne.py
	python3 proc_1_compute_center_frequencies.py
	python3 proc_2_compute_ssd.py
	python3 proc_3_compute_activation_patterns.py

figures_data:
	python3 fig5_alpha_map_single_subject.py
	python3 fig6_complexity_measure_across_participants.py
	python3 fig7_time_resolved_rhythm_contributions.py

all:
	figures_spatial_patterns figures_simulation preprocessing_data figures_data 