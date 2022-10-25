configs = {}
# params for mode 1 and mode 2
configs['base_dir'] = r'./jpegs_256/'
configs['output_path'] = r'./event_dvs/'
configs['num_frames'] = 30
configs['event_ratio'] = 0.15

# params for mode2
configs['gauss'] = {
    'CT_std': 0.03,
    'CT_min': 0.01
}