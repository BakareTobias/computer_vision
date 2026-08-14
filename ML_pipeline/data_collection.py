from modules.data_collection_module import DataCollection


data_collect = DataCollection()

data_collect.capture_data(
    label='A',
    capture_rate=5,
    path_to_datasource_file='1.avi',
    path_to_destination_folder='ML_pipeline',
    bothHands=True,
    )