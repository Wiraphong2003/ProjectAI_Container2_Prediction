import numpy as np


classes = [
    'dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock',
    'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane',
    'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops',
    'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog',
    'cow', 'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
    'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine', 'breathing',
    'crying_baby', 'hand_saw', 'coughing', 'glass_breaking', 'snoring',
    'toilet_flush', 'pig', 'washing_machine', 'clock_tick', 'sneezing', 'rooster',
    'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets'
]

def predictcar(model,Arrl):
    result = model.predict(Arrl)
    return classes[np.argmax(result)]


