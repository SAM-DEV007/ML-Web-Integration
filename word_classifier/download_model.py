import os
import requests
import traceback


def download_model():
    '''
    Downloads and saves the model.

    If the model fails to download, it can be downloaded manually:
    https://drive.google.com/file/d/1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm/view?usp=sharing
    '''

    model_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), 'Model')
    destination = os.path.join(model_folder, 'WordClassifier_Model.h5')

    CHUNK_SIZE = 32768

    session = requests.Session()

    response = session.get('https://www.googleapis.com/drive/v3/files/1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm?alt=media&key=AIzaSyD0M_vstmhIl8M242-qn3K544gWoWlG-3A', stream = True)

    try:
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    datasize = f.write(chunk)
        print(f'Model downloaded at {destination}')
    except BaseException as err:
        traceback.print_exc()
        if os.path.exists(destination): 
            os.remove(destination)
        exit()


download_model()