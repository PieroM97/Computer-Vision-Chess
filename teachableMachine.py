import tensorflow.keras
import tensorflow as tf

from PIL import Image, ImageOps
from MyChessFunction import *

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model = None

def set_model(flag):
    global model

    if(flag == 0 ):
        model = tensorflow.keras.models.load_model('default.h5')
    elif(flag == 1 ) :
        model = tensorflow.keras.models.load_model('my_keras.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(64, 224, 224, 3), dtype=np.float32)


def find_pieces(boxes , str ):

    positional_matrix = [[ 0 for x in range(8)] for y in range(8)]

    i = 0


    for box in boxes:

        size = (224, 224)

        # Conversione da OpenCV a PIL
        box = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
        box = Image.fromarray(box)

        # Ridimensionamento immagine
        image = ImageOps.fit(box, size, Image.ANTIALIAS)

        # Immagine convertita in array
        image_array = np.asarray(image)

        # Load the image into the array
        data[i] = image_array

        i = i +1

    # run the inference
    prediction = model.predict(data)
    predicted_categories = tf.argmax(prediction, axis=1)
    i = 0

    for x in range(8):
        for y in range(8):

            if str == "all" :
                # Ricerca tutti i pezzi posizionati sulla scacchiera
                positional_matrix[x][y] = get_prediction(predicted_categories,i)
            elif str == "white" :
                #Ricerca solo i pezzi bianchi posizionati sulla scacchiera
                positional_matrix[x][y] = get_white_prediction(predicted_categories,i)
            elif str == "black" :
                #Ricerca solo i pezzi neri posizionati sulla scacchiera
                positional_matrix[x][y] = get_black_prediction(predicted_categories,i)

            i = i + 1


    print_positional_matrix(positional_matrix)
    return positional_matrix

def get_prediction(prediction,i):
    if(prediction[i]==1):
        return 0
    else: return 1

def get_black_prediction(prediction,i):
    if(prediction[i]==0):
        return 1
    else: return 0

def get_white_prediction(prediction,i):
    if(prediction[i]==2):
        return 1
    else: return 0

