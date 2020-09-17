from flask import Flask, render_template, request
import tensorflow as tf
#from tensorflow import keras
import keras
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.initializers import glorot_uniform
from PIL import Image
import numpy as np
import numpy as np
from keras.preprocessing import image


#run the application using tensorflow version 1.15.0

config = tf.compat.v1.ConfigProto(
   device_count={'GPU': 1},
   intra_op_parallelism_threads=1,
   allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

print(tf.__version__)


app = Flask(__name__)

#m =  load_model('newmodel.h5',custom_objects={'GlorotUniform': glorot_uniform()})
m =  tf.keras.models.load_model('newmodel.h5',compile=True,custom_objects=None)


m._make_predict_function()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        if request.files["image"]:
            try:
                with session.as_default():
                    with session.graph.as_default():
                        #print("hereeklrfekl")
                        img = image.load_img(request.files['image'].stream,target_size=(64,64))
                        #img = Image.open(request.files['image'].stream)
                        #print(img)
                        #img = img.resize((64,64))
                        #img = img.img_to_array(img)
                        img = np.expand_dims(img, axis=0)
                        result = m.predict(img)
                        #print("hello")
                        print(result)
                        #return render_template('index.html')
                        if result[0][0] == 1:
                            print("OK")
                            return render_template('success.html')
                        else:
                            print("Defect")
                            #add route called defect piece
                            return render_template('defect.html')


            except Exception as ex:
                print(ex.__traceback__.tb_lineno)
        else:
            return render_template('error.html')
    else:
            return render_template('index.html')
    #return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=False)


