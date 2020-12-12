import sys
sys.path.append('../')

from flask_restful import Resource
from model import gen, get_latent_vector
import numpy as np
from PIL import Image
import base64
import io

class ImageApi(Resource):
    def get(self):
        try:
            latent = get_latent_vector()
            image = gen(latent).detach().numpy()[0]
            image = np.transpose(image, (1, 2, 0)) * .5 + .5
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            buff = io.BytesIO()
            image.save(buff, format = "PNG")
            img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
            image_bytes = img_str.encode('utf-8')
            return {"image": img_str}
        except Exception as e:
            print(e)
            return {"error": "A error happened"}, 404
