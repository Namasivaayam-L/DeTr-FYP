import os
import requests
import tensorflow as tf

WEIGHT_NAME_TO_CKPT = {
    "detr": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/checkpoint",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt"
    ]
}

def load_weights(model, weights: str, custom_objects):
    """ Load weight on a given model
    weights are supposed to be sotred in the weight folder at the root of the repository. If weights
    does not exists, but are publicly known, the weight will be download from gcloud.
    """
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    if "ckpt" in weights:
        print('Loading.. weights')
        # l = model.load_weights(filepath=weights)
        # l.expect_partial()
        checkpoint = tf.train.Checkpoint(model=model)
        # Restore values from the checkpoint
        status = checkpoint.restore(tf.train.latest_checkpoint(weights))
        status.expect_partial()
        
    elif 'keras' in weights:
        print('Loading.. model')
        model = tf.keras.models.load_model(weights, custom_objects=custom_objects)
        
    elif weights in WEIGHT_NAME_TO_CKPT:
        print('Downloading Weights')
        wdir = f"weights/{weights}"
        if not os.path.exists(wdir):
            os.makedirs(wdir)
        for f in WEIGHT_NAME_TO_CKPT[weights]:
            fname = f.split("/")[-1]
            if not os.path.exists(os.path.join(wdir, fname)):
                print("Download....", f)
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(wdir, fname), 'wb').write(r.content)
        print("Load weights from", os.path.join(wdir, f"{weights}.ckpt"))
        l = model.load_weights(os.path.join(wdir, f"{weights}.ckpt"))
        l.expect_partial()
    else:
        print("Load weights from", f"{weights}.ckpt")
        l = model.load_weights(f"{weights}.ckpt")
        l.expect_partial()
        
    # else:
    #     raise Exception(f'Cant load the weights: {weights}')