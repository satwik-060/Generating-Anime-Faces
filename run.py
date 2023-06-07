from flask import Flask, render_template, redirect, url_for
from load_model import netG,denorm,device,latent_size
import torch
from torchvision.utils import save_image
from flask_wtf import FlaskForm
from wtforms import SubmitField

app = Flask(__name__)
   
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/generate',methods = ['GET','POST'])
def generate_image():
    latent = torch.randn(1,latent_size,1,1,device = device)
    fake_image = netG.forward(latent)
    fake_fname = 'static/generated_image.png'
    save_image(denorm(fake_image),fake_fname)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 