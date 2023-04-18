from flask import Flask , render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    subprocess.run('python generate.py', shell=True)
    return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug = True)