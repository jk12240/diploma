from flask import Flask, render_template, request, send_file
import io
from function import plot_grav_lens_jet, save_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        param1 = float(request.form['param1'])
        param2 = float(request.form['param2'])
        param3 = float(request.form['param3'])
        param4 = float(request.form['param4'])
        param5 = float(request.form['param5'])
        param6 = float(request.form['param6'])
        param7 = float(request.form['param7'])
        param8 = float(request.form['param8'])
    except ValueError as e:
        return f"Invalid input: {e}", 400
    
    fig = plot_grav_lens_jet(param1, param2, param3, param4, param5, param6, param7, param8)
    img_io = save_image(fig)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
