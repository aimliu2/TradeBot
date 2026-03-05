# Visualize data

from flask import Flask, render_template, send_from_directory

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/script/<path:filename>')
def serve_script(filename):
    return send_from_directory('script', filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    # Running on localhost:8000
    app.run(host='localhost', port=8000, debug=True)
