from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def helloWorld():
    return "Hello World!"

@app.route('/test', methods=['GET', 'POST'])
def parse_request():
    name = request.args.get('name')
    return "Hello {}!".format(name)

if __name__ == '__main__':
    app.run()
    
