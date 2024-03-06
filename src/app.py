from myapp import app, routes, models
import logging



if __name__ == "__main__":
    logging.basicConfig(filename='/data/logs/python/app.log', level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)
