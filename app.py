from flask import Flask, jsonify
from flask import jsonify
from flask_cors import CORS
from flask_restful import Api, MethodNotAllowed, NotFound
from flask_swagger_ui import get_swaggerui_blueprint
from util.common import domain, port, build_swagger_config_json
from resources.swaggerConfig import SwaggerConfig
from resources.modelResource import modelsPOSTResource
from resources.geminiResource import geminiPOSTResource

# ============================================
# Main
# ============================================
application = Flask(__name__)
app = application
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app, catch_all_404s=True)

# ============================================
# Swagger
# ============================================
build_swagger_config_json()
swaggerui_blueprint = get_swaggerui_blueprint(
    "",
    f'http://{domain}:{port}/swagger-config',
    config={
        'app_name': "Flask API",
        "layout": "BaseLayout",
        "docExpansion": "none"
    },
)
app.register_blueprint(swaggerui_blueprint)

# ============================================
# Error Handler
# ============================================


@app.errorhandler(NotFound)
def handle_method_not_found(e):
    response = jsonify({"message": str(e)})
    response.status_code = 404
    return response

@app.errorhandler(MethodNotAllowed)
def handle_method_not_allowed_error(e):
    response = jsonify({"message": str(e)})
    response.status_code = 405
    return response

# @app.route('/')
# def redirect_to_prefix():
#     if prefix != '':
#         return redirect(prefix)

# ============================================
# Add Resource
# ============================================
# GET swagger config
api.add_resource(SwaggerConfig, '/swagger-config')

# POST methods
api.add_resource(modelsPOSTResource, '/predict')
api.add_resource(geminiPOSTResource, '/gemini-predict')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True)