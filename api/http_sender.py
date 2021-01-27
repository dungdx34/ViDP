from flask import jsonify
import api.http_response_message as http_response_message


def send_error_response(res_code):
    result = {'code': res_code,
              'message': http_response_message.get_http_response_message(res_code)}
    return jsonify(result), http_response_message.get_http_status(res_code)


def send_http_result(process_result):
    res_code = http_response_message.ResponseCode.OK
    request_result = {'code:': res_code,
                      'message': http_response_message.get_http_response_message(res_code),
                      'result': process_result}
    return jsonify(request_result), http_response_message.get_http_status(res_code)
