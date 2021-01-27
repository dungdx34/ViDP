import enum
from http import HTTPStatus

# from enum import Enum

DETECT_MISSPELLING_MAXIMUM_INPUT_LENGTH = 40000


class ResponseCode(enum.IntEnum):
    OK = 0
    JSON_SYNTAX_ERROR = 1
    INPUT_FORMAT_ERROR = 2
    GENERIC_SERVER_ERROR = 3
    AUTHORIZATION_ERROR = 4
    AUTHENTICATION_ERROR = 5
    DATABASE_ERROR = 6
    INVALID_PARAMETERS = 7
    EMPTY_REQUEST = 8
    EXCEED_MAXIMUM_LENGTH = 9
    SPLIT_TEXT_REQUEST_FAILED = 10
    INVALID_REQUEST_URL = 11


httpStatus = [
    HTTPStatus.OK,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.FORBIDDEN,
    HTTPStatus.UNAUTHORIZED,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.NOT_FOUND,
]

httpResponseMessage = [
    "OK",
    "Invalid json syntax! Cannot parse request!",
    "Request format is invalid!",
    "Internal Server Error",
    "Authorization error",
    "Authentication error",
    "Database error",
    "Invalid parameters",
    "Request text is empty",
    "Input too long! Please limit the length of input to < " + str(DETECT_MISSPELLING_MAXIMUM_INPUT_LENGTH) + "!",
    "Internal Server Error",
    "Request URL is incorrect",
]


def get_http_status(res_code):
    return httpStatus[res_code]


def get_http_response_message(res_code):
    return httpResponseMessage[res_code]
