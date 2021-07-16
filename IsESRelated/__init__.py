# -*- coding: utf-8 -*-
import logging

import azure.functions as func
import json

from . import api_impl as IMP

response_headers = {
    "Content-type": "application/json",
    "Access-Control-Allow-Origin": "*"
}

def get_param_from_request(req: func.HttpRequest, param_name: str):
    # GET
    result = req.params.get(param_name)
    if not result:
        # try POST
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            result = req_body.get(param_name)
    return result

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    subject = get_param_from_request(req, 'Subject')
    body = get_param_from_request(req, 'UniqueBody')
    thread_count = get_param_from_request(req, 'NumRelatedThreads')

    if subject is None or body is None:
        return func.HttpResponse("Must provide Subject and UniqueBody", status_code=200)
    else:
        # let's roll out
        rating = IMP.get_is_es_related_prediction(subject, body)
        rating = float(rating)
        result = {"Rating": rating}
        return func.HttpResponse(json.dumps(result), headers = response_headers)

'''
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
'''