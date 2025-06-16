This is a HTTP service with POST request that use model from Hugging Face to extract parts from non formalized pharm names

Usage:
py .\pharm-nlp-service.py 

Test from client:

curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d "@request.json"
(There is example of request.json  in repository)


